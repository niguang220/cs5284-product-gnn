"""Graph-local label-shift correction on top of LLM fusion + BP, for one (encoder seed, setting, GNN seed).

Usage: python localprior.py ENC SETTING SEED   ->  exp/g-localprior/runs/{ENC}_{SETTING}_s{SEED}.csv

Every variant replaces the fused log posterior log p(y|x) + log P(g|y) by
    log p(y|x) + log P(g|y) + log π_i(y) / π_tr(y)
(Saerens et al. 2002 prior correction, with a node-specific prior π_i) and then adds the same BP messages as
scripts/run_fusion.py. π_i comes from:
  global   one MAP-EM prior over all unlabelled nodes (Dirichlet strength α toward π_tr)
  louvain  MAP-EM prior per Louvain community of co-purchase ∪ text-kNN edges (Blondel et al. 2008), shrunk
           toward π_tr or toward the global EM prior with strength α (empirical-Bayes style)
  hop      MAP-EM prior per hop-distance-to-train bucket
  bbse     per-community black-box shift estimate from the LLM answers only (Lipton et al. 2018), shrunk with α
  ppr      personalised-PageRank diffusion of the neighbours' fused beliefs (APPNP-style, Klicpera et al. 2019),
           ratio raised to λ, optionally iterated
  oracle   per-community / per-bucket prior from the TRUE labels (upper bound; never selectable)
  calib    no prior; a graph-aware per-node temperature on the GT (GATS / CaGCN spirit) fitted on val NLL
Val nodes are treated as unlabelled in every estimate. Cold start uses only co-purchase edges that do not touch test
nodes, plus text-kNN edges.
"""
import sys
import time
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from scipy.optimize import minimize, nnls
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).parent))
from common import (BUCKETS, C, COMBOS, GRAPH, GRAPH_CO, HOP_REPORT, KNN_MSGS, LLMLOG, MSGS, N, OUT, RARE, K,  # noqa: E402
                    BETAS, GAMMAS, hop_distance, llm, load_base, normalise_log, te, tr, va, y)
from src.label_shift import fit_temperature, temper  # noqa: E402

ENC, SETTING, SEED = sys.argv[1], sys.argv[2], int(sys.argv[3])
TAG = f"GT@ft-gte-large_s{ENC}"
U = ~tr  # nodes whose label is treated as unknown (val + test)
PI_TR = np.maximum(np.bincount(y[tr], minlength=K) / tr.sum(), 1e-4)
PI_TR /= PI_TR.sum()
ALPHAS = [0.0, 10.0, 30.0, 100.0, 300.0, 1000.0]

# ---- fast scoring (same numbers as src/train.py: masked_metrics) ----
LAB = np.unique(y[te])
YV, YT = y[va], y[te]
BMASK = {b: m[te] for b, m in BUCKETS.items()}


def evaluate(z):
    pred = z.argmax(1)
    pv, pt = pred[va], pred[te]
    tp = np.bincount(YT[pt == YT], minlength=K)[LAB]
    npred = np.bincount(pt, minlength=K)[LAB]
    ntrue = np.bincount(YT, minlength=K)[LAB]
    f1 = (2 * tp / np.maximum(npred + ntrue, 1)).mean()
    out = {"val_acc": (pv == YV).mean(), "test_acc": (pt == YT).mean(), "test_f1": f1}
    for b, m in BMASK.items():
        out[f"hop{b}"] = (pt[m] == YT[m]).mean()
    for c, name in RARE.items():
        m = YT == c
        out[f"rec_{name}"] = (pt[m] == c).mean()
    return out


# ---- partitions ----
def communities(setting, res):
    path = OUT / f"louvain_{setting}_r{res}.npy"
    if not path.exists():
        comms = nx.community.louvain_communities(nx.from_scipy_sparse_array(GRAPH[setting]), resolution=res, seed=0)
        lab = np.zeros(N, dtype=np.int64)
        for i, c in enumerate(comms):
            lab[list(c)] = i
        np.save(path, lab)
    return np.load(path)


def hop_buckets(kind):
    if kind == "co":
        d = HOP_REPORT if SETTING == "normal" else hop_distance(GRAPH_CO["cold"], tr)
    else:
        d = hop_distance(GRAPH[SETTING], tr)
    b = np.minimum(d, 4)
    b[np.isinf(d)] = 5
    return b.astype(np.int64)


def indicator(group):
    """Sparse [G, |U|] membership matrix of the unlabelled nodes."""
    g = group[U]
    G = group.max() + 1
    return sp.csr_matrix((np.ones(len(g)), (g, np.arange(len(g)))), shape=(G, len(g)))


# ---- prior estimators ----
def map_em(p, group, alpha, target, iters=300):
    """MAP-EM (Saerens et al. 2002 with a Dirichlet(α·target) prior) of one class prior per group.

    p: fused posteriors under the train prior [N, K]. Returns the per-node log ratio log π_group / π_tr.
    """
    M = indicator(group)
    n = np.asarray(M.sum(1)).ravel()[:, None]
    tgt = np.broadcast_to(target, (M.shape[0], K)) if target.ndim == 1 else target
    pu, gu = p[U], group[U]
    pi = np.tile(PI_TR, (M.shape[0], 1))
    for _ in range(iters):
        q = pu * (pi / PI_TR)[gu]
        q /= q.sum(1, keepdims=True)
        with np.errstate(invalid="ignore", divide="ignore"):
            new = (M @ q + alpha * tgt) / (n + alpha)
        new = np.where(n + alpha > 0, new, PI_TR)  # a group with no unlabelled node keeps the train prior
        new = np.maximum(new, 1e-8)
        new /= new.sum(1, keepdims=True)
        if np.abs(new - pi).max() < 1e-6:
            pi = new
            break
        pi = new
    return np.log(pi / PI_TR)[group], pi


def bbse_group(combo, group, alpha):
    """Per-group BBSE from the LLM answers alone: NNLS of [C_k] π = [answer frequencies], shrunk toward π_tr."""
    M = indicator(group)
    n = np.asarray(M.sum(1)).ravel()
    seen = np.bincount(y[tr | va], minlength=K) > 0
    A = np.vstack([C[k][:, seen] for k in combo])
    pis = []
    for g in range(M.shape[0]):
        idx = np.flatnonzero(U)[M[g].indices]
        if len(idx) == 0:
            pis.append(PI_TR)
            continue
        mu = np.concatenate([np.bincount(llm[k][idx] % (K + 1), minlength=K + 1) / len(idx) for k in combo])
        w, _ = nnls(A, mu)
        pi = np.zeros(K)
        pi[seen] = w
        pi = pi / max(pi.sum(), 1e-12)
        pi = (n[g] * pi + alpha * PI_TR) / (n[g] + alpha)
        pis.append(np.maximum(pi, 1e-8) / np.maximum(pi, 1e-8).sum())
    pis = np.array(pis)
    return np.log(pis / PI_TR)[group]


def oracle_group(group):
    """Prior of the true labels of the unlabelled nodes in each group (+1 pseudo-count on π_tr): upper bound."""
    M = indicator(group)
    cnt = M @ np.eye(K)[y[U]]
    pi = (cnt + PI_TR) / (cnt.sum(1, keepdims=True) + 1)
    return np.log(pi / PI_TR)[group]


P_RW = {}


def ppr_prior(q, a, T=10):
    """Σ_{t=1..T} a(1-a)^(t-1) P^t q with P = D^-1 A: the PPR-weighted label mix around each node, itself excluded
    at t=0 (its own evidence is already in the posterior)."""
    if SETTING not in P_RW:
        A = GRAPH[SETTING]
        d = np.asarray(A.sum(1)).ravel()
        P_RW[SETTING] = sp.diags(1 / np.maximum(d, 1)) @ A
    P = P_RW[SETTING]
    s, h = np.zeros_like(q), q
    for t in range(1, T + 1):
        h = P @ h
        s += a * (1 - a) ** (t - 1) * h
    s /= np.maximum(s.sum(1, keepdims=True), 1e-12)
    return 0.9 * s + 0.1 * PI_TR


# ---- graph-aware temperature ----
def node_features(logp):
    A = GRAPH_CO[SETTING]
    deg = np.asarray(A.sum(1)).ravel()
    hop = hop_buckets("all")
    pred = logp.argmax(1)
    Ag = GRAPH[SETTING]
    rows, cols = Ag.nonzero()
    agree = np.bincount(rows, weights=(pred[rows] == pred[cols]).astype(float), minlength=N) / np.maximum(
        np.bincount(rows, minlength=N), 1)
    p = np.exp(logp)
    ent = -(p * logp).sum(1) / np.log(K)
    f = np.stack([np.ones(N), np.log1p(deg), (hop == 2), (hop >= 3), agree, ent], 1).astype(float)
    return f


def fit_node_temperature(logp, f, l2=1e-2):
    """T_i = exp(w·f_i), w minimising val NLL of softmax(log p / T_i) + l2·|w[1:]|²."""
    lv, fv, yv = logp[va], f[va], y[va]

    def obj(w):
        t = np.exp(fv @ w)[:, None]
        z = lv / t
        z = z - z.max(1, keepdims=True)
        lse = np.log(np.exp(z).sum(1))
        return -(z[np.arange(len(z)), yv] - lse).mean() + l2 * (w[1:] ** 2).sum()

    w0 = np.zeros(f.shape[1])
    w0[0] = np.log(fit_temperature(np.exp(logp), y, va))
    w = minimize(obj, w0, method="L-BFGS-B").x
    return normalise_log(logp / np.exp(f @ w)[:, None]), w


# ---- run ----
t0 = time.time()
raw = np.log(np.clip(load_base(TAG, SETTING, SEED), 1e-12, 1))
lp = normalise_log(raw / fit_temperature(np.exp(raw), y, va))
rows = []


def run_grid(variant, params, fused_fn):
    """fused_fn(combo) -> log posterior before BP; BP / kNN messages added as in run_fusion.py."""
    for combo in COMBOS:
        lf = fused_fn(combo)
        for beta in BETAS:
            zb = lf + beta * MSGS[(SETTING, combo)]
            for gamma in GAMMAS:
                z = zb + gamma * KNN_MSGS[combo]
                rows.append({"enc": ENC, "setting": SETTING, "seed": SEED, "variant": variant, "params": params,
                             "llms": "+".join(combo), "beta": beta, "gamma": gamma, **evaluate(z)})


FUSED = {c: normalise_log(lp + LLMLOG[c]) for c in COMBOS}
PF = {c: np.exp(FUSED[c]) for c in COMBOS}
run_grid("baseline", "-", lambda c: FUSED[c])

# global and local MAP-EM
GLOBAL = {}
for alpha in ALPHAS:
    for c in COMBOS:
        GLOBAL[(c, alpha)] = map_em(PF[c], np.zeros(N, dtype=np.int64), alpha * 100, PI_TR)
    run_grid("global-EM", f"a={alpha * 100:g}", lambda c: FUSED[c] + GLOBAL[(c, alpha)][0])
GROUPS = {f"louvain-r{r}": communities(SETTING, r) for r in [1, 10]}
GROUPS.update({"hop-co": hop_buckets("co"), "hop-all": hop_buckets("all")})
for gname, group in GROUPS.items():
    for alpha in ALPHAS:
        for target in ["tr", "glob"]:
            if target == "glob" and alpha == 0:
                continue
            lr = {}
            for c in COMBOS:
                tgt = PI_TR if target == "tr" else GLOBAL[(c, 0.0)][1][0]
                lr[c] = map_em(PF[c], group, alpha, tgt)[0]
            run_grid(f"EM-{gname}", f"a={alpha:g},to={target}", lambda c: FUSED[c] + lr[c])
    run_grid(f"oracle-{gname}", "-", lambda c, g=group: FUSED[c] + oracle_group(g))
print("EM done", time.time() - t0, flush=True)

# BBSE per community from the LLM answers alone (no self-reinforcement)
for gname in ["louvain-r1", "louvain-r10", "hop-all"]:
    for alpha in ALPHAS:
        lr = {c: bbse_group(c, GROUPS[gname], alpha) for c in COMBOS}
        run_grid(f"BBSE-{gname}", f"a={alpha:g}", lambda c: FUSED[c] + lr[c])
print("BBSE done", time.time() - t0, flush=True)

# PPR diffusion of beliefs
onehot_tr = np.eye(K)[y[tr]]
for a in [0.15, 0.5]:
    for iters in [1, 3]:
        priors = {}
        for c in COMBOS:
            q = PF[c].copy()
            for _ in range(iters):
                q[tr] = onehot_tr
                pi = ppr_prior(q, a)
                q = np.exp(normalise_log(FUSED[c] + np.log(pi / PI_TR)))
            priors[c] = np.log(pi / PI_TR)
        for lam in [0.25, 0.5, 1.0]:
            run_grid("PPR", f"a={a},it={iters},lam={lam}", lambda c: FUSED[c] + lam * priors[c])
print("PPR done", time.time() - t0, flush=True)

# graph-aware temperature
f = node_features(lp)
for l2 in [1e-3, 1e-2, 1e-1]:
    lpc, w = fit_node_temperature(raw, f, l2)
    run_grid("calib-node-T", f"l2={l2},w={np.round(w, 2).tolist()}", lambda c: normalise_log(lpc + LLMLOG[c]))
print("calib done", time.time() - t0, flush=True)

out = Path(__file__).parent / "runs"
out.mkdir(exist_ok=True)
pd.DataFrame(rows).to_csv(out / f"{ENC}_{SETTING}_s{SEED}.csv", index=False)
print("total", time.time() - t0)
