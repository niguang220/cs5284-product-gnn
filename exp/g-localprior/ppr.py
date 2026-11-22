"""PPR-diffused local class priors: ablations of the one variant of localprior.py that helped.

Usage: python ppr.py ENC SETTING SEED  ->  exp/g-localprior/runs_ppr/{ENC}_{SETTING}_s{SEED}.csv

π_i = 0.9 · normalise(Σ_{t=1..10} a(1-a)^(t-1) P^t q) + 0.1 · π_tr, with P = D^-1 A the random walk on a graph, and
the fused log posterior gets + λ · log(π_i / π_tr) (label-shift correction with a node-specific prior) before the
BP / kNN messages of scripts/run_fusion.py. Ablations:
  graph  all = co-purchase ∪ text-kNN5 (cold: co-purchase edges not touching test ∪ kNN), co = co-purchase only,
         knn = text-kNN5 only
  src    fused = the neighbours' fused posteriors (GT x LLM), train nodes one-hot;
         llm = the neighbours' normalised LLM likelihoods (no GT), train nodes one-hot;
         fused-nolab = fused posteriors everywhere (train labels not injected)
  div    ratio = log π_i/π_tr (prior correction); nodiv = log π_i only (plain neighbour-belief smoothing)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).parent))
from common import (BUCKETS, COMBOS, GRAPH, GRAPH_CO, A_KNN, KNN_MSGS, LLMLOG, MSGS, RARE, K, BETAS, GAMMAS,  # noqa: E402
                    load_base, normalise_log, te, tr, va, y)
from src.label_shift import fit_temperature  # noqa: E402

ENC, SETTING, SEED = sys.argv[1], sys.argv[2], int(sys.argv[3])
TAG = f"GT@ft-gte-large_s{ENC}"
PI_TR = np.maximum(np.bincount(y[tr], minlength=K) / tr.sum(), 1e-4)
PI_TR /= PI_TR.sum()
LAB = np.unique(y[te])
YV, YT = y[va], y[te]
BMASK = {b: m[te] for b, m in BUCKETS.items()}


def evaluate(z):
    pred = z.argmax(1)
    pv, pt = pred[va], pred[te]
    tp = np.bincount(YT[pt == YT], minlength=K)[LAB]
    f1 = (2 * tp / np.maximum(np.bincount(pt, minlength=K)[LAB] + np.bincount(YT, minlength=K)[LAB], 1)).mean()
    out = {"val_acc": (pv == YV).mean(), "test_acc": (pt == YT).mean(), "test_f1": f1}
    for b, m in BMASK.items():
        out[f"hop{b}"] = (pt[m] == YT[m]).mean()
    for c, name in RARE.items():
        out[f"rec_{name}"] = (pt[YT == c] == c).mean()
    return out


def rw(A):
    d = np.asarray(A.sum(1)).ravel()
    return sp.diags(1 / np.maximum(d, 1)) @ A


P = {"all": rw(GRAPH[SETTING]), "co": rw(GRAPH_CO[SETTING].astype(np.float64)), "knn": rw(A_KNN.astype(np.float64))}


def ppr(Pm, q, a, T=10):
    s, h = np.zeros_like(q), q
    for t in range(1, T + 1):
        h = Pm @ h
        s += a * (1 - a) ** (t - 1) * h
    has = s.sum(1, keepdims=True) > 1e-12  # nodes with no neighbour in this graph get the train prior
    s = np.where(has, s / np.maximum(s.sum(1, keepdims=True), 1e-12), PI_TR)
    return 0.9 * s + 0.1 * PI_TR


raw = np.log(np.clip(load_base(TAG, SETTING, SEED), 1e-12, 1))
lp = normalise_log(raw / fit_temperature(np.exp(raw), y, va))
FUSED = {c: normalise_log(lp + LLMLOG[c]) for c in COMBOS}
onehot = np.eye(K)[y[tr]]
rows = []


def run_grid(variant, params, lf_fn):
    for combo in COMBOS:
        lf = lf_fn(combo)
        for beta in BETAS:
            zb = lf + beta * MSGS[(SETTING, combo)]
            for gamma in GAMMAS:
                rows.append({"enc": ENC, "setting": SETTING, "seed": SEED, "variant": variant, "params": params,
                             "llms": "+".join(combo), "beta": beta, "gamma": gamma,
                             **evaluate(zb + gamma * KNN_MSGS[combo])})


run_grid("baseline", "-", lambda c: FUSED[c])
for g in ["all", "co", "knn"]:
    for src in ["fused", "llm", "fused-nolab"]:
        for a in [0.15, 0.5]:
            logpi = {}
            for c in COMBOS:
                q = np.exp(FUSED[c]) if src != "llm" else np.exp(normalise_log(LLMLOG[c]))
                if src != "fused-nolab":
                    q[tr] = onehot
                logpi[c] = np.log(ppr(P[g], q, a))
            for div in ["ratio", "nodiv"]:
                for lam in [0.25, 0.5, 1.0]:
                    run_grid(f"PPR-{g}-{src}-{div}", f"a={a},lam={lam}",
                             lambda c: FUSED[c] + lam * (logpi[c] - (np.log(PI_TR) if div == "ratio" else 0)))
    print(g, "done", flush=True)
out = Path(__file__).parent / "runs_ppr"
out.mkdir(exist_ok=True)
pd.DataFrame(rows).to_csv(out / f"{ENC}_{SETTING}_s{SEED}.csv", index=False)
