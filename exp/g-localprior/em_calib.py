"""Does EM fail because the fused posterior is over-confident? Calibrate it before EM (Alexandari et al. 2020:
EM with bias-corrected temperature scaling, BCTS) and measure how far each estimated regional prior is from the
true one.

Usage: python em_calib.py ENC SETTING SEED -> exp/g-localprior/runs_emcal2/{ENC}_{SETTING}_s{SEED}.csv and
       exp/g-localprior/prior_error2/{ENC}_{SETTING}_s{SEED}.csv (estimation error; test labels for analysis only)

calibration  none = fused posterior as is; T = one temperature on the fused log posterior (val NLL);
             BCTS = temperature + per-class bias (val NLL, L2 1e-2 on the biases)
EM updates only the classes that occur in train (runs_emcal2; the first run, runs_emcal, updated all 47 and
collapsed after calibration). The prior ratio estimated on the calibrated posteriors is added to the uncalibrated fused log posterior (the
quantity whose BP weights β, γ the grid selects), so only the prior estimate changes.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).parent))
from common import (BUCKETS, COMBOS, GRAPH, KNN_MSGS, LLMLOG, MSGS, OUT, RARE, K, N, BETAS, GAMMAS,  # noqa: E402
                    hop_distance, load_base, normalise_log, te, tr, va, y)
from src.label_shift import fit_temperature  # noqa: E402

ENC, SETTING, SEED = sys.argv[1], sys.argv[2], int(sys.argv[3])
TAG = f"GT@ft-gte-large_s{ENC}"
U = ~tr
SEEN = np.bincount(y[tr], minlength=K) > 0
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


def indicator(group):
    g = group[U]
    return sp.csr_matrix((np.ones(len(g)), (g, np.arange(len(g)))), shape=(group.max() + 1, len(g)))


def map_em(p, group, alpha, iters=300):
    M = indicator(group)
    n = np.asarray(M.sum(1)).ravel()[:, None]
    pu, gu = p[U], group[U]
    pi = np.tile(PI_TR, (M.shape[0], 1))
    for _ in range(iters):
        q = pu * (pi / PI_TR)[gu]
        q /= q.sum(1, keepdims=True)
        with np.errstate(invalid="ignore", divide="ignore"):
            new = (M @ q + alpha * PI_TR) / (n + alpha)
        new = np.where(n + alpha > 0, new, PI_TR)
        new = np.maximum(new, 1e-8)
        # classes with no train node keep their floor prior: p_tr(y|x) carries no information about them, and
        # dividing by the floor lets EM run away into them once the posterior is softened
        new[:, ~SEEN] = PI_TR[~SEEN]
        new[:, SEEN] *= (1 - PI_TR[~SEEN].sum()) / new[:, SEEN].sum(1, keepdims=True)
        done = np.abs(new - pi).max() < 1e-6
        pi = new
        if done:
            break
    return pi


def fit_bcts(z, bias=True, l2=1e-2):
    """argmin over (log T, b) of val NLL of softmax(z / T + b)."""
    zv, yv = z[va], y[va]

    def obj(w):
        s = zv / np.exp(w[0]) + (w[1:] if bias else 0)
        s = s - s.max(1, keepdims=True)
        nll = -(s[np.arange(len(s)), yv] - np.log(np.exp(s).sum(1))).mean()
        return nll + (l2 * (w[1:] ** 2).sum() if bias else 0)

    w = minimize(obj, np.zeros(1 + (K if bias else 0)), method="L-BFGS-B").x
    return normalise_log(z / np.exp(w[0]) + (w[1:] if bias else 0))


def tv_error(pi_group, group):
    """Test-node-weighted TV between each group's estimated prior and the true label mix of its test nodes."""
    err, w = 0.0, 0
    for g in np.unique(group[te]):
        m = te & (group == g)
        true = np.bincount(y[m], minlength=K) / m.sum()
        err += m.sum() * 0.5 * np.abs(pi_group[g] - true).sum()
        w += m.sum()
    return err / w


raw = np.log(np.clip(load_base(TAG, SETTING, SEED), 1e-12, 1))
lp = normalise_log(raw / fit_temperature(np.exp(raw), y, va))
FUSED = {c: normalise_log(lp + LLMLOG[c]) for c in COMBOS}
hop = hop_distance(GRAPH[SETTING], tr)
hopb = np.where(np.isinf(hop), 5, np.minimum(hop, 4)).astype(np.int64)
GROUPS = {"global": np.zeros(N, dtype=np.int64), "hop-all": hopb,
          "louvain-r1": np.load(OUT / f"louvain_{SETTING}_r1.npy"),
          "louvain-r10": np.load(OUT / f"louvain_{SETTING}_r10.npy")}
rows, errs = [], []


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
for cal in ["none", "T", "BCTS"]:
    CAL = {c: FUSED[c] if cal == "none" else fit_bcts(FUSED[c], bias=cal == "BCTS") for c in COMBOS}
    for gname, group in GROUPS.items():
        for alpha in [0.0, 100.0, 1000.0]:
            a = alpha * 100 if gname == "global" else alpha
            lr = {}
            for c in COMBOS:
                pi = map_em(np.exp(CAL[c]), group, a)
                lr[c] = np.log(pi / PI_TR)[group]
                if c == ("gpt",):
                    errs.append({"enc": ENC, "setting": SETTING, "seed": SEED, "calib": cal, "group": gname,
                                 "alpha": a, "tv_est": tv_error(pi, group),
                                 "tv_train_prior": tv_error(np.tile(PI_TR, (group.max() + 1, 1)), group)})
            run_grid(f"EMcal-{gname}", f"cal={cal},a={a:g}", lambda c: FUSED[c] + lr[c])
    # the fused prediction histogram per group (no EM) and the true prior: reference points for the error
    if cal == "none":
        for gname, group in GROUPS.items():
            M = indicator(group)
            hist = M @ np.eye(K)[FUSED[("gpt",)][U].argmax(1)]
            hist = hist / np.maximum(hist.sum(1, keepdims=True), 1)
            errs.append({"enc": ENC, "setting": SETTING, "seed": SEED, "calib": "argmax-hist", "group": gname,
                         "alpha": np.nan, "tv_est": tv_error(hist, group), "tv_train_prior": np.nan})
    print(cal, "done", flush=True)
for d, obj in [("runs_emcal2", rows), ("prior_error2", errs)]:
    (Path(__file__).parent / d).mkdir(exist_ok=True)
    pd.DataFrame(obj).to_csv(Path(__file__).parent / d / f"{ENC}_{SETTING}_s{SEED}.csv", index=False)
