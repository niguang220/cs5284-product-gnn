"""Calibration of the GT alone: one temperature vs a graph-aware per-node temperature (GATS / CaGCN spirit), test NLL
and ECE by hop bucket (test labels for evaluation only; both fitted on val NLL).
-> exp/g-localprior/calibration_by_hop.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).parent))
from common import BUCKETS, GRAPH, GRAPH_CO, K, N, hop_distance, load_base, normalise_log, te, tr, va, y  # noqa: E402
from src.label_shift import fit_temperature  # noqa: E402


def features(logp, setting):
    deg = np.asarray(GRAPH_CO[setting].sum(1)).ravel()
    hop = hop_distance(GRAPH[setting], tr)
    pred = logp.argmax(1)
    r, c = GRAPH[setting].nonzero()
    agree = np.bincount(r, weights=(pred[r] == pred[c]).astype(float), minlength=N) / np.maximum(
        np.bincount(r, minlength=N), 1)
    ent = -(np.exp(logp) * logp).sum(1) / np.log(K)
    return np.stack([np.ones(N), np.log1p(deg), hop == 2, hop >= 3, agree, ent], 1).astype(float)


def fit(raw, f, l2):
    lv, fv, yv = raw[va], f[va], y[va]

    def obj(w):
        z = lv / np.exp(fv @ w)[:, None]
        z = z - z.max(1, keepdims=True)
        return -(z[np.arange(len(z)), yv] - np.log(np.exp(z).sum(1))).mean() + l2 * (w[1:] ** 2).sum()

    w0 = np.zeros(f.shape[1])
    w0[0] = np.log(fit_temperature(np.exp(raw), y, va))
    return minimize(obj, w0, method="L-BFGS-B").x


def metrics(logp, m):
    p = np.exp(logp[m])
    conf, acc = p.max(1), p.argmax(1) == y[m]
    bins = np.minimum((conf * 15).astype(int), 14)
    ece = sum(abs(acc[bins == b].mean() - conf[bins == b].mean()) * (bins == b).mean() for b in range(15)
              if (bins == b).any())
    return {"nll": -logp[m, y[m]].mean(), "ece": ece, "conf": conf.mean(), "acc": acc.mean()}


rows = []
for setting in ["normal", "cold"]:
    for enc in range(3):
        for s in range(3):
            raw = np.log(np.clip(load_base(f"GT@ft-gte-large_s{enc}", setting, s), 1e-12, 1))
            one = normalise_log(raw / fit_temperature(np.exp(raw), y, va))
            f = features(one, setting)
            w = fit(raw, f, 1e-2)
            node = normalise_log(raw / np.exp(f @ w)[:, None])
            for name, z in [("single T", one), ("node T", node)]:
                for b, m in [("val", va), ("test", te)] + [(f"test hop{b}", mm & te) for b, mm in BUCKETS.items()]:
                    rows.append({"setting": setting, "enc": enc, "seed": s, "calib": name, "slice": b, **metrics(z, m)})
            rows.append({"setting": setting, "enc": enc, "seed": s, "calib": "node T weights", "slice": str(np.round(w, 3))})
df = pd.DataFrame(rows)
df.to_csv(Path(__file__).parent / "calibration_by_hop.csv", index=False)
d = df[df.calib != "node T weights"]
print(d.groupby(["setting", "slice", "calib"])[["nll", "ece", "conf", "acc"]].mean().round(4).to_string())
print(df[df.calib == "node T weights"][["setting", "enc", "seed", "slice"]].head(6).to_string())
