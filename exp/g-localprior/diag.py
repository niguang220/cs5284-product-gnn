"""Diagnostics: reproduce the fusion baseline, node counts and label TV per hop bucket, and reliability (ECE,
mean confidence vs accuracy) per hop bucket for the calibrated GT and the fused posterior. Test labels: analysis only.
"""
import time

import numpy as np

from common import (BUCKETS, COMBOS, HOP_REPORT, LLMLOG, K, calibrated_logp, grid, load_base, normalise_log, score,
                    te, tr, va, y)

import pandas as pd

pi_tr = np.bincount(y[tr], minlength=K) / tr.sum()
print("hop counts (val / test) and label TV(test bucket, train prior)")
for b, m in BUCKETS.items():
    pt = np.bincount(y[m & te], minlength=K) / max((m & te).sum(), 1)
    print(b, (m & va).sum(), (m & te).sum(), round(0.5 * np.abs(pt - pi_tr).sum(), 3))
print("unreachable test", (np.isinf(HOP_REPORT) & te).sum(), "val", (np.isinf(HOP_REPORT) & va).sum())


def ece(logp, mask, bins=15):
    p = np.exp(logp[mask])
    conf, pred = p.max(1), p.argmax(1)
    acc = pred == y[mask]
    e = 0
    edges = np.linspace(0, 1, bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs(acc[m].mean() - conf[m].mean())
    return e, conf.mean(), acc.mean()


rows = []
for setting in ["normal", "cold"]:
    for s in range(3):
        lp = calibrated_logp(load_base("GT@ft-gte-large_s0", setting, s))
        lf = normalise_log(lp + LLMLOG[("gpt",)])
        for name, z in [("GT", lp), ("fused", lf)]:
            for b, m in [("val", va), ("test", te)] + [(f"test hop{b}", m & te) for b, m in BUCKETS.items()] + \
                    [(f"val hop{b}", m & va) for b, m in BUCKETS.items() if (m & va).sum() > 30]:
                e, c, a = ece(z, m)
                rows.append({"setting": setting, "seed": s, "model": name, "slice": b, "ece": e, "conf": c, "acc": a})
df = pd.DataFrame(rows).groupby(["setting", "model", "slice"])[["ece", "conf", "acc"]].mean().round(3)
print(df.to_string())
df.to_csv("exp/g-localprior/reliability.csv")

t = time.time()
res = []
for setting in ["normal", "cold"]:
    for s in range(3):
        lp = calibrated_logp(load_base("GT@ft-gte-large_s0", setting, s))
        for cfg, z in grid(lambda c: lp + LLMLOG[c], setting):
            res.append({"setting": setting, "seed": s, **cfg, **score(z)})
r = pd.DataFrame(res)
for setting in ["normal", "cold"]:
    d = r[(r.setting == setting) & ((r.beta > 0) | (r.gamma > 0))]
    g = d.groupby(["llms", "beta", "gamma"])[["val_acc", "test_acc", "test_f1"]].mean()
    print(setting, g.val_acc.idxmax(), g.loc[g.val_acc.idxmax()].round(4).to_dict())
print("grid time", time.time() - t)
