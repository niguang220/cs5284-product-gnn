"""Would a more test-like selection set pick a stronger PPR prior? Same PPR family as localprior.py, also scoring the
val nodes at >= 2 hops from train (co-purchase graph of the setting; 790 nodes in normal). Val labels only.
Usage: python ppr_valfar.py ENC SETTING SEED -> exp/g-localprior/runs_valfar/{ENC}_{SETTING}_s{SEED}.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).parent))
from common import (COMBOS, GRAPH, GRAPH_CO, KNN_MSGS, LLMLOG, MSGS, K, BETAS, GAMMAS, hop_distance,  # noqa: E402
                    load_base, normalise_log, te, tr, va, y)
from src.label_shift import fit_temperature  # noqa: E402

ENC, SETTING, SEED = sys.argv[1], sys.argv[2], int(sys.argv[3])
PI_TR = np.maximum(np.bincount(y[tr], minlength=K) / tr.sum(), 1e-4)
PI_TR /= PI_TR.sum()
far = va & (hop_distance(GRAPH_CO[SETTING], tr) >= 2)
A = GRAPH[SETTING]
P = sp.diags(1 / np.maximum(np.asarray(A.sum(1)).ravel(), 1)) @ A
raw = np.log(np.clip(load_base(f"GT@ft-gte-large_s{ENC}", SETTING, SEED), 1e-12, 1))
lp = normalise_log(raw / fit_temperature(np.exp(raw), y, va))
FUSED = {c: normalise_log(lp + LLMLOG[c]) for c in COMBOS}
rows = []


def run_grid(params, lf_fn):
    for combo in COMBOS:
        lf = lf_fn(combo)
        for beta in BETAS:
            zb = lf + beta * MSGS[(SETTING, combo)]
            for gamma in GAMMAS:
                pred = (zb + gamma * KNN_MSGS[combo]).argmax(1)
                rows.append({"enc": ENC, "setting": SETTING, "seed": SEED, "params": params, "llms": "+".join(combo),
                             "beta": beta, "gamma": gamma, "val_acc": (pred[va] == y[va]).mean(),
                             "valfar_acc": (pred[far] == y[far]).mean(), "test_acc": (pred[te] == y[te]).mean()})


run_grid("baseline", lambda c: FUSED[c])
for a in [0.15, 0.5]:
    for iters in [1, 3]:
        pri = {}
        for c in COMBOS:
            q = np.exp(FUSED[c])
            for _ in range(iters):
                q[tr] = np.eye(K)[y[tr]]
                s, h = np.zeros_like(q), q
                for t in range(1, 11):
                    h = P @ h
                    s += a * (1 - a) ** (t - 1) * h
                pi = 0.9 * s / np.maximum(s.sum(1, keepdims=True), 1e-12) + 0.1 * PI_TR
                q = np.exp(normalise_log(FUSED[c] + np.log(pi / PI_TR)))
            pri[c] = np.log(pi / PI_TR)
        for lam in [0.25, 0.5, 1.0]:
            run_grid(f"a={a},it={iters},lam={lam}", lambda c: FUSED[c] + lam * pri[c])
out = Path(__file__).parent / "runs_valfar"
out.mkdir(exist_ok=True)
pd.DataFrame(rows).to_csv(out / f"{ENC}_{SETTING}_s{SEED}.csv", index=False)
print("val far nodes", far.sum())
