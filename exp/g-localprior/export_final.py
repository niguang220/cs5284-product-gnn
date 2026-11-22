"""Save the final posteriors of the val-selected PPR-prior pipeline (fusion + PPR local prior + BP [+ kNN]) for every
encoder seed, setting and GNN seed, so other agents can reuse them.
-> outputs/g-localprior/probs/GT@ft-gte-large_s{ENC}+fuse+PPR+BP_{normal,cold}_s{SEED}.npy (float32 [N, 47]) and
   exp/g-localprior/final_ppr.csv (metrics, recomputed with src/train.py: masked_metrics).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

sys.path.insert(0, str(Path(__file__).parent))
from common import (GRAPH, KNN_MSGS, LLMLOG, MSGS, OUT, K, detail, load_base, normalise_log, score, tr, va, y)  # noqa: E402
from src.label_shift import fit_temperature  # noqa: E402

PI_TR = np.maximum(np.bincount(y[tr], minlength=K) / tr.sum(), 1e-4)
PI_TR /= PI_TR.sum()
sel = pd.read_csv(Path(__file__).parent / "summary_selected_per_enc.csv")
sel = sel[sel.variant == "PPR"].set_index(["enc", "setting"])
(OUT / "probs").mkdir(exist_ok=True)
rows = []
for (enc, setting), r in sel.iterrows():
    prm = dict(kv.split("=") for kv in r.params.split(","))
    a, iters, lam = float(prm["a"]), int(prm["it"]), float(prm["lam"])
    combo = tuple(r.llms.split("+"))
    A = GRAPH[setting]
    P = sp.diags(1 / np.maximum(np.asarray(A.sum(1)).ravel(), 1)) @ A
    for seed in range(3):
        raw = np.log(np.clip(load_base(f"GT@ft-gte-large_s{enc}", setting, seed), 1e-12, 1))
        fused = normalise_log(normalise_log(raw / fit_temperature(np.exp(raw), y, va)) + LLMLOG[combo])
        q = np.exp(fused)
        for _ in range(iters):
            q[tr] = np.eye(K)[y[tr]]
            s, h = np.zeros_like(q), q
            for t in range(1, 11):
                h = P @ h
                s += a * (1 - a) ** (t - 1) * h
            pi = 0.9 * s / np.maximum(s.sum(1, keepdims=True), 1e-12) + 0.1 * PI_TR
            q = np.exp(normalise_log(fused + np.log(pi / PI_TR)))
        z = fused + lam * np.log(pi / PI_TR) + r.beta * MSGS[(setting, combo)] + r.gamma * KNN_MSGS[combo]
        z = normalise_log(z)
        np.save(OUT / "probs" / f"GT@ft-gte-large_s{enc}+fuse+PPR+BP_{setting}_s{seed}.npy", np.exp(z).astype(np.float32))
        rows.append({"enc": enc, "setting": setting, "seed": seed, "params": r.params, "llms": r.llms, "beta": r.beta,
                     "gamma": r.gamma, **score(z), **detail(z)})
df = pd.DataFrame(rows)
df.to_csv(Path(__file__).parent / "final_ppr.csv", index=False)
print(df.groupby(["setting", "enc"])[["val_acc", "test_acc", "test_f1"]].mean().round(4))
print(df.groupby("setting")[["val_acc", "test_acc", "test_f1"]].mean().round(4))
