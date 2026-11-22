"""Combine the two small positives of the graph sprint: the NAGphormer backbone (g-arch) and the PPR-diffused local
prior (g-localprior), on top of LLM fusion + BP. Everything is chosen on val (mean over the 3 GNN seeds) per encoder
seed and setting; lam = 0 is the plain fusion + BP pipeline, so val may also decline the prior.

Usage: python exp/main/combine_graph.py  ->  exp/main/combine_graph_{runs,selected}.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "g-localprior"))
from common import (BETAS, COMBOS, GAMMAS, GRAPH, KNN_MSGS, LLMLOG, MSGS, K, detail, load_base, normalise_log,  # noqa: E402
                    score, tr, va, y)
from src.label_shift import fit_temperature  # noqa: E402

PI_TR = np.maximum(np.bincount(y[tr], minlength=K) / tr.sum(), 1e-4)
PI_TR /= PI_TR.sum()
ONEHOT = np.eye(K)[y[tr]]


def ppr_prior(fused, P, a, iters):
    """π_i from PPR diffusion of the neighbours' fused beliefs (train nodes one-hot), as exp/g-localprior/export_final.py."""
    q = np.exp(fused)
    for _ in range(iters):
        q[tr] = ONEHOT
        s, h = np.zeros_like(q), q
        for t in range(1, 11):
            h = P @ h
            s += a * (1 - a) ** (t - 1) * h
        pi = 0.9 * s / np.maximum(s.sum(1, keepdims=True), 1e-12) + 0.1 * PI_TR
        q = np.exp(normalise_log(fused + np.log(pi / PI_TR)))
    return np.log(pi / PI_TR)


rows = []
for model in ["GT", "garch-NAGphormer"]:
    for enc in range(3):
        for setting in ["normal", "cold"]:
            A = GRAPH[setting]
            P = sp.diags(1 / np.maximum(np.asarray(A.sum(1)).ravel(), 1)) @ A
            for seed in range(3):
                raw = np.log(np.clip(load_base(f"{model}@ft-gte-large_s{enc}", setting, seed), 1e-12, 1))
                lp = normalise_log(raw / fit_temperature(np.exp(raw), y, va))
                for combo in COMBOS:
                    fused = normalise_log(lp + LLMLOG[combo])
                    priors = {("-", 0): np.zeros_like(fused)}
                    for a in [0.15, 0.5]:
                        for it in [1, 3]:
                            priors[(f"a={a},it={it}", 1)] = ppr_prior(fused, P, a, it)
                    for (prm, _), lr in priors.items():
                        for lam in ([0.0] if prm == "-" else [0.25, 0.5, 1.0]):
                            for beta in BETAS:
                                zb = fused + lam * lr + beta * MSGS[(setting, combo)]
                                for gamma in GAMMAS:
                                    z = normalise_log(zb + gamma * KNN_MSGS[combo])
                                    rows.append({"model": model, "enc": enc, "setting": setting, "seed": seed,
                                                 "prior": prm, "lam": lam, "llms": "+".join(combo), "beta": beta,
                                                 "gamma": gamma, **score(z)})
            print(model, enc, setting, flush=True)
df = pd.DataFrame(rows)
out = Path(__file__).parent
df.to_csv(out / "combine_graph_runs.csv", index=False)
keys = ["model", "enc", "setting", "prior", "lam", "llms", "beta", "gamma"]
g = df.groupby(keys)[["val_acc", "test_acc", "test_f1"]].mean().reset_index()
g["with_prior"] = g.lam > 0
sel = g.loc[g.groupby(["model", "enc", "setting", "with_prior"]).val_acc.idxmax()]
sel.to_csv(out / "combine_graph_selected.csv", index=False)
print(sel.round(4).to_string(index=False))
# "val decides": the prior is used only if its best config beats the best lam = 0 config on val
best = g.loc[g.groupby(["model", "enc", "setting"]).val_acc.idxmax()]
print(best.round(4).to_string(index=False))
print(best.groupby(["model", "setting"])[["val_acc", "test_acc", "test_f1"]].agg(["mean", "std"]).round(4))
print(sel.groupby(["model", "setting", "with_prior"])[["val_acc", "test_acc", "test_f1"]].agg(["mean", "std"]).round(4))
