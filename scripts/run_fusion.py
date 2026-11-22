"""LLM fusion and belief propagation of neighbours' LLM answers on top of saved base-model probabilities.

Usage: python scripts/run_fusion.py TAG [TAG ...] [--knn-emb NAME]
  TAG names outputs/probs/{TAG}_{normal|cold}_s{seed}.npy; --knn-emb picks the embedding in outputs/embeddings used
  for the text-kNN edges (default: MiniLM).
Per model and setting: temperature on val NLL; fuse with the LLM answers through their confusion matrices
(train+val labels); then add β·Σ_j log m_j from neighbours (src/label_shift.py). Which LLMs to fuse and β are chosen
on val accuracy. "+kNN" also passes messages along text-similarity edges (each node's 5 nearest MiniLM neighbours,
weight γ), which a cold-start node still has. Test labels are only read for reporting.
LLM answers: GPT-3.5 (outputs/tape_gpt_preds/ogbn-products.csv), plus outputs/llm/qwen35_9b.npy
(scripts/llm_zero_shot.py) and any outputs/llm_answers/*.npy if present.
Writes results/3_gnn_text/fusion[_TAGS]_{runs,summary}.csv and _meta.json; exp/main/show_fusion.py prints the val-selected rows.
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from src.data import load_products  # noqa: E402
from src.label_shift import (K, compatibility, confusion, fit_temperature, fuse, neighbour_log_messages,  # noqa: E402
                             temper)
from src.improve import nearest_in_pool  # noqa: E402
from src.train import cold_start_edges, masked_metrics  # noqa: E402

BETAS = [0.0, 0.25, 0.5, 1.0]
GAMMAS = [0.25, 0.5, 1.0]
KNN = 5
graph, _ = load_products()
y, N = graph.y.numpy(), graph.num_nodes
tr, va, te = graph.train_mask.numpy(), graph.val_mask.numpy(), graph.test_mask.numpy()
yt = torch.from_numpy(y)

llm = {"gpt": np.array([int(r[0]) for r in csv.reader(open(REPO / "outputs/tape_gpt_preds/ogbn-products.csv"))])}
for name, path in [("qwen", REPO / "outputs/llm/qwen35_9b.npy")] + [
        (p.stem, p) for p in sorted((REPO / "outputs/llm_answers").glob("*.npy"))]:
    if path.exists():
        llm[name] = np.load(path)
C = {k: confusion(a, y, tr | va) for k, a in llm.items()}
# GPT alone, or GPT plus one other LLM: several answer sets from the same LLM are not independent observations.
combos = [("gpt",)] + [("gpt", k) for k in llm if k != "gpt"]
H = compatibility(graph.edge_index.numpy(), y, tr)
edges = {"normal": graph.edge_index.numpy(), "cold": cold_start_edges(graph).numpy()}


def evidence(combo):
    """What a neighbour tells about its own label: its train label, else its LLM answers' likelihood."""
    e = np.ones((N, K))
    for k in combo:
        e *= C[k][llm[k] % (K + 1)]
    e /= e.sum(1, keepdims=True)
    e[tr] = np.eye(K)[y[tr]]  # val labels are never used as evidence
    return e


msgs = {(s, c): neighbour_log_messages(edges[s], N, evidence(c), H) for s in edges for c in combos}

args = sys.argv[1:]
knn_emb = "original__all-MiniLM-L6-v2"
if "--knn-emb" in args:
    i = args.index("--knn-emb")
    knn_emb = args[i + 1]
    del args[i:i + 2]
# Text-similarity edges j -> i from each node's KNN nearest neighbours (cosine); available in cold start.
x = torch.from_numpy(np.load(REPO / f"outputs/embeddings/{knn_emb}.npy")).float()
x = torch.nn.functional.normalize(x, dim=1)
knn_edges = nearest_in_pool(torch.arange(N), x, torch.arange(N), KNN).numpy()
H_knn = compatibility(knn_edges, y, tr)
knn_msgs = {c: neighbour_log_messages(knn_edges, N, evidence(c), H_knn) for c in combos}


def scores(q):
    pred = torch.from_numpy(q.argmax(1))
    v, t = masked_metrics(yt, pred, torch.from_numpy(va)), masked_metrics(yt, pred, torch.from_numpy(te))
    return {"val_acc": v["acc"], "test_acc": t["acc"], "test_f1": t["macro_f1"]}


tags = args
stem = "fusion" if tags == ["MLP", "GCN", "SAGE", "GT"] else "fusion_" + "_".join(tags)
if knn_emb != "original__all-MiniLM-L6-v2":
    stem += f"_knn-{knn_emb}"
rows = []
for tag in tags:
    for setting in ["normal", "cold"]:
        for seed in range(3):
            path = REPO / f"outputs/probs/{tag}_{setting}_s{seed}.npy"
            if not path.exists():
                path = REPO / f"outputs/probs/{tag}_normal_s{seed}.npy"  # MLP: no edges, same in both settings
                if not path.exists():
                    continue
            p = np.load(path).astype(np.float64)
            p = temper(p, fit_temperature(p, y, va))
            base = {"model": tag, "setting": setting, "seed": seed}
            rows.append({**base, "method": "base", "llms": "", "beta": np.nan, "gamma": np.nan, **scores(p)})
            for combo in combos:
                logf = np.log(fuse(p, [llm[k] for k in combo], [C[k] for k in combo]))
                for beta in BETAS:
                    z = logf + beta * msgs[(setting, combo)]
                    rows.append({**base, "method": "fuse" if beta == 0 else "fuse+BP", "llms": "+".join(combo),
                                 "beta": beta, "gamma": 0.0, **scores(np.exp(z - z.max(1, keepdims=True)))})
                    for gamma in GAMMAS:
                        zk = z + gamma * knn_msgs[combo]
                        rows.append({**base, "method": "fuse+BP+kNN", "llms": "+".join(combo), "beta": beta,
                                     "gamma": gamma, **scores(np.exp(zk - zk.max(1, keepdims=True)))})

df = pd.DataFrame(rows)
df.to_csv(REPO / f"results/3_gnn_text/{stem}_runs.csv", index=False)
key = ["model", "setting", "method", "llms", "beta", "gamma"]
s = df.groupby(key, dropna=False)[["val_acc", "test_acc", "test_f1"]].agg(["mean", "std"]).round(4)
# val-selected configurations: best LLM combo for "fuse", best (combo, β>0) for "fuse+BP"
sel = []
for (m, st, meth), g in s.groupby(level=[0, 1, 2]):
    sel.append(g[("val_acc", "mean")].idxmax())
s.to_csv(REPO / f"results/3_gnn_text/{stem}_summary.csv")
with pd.option_context("display.width", 200, "display.max_rows", 500):
    print(s.loc[sel])
(REPO / f"results/3_gnn_text/{stem}_meta.json").write_text(json.dumps(
    {"llms": list(llm), "betas": BETAS, "selection": "val accuracy, mean over seeds", "knn": KNN, "knn_emb": knn_emb, "gammas": GAMMAS, "compatibility":
     "train-train edges (separately for co-purchase and kNN edges), 5 pseudo-edges toward 0.8 I + 0.2 uniform", "temperature": "val NLL per seed"}, indent=1))
