"""Graph Transformer arms on the frozen sentence embeddings, over three seeds, with the cold-start test and slices.

Arms: GT (Laplacian positional encodings), GT-noPE (ablation) and GT-noEdge (same architecture trained and
evaluated without edges: separates the gain of the architecture from the gain of attention over neighbours).
Same features, split, seeds, training loop and metrics as scripts/run_baselines.py, so rows are comparable with
arms A/B/SAGE.

Usage: python scripts/run_graph_transformer.py [max_epochs]
A small grid is tried on arm GT, seed 0; the configuration with the best validation accuracy is used for all
arms.
Writes results/3_gnn_text/graph_transformer_{runs,summary,sweep}.csv and graph_transformer_meta.json.
"""
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from src.data import load_products  # noqa: E402
from src.models import GraphTransformer  # noqa: E402
from src.positional import laplacian_pe  # noqa: E402
from src.train import cold_start_edges, evaluate, masked_metrics, set_seed, test_subsets, train  # noqa: E402

EMB_NAME = "original__all-MiniLM-L6-v2"
SEEDS = [0, 1, 2]
NUM_CLASSES = 47
HIDDEN = 256
HEADS = 8
LAYERS = 2
PE_DIM = 16
GRID = [{"lr": lr, "dropout": d} for lr in (1e-3, 5e-3) for d in (0.2, 0.5)]
# arm: (positional-encoding dim, use edges)
ARMS = {"GT": (PE_DIM, True), "GT-noPE": (0, True), "GT-noEdge": (0, False)}

max_epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 500
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

emb_path = REPO / "outputs" / "embeddings" / f"{EMB_NAME}.npy"
emb_meta = json.loads((REPO / "results" / "shared" / "embeddings" / f"{EMB_NAME}.json").read_text())
emb_sha = hashlib.sha256(emb_path.read_bytes()).hexdigest()
if emb_sha != emb_meta["sha256"]:
    raise ValueError(f"{emb_path.name} does not match its manifest in results/shared/embeddings/")

graph, text = load_products()
x_orig = torch.from_numpy(np.load(emb_path))
subsets = test_subsets(graph, text)
cold_edges = cold_start_edges(graph)

# Positional encodings come from whichever graph the model sees: the cold-start run must not use test edges.
t0 = time.time()
pe = {"normal": laplacian_pe(graph.edge_index, graph.num_nodes, PE_DIM),
      "cold": laplacian_pe(cold_edges, graph.num_nodes, PE_DIM)}
print(f"Laplacian PE: {time.time() - t0:.1f} s", flush=True)

graph = graph.to(device)
edges = {"normal": graph.edge_index, "cold": cold_edges.to(device)}
no_edges = graph.clone()
no_edges.edge_index = torch.empty(2, 0, dtype=torch.long, device=device)


def features(pe_dim, setting):
    x = torch.cat([x_orig, pe[setting]], dim=1) if pe_dim else x_orig
    return x.to(device)


def run(pe_dim, cfg, seed, use_edges=True):
    set_seed(seed)
    model = GraphTransformer(x_orig.size(1) + pe_dim, HIDDEN, NUM_CLASSES, dropout=cfg["dropout"],
                             num_layers=LAYERS, heads=HEADS, pe_dim=pe_dim).to(device)
    t0 = time.time()
    model, hist = train(model, features(pe_dim, "normal"), graph if use_edges else no_edges, lr=cfg["lr"],
                        max_epochs=max_epochs)
    return model, hist, (time.time() - t0) / 60


sweep = []
for cfg in GRID:
    _, hist, minutes = run(PE_DIM, cfg, SEEDS[0])
    best = max(hist, key=lambda h: h["val_acc"])
    sweep.append({**cfg, "arm": "GT", "seed": SEEDS[0], "best_epoch": best["epoch"], "val_acc": best["val_acc"],
                  "epochs_run": len(hist), "minutes": round(minutes, 2)})
    print("sweep", sweep[-1], flush=True)
best_cfg = {k: v for k, v in max(sweep, key=lambda r: r["val_acc"]).items() if k in GRID[0]}
print("selected", best_cfg)

rows = []
pred_dir = REPO / "outputs" / "predictions"
pred_dir.mkdir(parents=True, exist_ok=True)
y = graph.y.cpu()
for seed in SEEDS:
    for arm, (pe_dim, use_edges) in ARMS.items():
        model, hist, minutes = run(pe_dim, best_cfg, seed, use_edges)
        best_epoch = int(np.argmax([h["val_acc"] for h in hist]))
        for setting in ["normal", "cold"] if use_edges else ["normal"]:
            eval_edges = edges[setting] if use_edges else no_edges.edge_index
            split_metrics, pred = evaluate(model, features(pe_dim, setting), graph, edge_index=eval_edges)
            pred = pred.cpu()
            np.save(pred_dir / f"{arm}_{setting}_seed{seed}.npy", pred.numpy())
            if setting == "normal":
                normal_metrics = split_metrics
            base = {"arm": arm, "model": "GraphTransformer", "features": "orig+lappe" if pe_dim else "orig",
                    "edges": use_edges,
                    "seed": seed, "setting": setting, "epochs_run": len(hist), "best_epoch": best_epoch,
                    "minutes": round(minutes, 2), "device": device.type}
            for split in ["train", "val"]:
                rows.append({**base, "eval": split, "n": int(graph[f"{split}_mask"].sum()),
                             "acc": split_metrics[f"{split}_acc"], "macro_f1": split_metrics[f"{split}_macro_f1"]})
            for name, mask in subsets.items():
                rows.append({**base, "eval": name, "n": int(mask.sum()), **masked_metrics(y, pred, mask)})
        print(f"seed {seed} arm {arm}: {len(hist)} epochs, best {best_epoch}, {minutes:.1f} min, "
              f"val {normal_metrics['val_acc']:.4f}, test {normal_metrics['test_acc']:.4f}", flush=True)

res = REPO / "results" / "3_gnn_text"
df = pd.DataFrame(rows)
df.to_csv(res / "graph_transformer_runs.csv", index=False)
pd.DataFrame(sweep).to_csv(res / "graph_transformer_sweep.csv", index=False)
summary = df.groupby(["arm", "setting", "eval"])[["acc", "macro_f1"]].agg(["mean", "std"]).round(4)
summary.to_csv(res / "graph_transformer_summary.csv")
meta = {"embeddings": emb_path.name, "embeddings_sha256": emb_sha, "hidden": HIDDEN, "heads": HEADS,
        "layers": LAYERS, "pe": f"Laplacian eigenvectors, k={PE_DIM}, per connected component, random sign flips "
        "in training", "selected": best_cfg, "grid": GRID, "selection": "best validation accuracy, arm GT, seed 0",
        "weight_decay": 5e-4, "max_epochs": max_epochs, "patience": 50, "seeds": SEEDS, "device": device.type,
        "cold_start": "at inference, remove every edge with a test node at either end; PE recomputed on that graph",
        "std": "sample std (ddof=1) over seeds", "code": "scripts/run_graph_transformer.py"}
(res / "graph_transformer_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
with pd.option_context("display.width", 200, "display.max_rows", 200):
    print(summary)
