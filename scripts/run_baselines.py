"""Arms A/B/K/L and GraphSAGE over three seeds, with the cold-start test and the test-set slices.

Usage: python scripts/run_baselines.py [max_epochs]
Needs outputs/embeddings/original__all-MiniLM-L6-v2.npy (scripts/encode_original.py, or the shared copy);
its SHA-256 must match results/shared/embeddings/original__all-MiniLM-L6-v2.json.
Writes results/3_gnn_text/baselines_runs.csv (one row per run x evaluation), baselines_summary.csv and baselines_meta.json.
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
from src.models import GCN, MLP, SAGE  # noqa: E402
from src.train import cold_start_edges, evaluate, masked_metrics, set_seed, test_subsets, train  # noqa: E402

EMB_NAME = "original__all-MiniLM-L6-v2"
THREADS = 4  # results are bit-identical only with the same thread count (see set_seed)
SEEDS = [0, 1, 2]
NUM_CLASSES = 47
HIDDEN = 256
# arm: (model class, append random vectors). K/L are the capacity control for later concatenation arms.
ARMS = {"A": (MLP, False), "B": (GCN, False), "K": (MLP, True), "L": (GCN, True), "SAGE": (SAGE, False)}

max_epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 500
torch.set_num_threads(THREADS)
WEIGHTS_DIR = REPO / "outputs" / "weights"  # reused by later scripts; not kept in git
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

emb_path = REPO / "outputs" / "embeddings" / f"{EMB_NAME}.npy"
emb_meta = json.loads((REPO / "results" / "shared" / "embeddings" / f"{EMB_NAME}.json").read_text())
emb_sha = hashlib.sha256(emb_path.read_bytes()).hexdigest()
if emb_sha != emb_meta["sha256"]:
    raise ValueError(f"{emb_path.name} does not match its manifest in results/shared/embeddings/")

graph, text = load_products()
x_orig = torch.from_numpy(np.load(emb_path))
assert x_orig.shape == (graph.num_nodes, 384), x_orig.shape
assert torch.isfinite(x_orig).all()

subsets = test_subsets(graph, text)
cold_edges = cold_start_edges(graph)
print("subset sizes:", {k: int(v.sum()) for k, v in subsets.items()})
print("cold-start edges kept (undirected):", cold_edges.size(1) // 2, "of", graph.edge_index.size(1) // 2)

rows = []
for seed in SEEDS:
    # K/L: same random vectors for both arms within a seed; unit-norm like the sentence embeddings.
    gen = torch.Generator().manual_seed(1000 + seed)
    rand = torch.randn(graph.num_nodes, 384, generator=gen)
    rand = rand / rand.norm(dim=1, keepdim=True)
    x_concat = torch.cat([x_orig, rand], dim=1)
    for arm, (cls, use_rand) in ARMS.items():
        x = x_concat if use_rand else x_orig
        set_seed(seed)
        model = cls(x.size(1), HIDDEN, NUM_CLASSES)
        t0 = time.time()
        model, hist = train(model, x, graph, max_epochs=max_epochs)
        minutes = (time.time() - t0) / 60
        torch.save(model.state_dict(), WEIGHTS_DIR / f"{arm}_s{seed}.pt")
        best_epoch = int(np.argmax([h["val_acc"] for h in hist]))
        settings = [("normal", graph.edge_index)] + ([("cold", cold_edges)] if cls is not MLP else [])
        for setting, edges in settings:
            split_metrics, pred = evaluate(model, x, graph, edge_index=edges)
            if setting == "normal":
                normal_metrics = split_metrics
            base = {"arm": arm, "model": cls.__name__, "features": "orig+rand" if use_rand else "orig",
                    "seed": seed, "setting": setting, "epochs_run": len(hist), "best_epoch": best_epoch,
                    "minutes": round(minutes, 2), "threads": THREADS}
            for split in ["train", "val"]:
                rows.append({**base, "eval": split, "n": int(graph[f"{split}_mask"].sum()),
                             "acc": split_metrics[f"{split}_acc"], "macro_f1": split_metrics[f"{split}_macro_f1"]})
            for name, mask in subsets.items():
                rows.append({**base, "eval": name, "n": int(mask.sum()), **masked_metrics(graph.y, pred, mask)})
        print(f"seed {seed} arm {arm}: {len(hist)} epochs, best {best_epoch}, {minutes:.1f} min, "
              f"val {normal_metrics['val_acc']:.4f}, test {normal_metrics['test_acc']:.4f}", flush=True)

out_dir = REPO / "results" / "3_gnn_text"
df = pd.DataFrame(rows)
df.to_csv(out_dir / "baselines_runs.csv", index=False)
summary = df.groupby(["arm", "setting", "eval"])[["acc", "macro_f1"]].agg(["mean", "std"]).round(4)
summary.to_csv(out_dir / "baselines_summary.csv")
meta = {"embeddings": emb_path.name, "embeddings_sha256": emb_sha,
        "encoder": emb_meta["encoder"], "encoder_revision": emb_meta["encoder_revision"],
        "seeds": SEEDS, "num_classes": NUM_CLASSES, "hidden": HIDDEN, "threads": THREADS,
        "max_epochs": max_epochs, "lr": 0.01, "weight_decay": 5e-4, "dropout": 0.5, "patience": 50,
        "random_vectors": "K/L: torch.randn(N,384), Generator seed 1000+seed, rows scaled to unit norm; same for K and L",
        "cold_start": "at inference, remove every edge with a test node at either end (GCN and GraphSAGE)",
        "std": "sample std (ddof=1) over seeds",
        "code": "scripts/run_baselines.py"}
(out_dir / "baselines_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
with pd.option_context("display.width", 200, "display.max_rows", 200):
    print(summary)
