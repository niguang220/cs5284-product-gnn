"""Arms G/H: [original ; TAPE explanation] embeddings, over three seeds, with the cold-start test and the test-set slices.

Usage: python scripts/run_tape.py [max_epochs]
Needs outputs/embeddings/original__all-MiniLM-L6-v2.npy and tape__all-MiniLM-L6-v2.npy; each SHA-256 must
match its manifest in results/shared/embeddings/. Same settings as scripts/run_baselines.py, so G/H compare with K/L.
Writes results/2_text_only/tape_runs.csv (one row per run x evaluation), tape_summary.csv and tape_meta.json.
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
from src.models import GCN, MLP  # noqa: E402
from src.train import cold_start_edges, evaluate, masked_metrics, set_seed, test_subsets, train  # noqa: E402

EMB_NAMES = ["original__all-MiniLM-L6-v2", "tape__all-MiniLM-L6-v2"]
THREADS = 4  # results are bit-identical only with the same thread count (see set_seed)
SEEDS = [0, 1, 2]
NUM_CLASSES = 47
HIDDEN = 256
ARMS = {"G": MLP, "H": GCN}

max_epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 500
torch.set_num_threads(THREADS)
WEIGHTS_DIR = REPO / "outputs" / "weights"  # reused by later scripts; not kept in git
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

graph, text = load_products()
embs, emb_metas = [], []
for name in EMB_NAMES:
    path = REPO / "outputs" / "embeddings" / f"{name}.npy"
    meta = json.loads((REPO / "results" / "shared" / "embeddings" / f"{name}.json").read_text())
    if hashlib.sha256(path.read_bytes()).hexdigest() != meta["sha256"]:
        raise ValueError(f"{path.name} does not match its manifest in results/shared/embeddings/")
    emb = torch.from_numpy(np.load(path))
    assert emb.shape == (graph.num_nodes, 384), emb.shape
    assert torch.isfinite(emb).all()
    embs.append(emb)
    emb_metas.append(meta)
x = torch.cat(embs, dim=1)

subsets = test_subsets(graph, text)
cold_edges = cold_start_edges(graph)

rows = []
for seed in SEEDS:
    for arm, cls in ARMS.items():
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
            base = {"arm": arm, "model": cls.__name__, "features": "orig+tape", "seed": seed, "setting": setting,
                    "epochs_run": len(hist), "best_epoch": best_epoch, "minutes": round(minutes, 2),
                    "threads": THREADS}
            for split in ["train", "val"]:
                rows.append({**base, "eval": split, "n": int(graph[f"{split}_mask"].sum()),
                             "acc": split_metrics[f"{split}_acc"], "macro_f1": split_metrics[f"{split}_macro_f1"]})
            for name, mask in subsets.items():
                rows.append({**base, "eval": name, "n": int(mask.sum()), **masked_metrics(graph.y, pred, mask)})
        print(f"seed {seed} arm {arm}: {len(hist)} epochs, best {best_epoch}, {minutes:.1f} min, "
              f"val {normal_metrics['val_acc']:.4f}, test {normal_metrics['test_acc']:.4f}", flush=True)

out_dir = REPO / "results" / "2_text_only"
df = pd.DataFrame(rows)
df.to_csv(out_dir / "tape_runs.csv", index=False)
summary = df.groupby(["arm", "setting", "eval"])[["acc", "macro_f1"]].agg(["mean", "std"]).round(4)
summary.to_csv(out_dir / "tape_summary.csv")
meta = {"embeddings": [m["file"] for m in emb_metas], "embeddings_sha256": [m["sha256"] for m in emb_metas],
        "encoder": emb_metas[0]["encoder"], "encoder_revision": emb_metas[0]["encoder_revision"],
        "seeds": SEEDS, "num_classes": NUM_CLASSES, "hidden": HIDDEN, "threads": THREADS,
        "max_epochs": max_epochs, "lr": 0.01, "weight_decay": 5e-4, "dropout": 0.5, "patience": 50,
        "features": "concatenation [original ; TAPE explanation], 768-d",
        "cold_start": "at inference, remove every edge with a test node at either end (GCN only)",
        "std": "sample std (ddof=1) over seeds",
        "code": "scripts/run_tape.py"}
(out_dir / "tape_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
with pd.option_context("display.width", 200, "display.max_rows", 200):
    print(summary)
