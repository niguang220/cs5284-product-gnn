"""GPT's own answer, the original text plus GPT's ranked answer, and label propagation (graph only).

Usage: python scripts/run_gpt_pred.py [max_epochs]
Arms:
  GPT    no training: the first category named in TAPE's GPT-3.5 response (missing answer = wrong).
  P_MLP  MLP on [original embedding ; h_pred], h_pred = one-hot of GPT's 1st, 2nd and 3rd category (3 x 47).
  P_GCN  GCN on the same features. TAPE's h_pred also encodes GPT's ranked top-k (core/GNNs, use_pred).
  O      label propagation of the train labels over the co-purchase graph, no text (PyG LabelPropagation);
         alpha and the number of layers are chosen on val accuracy.
Needs outputs/embeddings/original__all-MiniLM-L6-v2.npy and outputs/tape_responses/ogbn-products.zip.
Same seeds, threads and training settings as scripts/run_tape.py. Writes results/2_text_only/gpt_pred_runs.csv,
gpt_pred_summary.csv and gpt_pred_meta.json.
"""
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.nn.models import LabelPropagation

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from src.data import load_products, load_tape_explanations  # noqa: E402
from src.gpt_parse import category_patterns, ranked_categories  # noqa: E402
from src.models import GCN, MLP  # noqa: E402
from src.train import cold_start_edges, evaluate, masked_metrics, set_seed, test_subsets, train  # noqa: E402

EMB_NAME = "original__all-MiniLM-L6-v2"
THREADS = 4
SEEDS = [0, 1, 2]
NUM_CLASSES = 47
HIDDEN = 256
TOP_K = 3
LP_ALPHAS = [0.5, 0.7, 0.9, 0.99]
LP_LAYERS = [3, 10, 50]

max_epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 500
torch.set_num_threads(THREADS)
WEIGHTS_DIR = REPO / "outputs" / "weights"  # reused by later scripts; not kept in git
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

graph, text = load_products()
subsets = test_subsets(graph, text)
cold_edges = cold_start_edges(graph)

path = REPO / "outputs" / "embeddings" / f"{EMB_NAME}.npy"
emb_meta = json.loads((REPO / "results" / "shared" / "embeddings" / f"{EMB_NAME}.json").read_text())
if hashlib.sha256(path.read_bytes()).hexdigest() != emb_meta["sha256"]:
    raise ValueError(f"{path.name} does not match its manifest in results/shared/embeddings/")
x_orig = torch.from_numpy(np.load(path))

patterns = category_patterns()
rank = torch.tensor([ranked_categories(t, patterns, TOP_K) for t in load_tape_explanations(graph.num_nodes)])
missing = {s: float((rank[graph[f"{s}_mask"], 0] < 0).float().mean()) for s in ["train", "val", "test"]}
print("GPT answer missing:", missing, flush=True)
h_pred = torch.cat([F.one_hot(rank[:, k].clamp(min=0), NUM_CLASSES) * (rank[:, k:k + 1] >= 0)
                    for k in range(TOP_K)], dim=1).float()
x_pred = torch.cat([x_orig, h_pred], dim=1)

rows = []


def add_rows(base, pred):
    for split in ["train", "val"]:
        mask = graph[f"{split}_mask"]
        rows.append({**base, "eval": split, "n": int(mask.sum()), **masked_metrics(graph.y, pred, mask)})
    for name, mask in subsets.items():
        rows.append({**base, "eval": name, "n": int(mask.sum()), **masked_metrics(graph.y, pred, mask)})


# GPT: no training. A missing answer gets class -1, which never equals a label.
add_rows({"arm": "GPT", "model": "none", "features": "gpt_top1", "seed": -1, "setting": "normal"}, rank[:, 0])

# O: label propagation; pick (alpha, layers) on val, then apply with full and cold-start edges.
y_masked = graph.y.clone()
lp_val = {}
for alpha in LP_ALPHAS:
    for layers in LP_LAYERS:
        out = LabelPropagation(num_layers=layers, alpha=alpha)(y_masked, graph.edge_index, mask=graph.train_mask)
        lp_val[(alpha, layers)] = float((out[graph.val_mask].argmax(1) == graph.y[graph.val_mask]).float().mean())
alpha, layers = max(lp_val, key=lp_val.get)
print(f"O: best on val alpha={alpha}, layers={layers}, val acc {lp_val[(alpha, layers)]:.4f}", flush=True)
for setting, edges in [("normal", graph.edge_index), ("cold", cold_edges)]:
    out = LabelPropagation(num_layers=layers, alpha=alpha)(y_masked, edges, mask=graph.train_mask)
    add_rows({"arm": "O", "model": "LabelPropagation", "features": "graph+train_labels", "seed": -1,
              "setting": setting, "alpha": alpha, "lp_layers": layers}, out.argmax(1))

# P_MLP / P_GCN
for seed in SEEDS:
    for arm, cls in [("P_MLP", MLP), ("P_GCN", GCN)]:
        set_seed(seed)
        model = cls(x_pred.size(1), HIDDEN, NUM_CLASSES)
        t0 = time.time()
        model, hist = train(model, x_pred, graph, max_epochs=max_epochs)
        minutes = (time.time() - t0) / 60
        torch.save(model.state_dict(), WEIGHTS_DIR / f"{arm}_s{seed}.pt")
        best_epoch = int(np.argmax([h["val_acc"] for h in hist]))
        settings = [("normal", graph.edge_index)] + ([("cold", cold_edges)] if cls is not MLP else [])
        for setting, edges in settings:
            _, pred = evaluate(model, x_pred, graph, edge_index=edges)
            add_rows({"arm": arm, "model": cls.__name__, "features": "orig+gpt_rank", "seed": seed,
                      "setting": setting, "epochs_run": len(hist), "best_epoch": best_epoch,
                      "minutes": round(minutes, 2), "threads": THREADS}, pred)
        val = [r["acc"] for r in rows if r["arm"] == arm and r["seed"] == seed and r["eval"] == "val"][0]
        print(f"seed {seed} arm {arm}: {len(hist)} epochs, best {best_epoch}, {minutes:.1f} min, val {val:.4f}",
              flush=True)

out_dir = REPO / "results" / "2_text_only"
df = pd.DataFrame(rows)
df.to_csv(out_dir / "gpt_pred_runs.csv", index=False)
summary = df.groupby(["arm", "setting", "eval"])[["acc", "macro_f1"]].agg(["mean", "std"]).round(4)
summary.to_csv(out_dir / "gpt_pred_summary.csv")
meta = {"embedding": emb_meta["file"], "embedding_sha256": emb_meta["sha256"], "gpt_answer_missing": missing,
        "top_k": TOP_K, "seeds": SEEDS, "num_classes": NUM_CLASSES, "hidden": HIDDEN, "threads": THREADS,
        "max_epochs": max_epochs, "lr": 0.01, "weight_decay": 5e-4, "dropout": 0.5, "patience": 50,
        "label_propagation_val_acc": {f"alpha={a},layers={k}": v for (a, k), v in lp_val.items()},
        "label_propagation_chosen": {"alpha": alpha, "layers": layers},
        "cold_start": "at inference, remove every edge with a test node at either end (GCN and O)",
        "std": "sample std (ddof=1) over seeds; GPT and O are deterministic (seed -1)",
        "code": "scripts/run_gpt_pred.py"}
(out_dir / "gpt_pred_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
with pd.option_context("display.width", 200, "display.max_rows", 200):
    print(summary.loc[(slice(None), slice(None), ["train", "val"]), :])
