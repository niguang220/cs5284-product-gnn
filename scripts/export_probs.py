"""Train a base model over three seeds and save its softmax probabilities on every node, for post-hoc methods.

Usage: python scripts/export_probs.py MODEL [--emb NAME] [--la] [--device cuda:0] [--save-ckpt | --load-ckpt]
  MODEL: MLP, GCN, SAGE, GT (Graph Transformer without PE, the best GT arm) or NAGphormer (K = 6, the final base).
  --emb: embedding file in outputs/embeddings (default original__all-MiniLM-L6-v2).
  --la:  logit-adjusted loss (src/train.py), so the model estimates a class-balanced posterior.
  --save-ckpt: also save each seed's weights to outputs/ckpt/gnn/{tag}_s{seed}.pt.
  --load-ckpt: skip training and load those weights (inference only).
Writes outputs/probs/{tag}_{normal|cold}_s{seed}.npy (float32, [N, 47]) and a json with val accuracy per seed.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from src.data import load_products  # noqa: E402
from src.models import GCN, MLP, NAGphormer, SAGE, GraphTransformer  # noqa: E402
from src.train import cold_start_edges, set_seed, train  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("--emb", default="original__all-MiniLM-L6-v2")
ap.add_argument("--la", action="store_true")
ap.add_argument("--device", default="cuda:0")
ap.add_argument("--seeds", default="0,1,2")
ap.add_argument("--save-ckpt", action="store_true")
ap.add_argument("--load-ckpt", action="store_true")
args = ap.parse_args()

dev = torch.device(args.device)
graph, _ = load_products()
x = torch.from_numpy(np.load(REPO / "outputs" / "embeddings" / f"{args.emb}.npy")).float().to(dev)
cold = cold_start_edges(graph).to(dev)
graph = graph.to(dev)
tag = args.model + ("" if args.emb == "original__all-MiniLM-L6-v2" else f"@{args.emb}") + ("-LA" if args.la else "")
out_dir = REPO / "outputs" / "probs"
out_dir.mkdir(parents=True, exist_ok=True)

offset = None
if args.la:
    counts = torch.bincount(graph.y[graph.train_mask], minlength=47).float()
    # Classes with no training node get a pseudo-count of 1 so the offset stays finite.
    offset = torch.log(counts.clamp(min=1) / counts.sum())


def build():
    if args.model == "GT":  # configuration selected in scripts/run_graph_transformer.py
        return GraphTransformer(x.size(1), 256, 47, dropout=0.5, num_layers=2, heads=8), {"lr": 1e-3}
    if args.model == "NAGphormer":  # configuration selected on val in exp/g-arch
        return NAGphormer(x.size(1), 256, 47, dropout=0.5, K=6), {"lr": 1e-3}
    return {"MLP": MLP, "GCN": GCN, "SAGE": SAGE}[args.model](x.size(1), 256, 47), {}


log = {}
for seed in [int(s) for s in args.seeds.split(",")]:
    set_seed(seed)
    model, kw = build()
    ckpt = REPO / "outputs" / "ckpt" / "gnn" / f"{tag}_s{seed}.pt"
    if args.load_ckpt:
        model.load_state_dict(torch.load(ckpt, map_location=dev))
        model.to(dev)
        hist = [{"val_acc": float("nan")}]
    else:
        model, hist = train(model.to(dev), x, graph, loss_offset=offset, **kw)
        if args.save_ckpt:
            ckpt.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), ckpt)
    model.eval()
    with torch.no_grad():
        for setting, edges in [("normal", graph.edge_index), ("cold", cold)]:
            if args.model == "MLP" and setting == "cold":
                continue
            p = model(x, edges).softmax(1).cpu().numpy().astype(np.float32)
            np.save(out_dir / f"{tag}_{setting}_s{seed}.npy", p)
    log[seed] = max(h["val_acc"] for h in hist)
    print(tag, seed, "loaded" if args.load_ckpt else f"best val {log[seed]:.4f}", flush=True)
if not args.load_ckpt:
    (out_dir / f"{tag}.json").write_text(json.dumps({"val_acc": log, "args": vars(args)}, indent=1))
