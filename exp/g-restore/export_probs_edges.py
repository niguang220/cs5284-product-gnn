"""scripts/export_probs.py with a choice of graph: the given (sampled) edges or the restored induced subgraph.

Usage: python exp/g-restore/export_probs_edges.py MODEL --emb NAME --graph {given,induced} [--device cuda:0]
Writes outputs/probs/grestore-{graph}-{MODEL}@{emb}_{normal|cold}_s{seed}.npy. Cold start drops every edge touching
a test node at inference, as src/train.py: cold_start_edges does.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from src.data import load_products  # noqa: E402
from src.models import GCN, MLP, SAGE, GraphTransformer  # noqa: E402
from src.train import set_seed, train  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("--emb", default="ft-gte-large_s0")
ap.add_argument("--graph", default="induced")
ap.add_argument("--device", default="cuda:0")
ap.add_argument("--seeds", default="0,1,2")
args = ap.parse_args()

dev = torch.device(args.device)
graph, _ = load_products()
if args.graph == "induced":
    graph.edge_index = torch.from_numpy(np.load(REPO / "outputs/ogb_full/induced_edges.npy")).long()
te = graph.test_mask
cold = graph.edge_index[:, ~(te[graph.edge_index[0]] | te[graph.edge_index[1]])].to(dev)
x = torch.from_numpy(np.load(REPO / "outputs" / "embeddings" / f"{args.emb}.npy")).float().to(dev)
graph = graph.to(dev)
tag = f"grestore-{args.graph}-{args.model}@{args.emb}"
out_dir = REPO / "outputs" / "probs"


def build():
    if args.model == "GT":
        return GraphTransformer(x.size(1), 256, 47, dropout=0.5, num_layers=2, heads=8), {"lr": 1e-3}
    return {"MLP": MLP, "GCN": GCN, "SAGE": SAGE}[args.model](x.size(1), 256, 47), {}


log = {}
for seed in [int(s) for s in args.seeds.split(",")]:
    set_seed(seed)
    model, kw = build()
    model, hist = train(model.to(dev), x, graph, **kw)
    model.eval()
    with torch.no_grad():
        for setting, edges in [("normal", graph.edge_index), ("cold", cold)]:
            p = model(x, edges).softmax(1).cpu().numpy().astype(np.float32)
            np.save(out_dir / f"{tag}_{setting}_s{seed}.npy", p)
            yt, m = graph.y.cpu().numpy(), graph.test_mask.cpu().numpy()
            print(tag, seed, setting, "test acc %.4f" % (p.argmax(1)[m] == yt[m]).mean(), flush=True)
    log[seed] = max(h["val_acc"] for h in hist)
    print(tag, seed, "best val", round(log[seed], 4), flush=True)
(out_dir / f"{tag}.json").write_text(json.dumps({"val_acc": log, "args": vars(args)}, indent=1))
