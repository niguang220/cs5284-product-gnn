"""Correct & Smooth (Huang et al., 2021) on saved base-model probabilities. Alphas chosen on val accuracy
(mean over seeds); train labels are the only labels used as C&S input. Cold: C&S runs on the cold-start edges.

Usage: python exp/graph/cs.py MODEL[,MODEL...] [--probs DIR] [--knn K --knn-mode all|low]
With --knn, C&S propagates over co-purchase + text-kNN edges (cold: kNN edges are kept, co-purchase test edges dropped).
"""
import argparse
import itertools
import numpy as np
import torch
from torch_geometric.nn.models import CorrectAndSmooth

from common import REPO, augment, cold_start_edges, load, log_rows, metrics

torch.set_num_threads(16)
ap = argparse.ArgumentParser()
ap.add_argument("models")
ap.add_argument("--probs", default="outputs/probs")
ap.add_argument("--knn", type=int, default=0)
ap.add_argument("--knn-mode", default="all")
ap.add_argument("--device", default="cpu")
ap.add_argument("--seed-val", action="store_true",
                help="after choosing alphas on val (train seeds only), rerun C&S on test with train+val labels as seeds")
args = ap.parse_args()
graph, x = load()
N = graph.num_nodes
y = graph.y
ei = augment(graph.edge_index, x, args.knn, args.knn_mode, N, args.device)
cold = augment(cold_start_edges(graph), x, args.knn, args.knn_mode, N, args.device)
vt = graph.val_mask | graph.test_mask  # val-cold proxy: also drop edges touching val nodes
vcold = augment(graph.edge_index[:, ~(vt[graph.edge_index[0]] | vt[graph.edge_index[1]])], x, args.knn, args.knn_mode,
                N, args.device)
dev = torch.device(args.device)
ei, cold, vcold, y, vmask = ei.to(dev), cold.to(dev), vcold.to(dev), y.to(dev), graph.val_mask.to(dev)
sfx = (f"-knn{args.knn}{'low' if args.knn_mode == 'low' else ''}" if args.knn else "") + ("-seedval" if args.seed_val else "")
train_idx = graph.train_mask.nonzero().flatten().to(dev)
GRID = list(itertools.product([0.0, 0.3, 0.5, 0.7, 0.9, 1.0], [0.0, 0.3, 0.5, 0.7, 0.8, 0.9]))
pdir = REPO / args.probs


trval_idx = (graph.train_mask | graph.val_mask).nonzero().flatten().to(dev)


def cs(prob, edges, ac, as_, idx=None):
    idx = train_idx if idx is None else idx
    post = CorrectAndSmooth(50, ac, 50, as_, autoscale=True)
    out = prob
    if ac > 0:
        out = post.correct(out, y[idx], idx, edges)
    if as_ > 0:
        out = post.smooth(out, y[idx], idx, edges)
    return out


for model in args.models.split(","):
    probs = {(s, st): torch.from_numpy(np.load(pdir / f"{model}_{st}_s{s}.npy")).to(dev)
             for s in range(3) for st in ["normal", "cold"] if (pdir / f"{model}_{st}_s{s}.npy").exists()}
    if model.startswith("MLP"):  # an MLP ignores edges, so its cold-start probabilities equal the normal ones
        probs.update({(s, "cold"): probs[(s, "normal")] for s in range(3)})
    has_cold = (0, "cold") in probs
    val = {}
    for ac, as_ in GRID:
        val[(ac, as_)] = np.mean([float((cs(probs[(s, "normal")], ei, ac, as_).argmax(1)[vmask]
                                         == y[vmask]).float().mean()) for s in range(3)])
    best = max(GRID, key=lambda g: (val[g], -g[0] - g[1]))  # ties -> smaller alphas
    print(model, "best", best, round(val[best], 4), "no C&S", round(val[(0.0, 0.0)], 4), flush=True)
    rows = []
    for tag, (ac, as_) in [("base", (0.0, 0.0)), ("C&S", best)]:
        for s in range(3):
            idx = trval_idx if (args.seed_val and tag == "C&S") else None
            pn = cs(probs[(s, "normal")], ei, ac, as_, idx).cpu()
            pc = cs(probs[(s, "cold")], cold, ac, as_, idx).cpu() if has_cold else None
            m = metrics(graph, pn, pc)
            vpath = pdir / f"{model}_vcold_s{s}.npy"
            if vpath.exists():
                pv = cs(torch.from_numpy(np.load(vpath)).to(dev), vcold, ac, as_).argmax(1)
                m["val_cold_acc"] = float((pv[vmask] == y[vmask]).float().mean())
            if idx is not None:  # val acc is meaningless once val labels are seeds; report the train-seed val acc
                m["val_acc"] = float(val[best])
            rows.append({"variant": f"{model}+{tag}{sfx}", "seed": s, "ac": ac, "as": as_, **m})
            if tag == "C&S":
                np.save(REPO / "outputs/graph" / f"{model}+CS{sfx}_normal_s{s}.npy", pn.numpy().astype(np.float32))
                if pc is not None:
                    np.save(REPO / "outputs/graph" / f"{model}+CS{sfx}_cold_s{s}.npy", pc.numpy().astype(np.float32))
    log_rows("cs", rows)
    for tag in ["base", "C&S"]:
        r = [q for q in rows if q["variant"].endswith(tag + sfx)]
        print(model, tag, {k: round(np.mean([q[k] for q in r]), 4) for k in r[0] if k not in ("variant", "seed", "ac", "as")}, flush=True)
