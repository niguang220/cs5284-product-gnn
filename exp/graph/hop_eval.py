"""Test accuracy by hop distance to the nearest train node (full co-purchase graph), from saved probabilities.

Usage: python exp/graph/hop_eval.py DIR/VARIANT [DIR/VARIANT ...]   (reads VARIANT_{normal,cold}_s{0,1,2}.npy)
"""
import sys

import numpy as np
import torch

from common import REPO, hop_buckets, hops_to_train, load, metrics

graph, _ = load()
b = hop_buckets(graph, hops_to_train(graph))
# Heterophilous test nodes (analysis only): degree >= 1 but no neighbour shares the node's label.
row, col = graph.edge_index
same = torch.zeros(graph.num_nodes, dtype=torch.long).index_add_(0, col, (graph.y[row] == graph.y[col]).long())
deg = torch.bincount(col, minlength=graph.num_nodes)
b["het"] = graph.test_mask & (deg > 0) & (same == 0)
print("share of test:", {k: round(float(v.sum()) / int(graph.test_mask.sum()), 3) for k, v in b.items()})
for spec in sys.argv[1:]:
    res = []
    for s in range(3):
        pn = torch.from_numpy(np.load(REPO / f"{spec}_normal_s{s}.npy"))
        cpath = REPO / f"{spec}_cold_s{s}.npy"
        pc = torch.from_numpy(np.load(cpath)) if cpath.exists() else None
        r = metrics(graph, pn, pc)
        for st, p in [("", pn), ("cold_", pc)]:
            if p is not None:
                r.update({f"{st}hop{k}": float((p.argmax(1)[m] == graph.y[m]).float().mean()) for k, m in b.items()})
        res.append(r)
    print(spec, {k: f"{np.mean([r[k] for r in res]):.4f}" for k in res[0]})
