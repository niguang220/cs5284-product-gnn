"""Build the label-free holdouts of train nodes (src/shift.py) and save their node ids to results/shared/holdouts.json.

Usage: python scripts/make_holdouts.py
Uses split membership, MiniLM embeddings and node degree only. No labels are read.
"""
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from src.arms import checked_load  # noqa: E402
from src.data import load_products  # noqa: E402
from src.shift import domain_scores, make_holdouts  # noqa: E402

graph, _ = load_products()
name = "original__all-MiniLM-L6-v2"
x = checked_load(REPO / "outputs" / "embeddings" / f"{name}.npy", REPO / "results" / "shared" / "embeddings" / f"{name}.json")
x = x.astype(np.float32)
degree = np.bincount(graph.edge_index[0].numpy(), minlength=graph.num_nodes)
train_mask, test_mask = graph.train_mask.numpy(), graph.test_mask.numpy()

p_test, auc = domain_scores(x, degree, train_mask, test_mask)
holdouts = make_holdouts(p_test, train_mask)
print(f"train vs test cross-validated AUC {auc:.4f}")
print(f"mean P(test): train {np.nanmean(p_test[train_mask]):.3f}, test {np.nanmean(p_test[test_mask]):.3f}")
for key, nodes in holdouts.items():
    print(f"{key}: {len(nodes)} nodes, mean P(test) {p_test[nodes].mean():.3f}")
out = {"auc_train_vs_test": auc, "n": 2000, "code": "scripts/make_holdouts.py",
       **{key: [int(i) for i in nodes] for key, nodes in holdouts.items()}}
(REPO / "results" / "shared" / "holdouts.json").write_text(json.dumps(out) + "\n")
