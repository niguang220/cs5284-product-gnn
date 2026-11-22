"""Louvain partitions of co-purchase ∪ text-kNN (normal) and cold-start co-purchase ∪ text-kNN (cold)."""
import sys
from pathlib import Path
import networkx as nx
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from common import GRAPH, OUT, N, y, tr, te, K  # noqa: E402

setting, res = sys.argv[1], float(sys.argv[2])
comms = nx.community.louvain_communities(nx.from_scipy_sparse_array(GRAPH[setting]), resolution=res, seed=0)
lab = np.zeros(N, dtype=np.int64)
for i, c in enumerate(comms):
    lab[list(c)] = i
np.save(OUT / f"louvain_{setting}_r{res:g}.npy", lab)
sizes = np.bincount(lab)
# label TV of each community's test nodes to the train prior, weighted by test nodes (analysis only)
pi_tr = np.bincount(y[tr], minlength=K) / tr.sum()
tv = []
for g in range(lab.max() + 1):
    m = (lab == g) & te
    if m.sum():
        tv.append((m.sum(), 0.5 * np.abs(np.bincount(y[m], minlength=K) / m.sum() - pi_tr).sum()))
tv = np.array(tv)
print(setting, res, "communities", len(sizes), "median size", np.median(sizes), "max", sizes.max(),
      "test-weighted TV to train prior", round((tv[:, 0] * tv[:, 1]).sum() / tv[:, 0].sum(), 3))
