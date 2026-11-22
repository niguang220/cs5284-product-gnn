"""Compute and cache every encoding (normal and cold-start graph) and print simple stats."""
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.data import load_products  # noqa: E402
from src.train import cold_start_edges  # noqa: E402
import encodings_ga as enc  # noqa: E402

dev = sys.argv[1] if len(sys.argv) > 1 else "cuda:0"
graph, _ = load_products()
n = graph.num_nodes
for setting, ei in [("normal", graph.edge_index), ("cold", cold_start_edges(graph))]:
    for name, dims in [("deg", [16]), ("dist", [16]), ("rw", [8, 16]), ("lap", [8, 16])]:
        for d in dims:
            t0 = time.time()
            e = enc.get(name, ei, n, setting, graph.train_mask, d, dev)
            print(setting, name, d, tuple(e.shape), f"{time.time() - t0:.1f}s", "colmean", np.round(e.mean(0).numpy()[:8], 3), flush=True)
