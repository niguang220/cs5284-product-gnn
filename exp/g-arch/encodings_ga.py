"""Positional / structural encodings for the g-arch sprint, each computed on whichever graph the model sees.

Cold start: every encoding is recomputed on the cold-start graph (no edge touching a test node), so no test edge
leaks into inference. Cached in outputs/g-arch/enc_{name}_{setting}.npy.
"""
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
from scipy.sparse.csgraph import dijkstra

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from src.positional import laplacian_pe  # noqa: E402

CACHE = REPO / "outputs" / "g-arch"


def _adj(edge_index, n):
    row, col = edge_index.cpu().numpy()
    a = sp.coo_matrix((np.ones(len(row)), (row, col)), shape=(n, n)).tocsr()
    a.data[:] = 1.0
    a.setdiag(0)
    a.eliminate_zeros()
    return a


def rwpe(edge_index, n, k=16, device="cuda", block=4096):
    """RWPE (Dwivedi et al. 2022): [P_ii, (P^2)_ii, ..., (P^k)_ii] with P = D^-1 A, the probability that a random
    walk returns to its start after t steps. Exact, by pushing blocks of unit vectors through P on the GPU.
    Isolated nodes get zeros. Unlike Laplacian eigenvectors it has no sign ambiguity and is defined per node, so the
    many small components need no special handling."""
    a = _adj(edge_index, n)
    deg = np.asarray(a.sum(1)).ravel()
    inv = np.where(deg > 0, 1.0 / np.maximum(deg, 1), 0.0)
    p = (sp.diags(inv) @ a).tocoo()
    # Right-multiplying a column block by P^T gives rows of P^t: x_{t} = P x_{t-1} with x_0 = e_i.
    P = torch.sparse_coo_tensor(np.vstack([p.row, p.col]), p.data, (n, n), dtype=torch.float32).to(device).coalesce()
    out = np.zeros((n, k), dtype=np.float32)
    for s in range(0, n, block):
        idx = torch.arange(s, min(s + block, n), device=device)
        x = torch.zeros(n, len(idx), device=device)
        x[idx, torch.arange(len(idx), device=device)] = 1.0
        for t in range(k):
            x = torch.sparse.mm(P, x)
            out[s:s + len(idx), t] = x[idx, torch.arange(len(idx), device=device)].cpu().numpy()
    return out


def degree_onehot(edge_index, n):
    """One-hot degree bucket: 0, 1, 2, 3, 4-5, 6-10, 11-20, 21+ (median degree is 1)."""
    deg = np.asarray(_adj(edge_index, n).sum(1)).ravel()
    edges = [0, 1, 2, 3, 4, 6, 11, 21]
    b = np.searchsorted(edges, deg, side="right") - 1
    return np.eye(len(edges), dtype=np.float32)[b]


def train_distance(edge_index, n, train_mask, cap=6, chunk=1000):
    """Hop distance to the nearest *other* train node (leave-self-out), one-hot over 1..cap-1, cap+ / unreachable.

    Uses only which nodes are in train, never a label. Leave-self-out: with plain BFS every train node would sit at
    distance 0, so the training loss would never see the other buckets; excluding the node itself puts train nodes
    on the same scale as val/test nodes."""
    a = _adj(edge_index, n)
    src = np.flatnonzero(train_mask)
    best = np.full(n, np.inf)
    for s in range(0, len(src), chunk):
        d = dijkstra(a, directed=False, indices=src[s:s + chunk], unweighted=True, limit=cap)
        d[np.arange(d.shape[0]), src[s:s + chunk]] = np.inf  # leave self out
        best = np.minimum(best, d.min(0))
    b = np.where(np.isinf(best), cap, np.minimum(best, cap)).astype(int) - 1  # 1..cap -> 0..cap-1
    return np.eye(cap, dtype=np.float32)[b], best


def get(name, edge_index, n, setting, train_mask=None, dim=16, device="cuda"):
    path = CACHE / f"enc_{name}{dim}_{setting}.npy"
    if path.exists():
        return torch.from_numpy(np.load(path))
    if name == "lap":
        e = laplacian_pe(edge_index, n, dim).numpy()
    elif name == "rw":
        e = rwpe(edge_index, n, dim, device)
    elif name == "deg":
        e = degree_onehot(edge_index, n)
    elif name == "dist":
        e = train_distance(edge_index, n, train_mask.cpu().numpy())[0]
    else:
        raise ValueError(name)
    np.save(path, e.astype(np.float32))
    return torch.from_numpy(e.astype(np.float32))
