"""Laplacian eigenvector positional encodings for the Graph Transformer."""
import numpy as np
import scipy.sparse as sp
import torch
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import eigsh


def laplacian_pe(edge_index, num_nodes, k=16, dense_limit=2000):
    """Return a float32 tensor [num_nodes, k] of Laplacian eigenvectors.

    Computed per connected component on the symmetric normalised Laplacian: on a disconnected graph each
    component has its own zero eigenvalue, so a global decomposition would spend its k vectors on component
    indicators. Each component's trivial eigenvector is skipped; components with at most k nodes are
    zero-padded, and isolated nodes get all zeros.
    """
    row, col = edge_index.cpu().numpy()
    adj = sp.coo_matrix((np.ones(len(row)), (row, col)), shape=(num_nodes, num_nodes)).tocsr()
    adj.data[:] = 1.0  # duplicate edges count once
    num_comp, comp = connected_components(adj, directed=False)
    pe = np.zeros((num_nodes, k), dtype=np.float32)
    rng = np.random.default_rng(0)
    for c in range(num_comp):
        idx = np.flatnonzero(comp == c)
        if len(idx) < 2:
            continue
        sub = adj[idx][:, idx]
        d = 1.0 / np.sqrt(np.asarray(sub.sum(axis=1)).ravel())
        lap = sp.identity(len(idx)) - sp.diags(d) @ sub @ sp.diags(d)
        m = min(k, len(idx) - 1)
        if len(idx) <= dense_limit:
            vals, vecs = np.linalg.eigh(lap.toarray())
        else:
            # Shift-invert just below 0 finds the smallest eigenvalues of this positive semi-definite matrix.
            vals, vecs = eigsh(lap.tocsc(), k=m + 1, sigma=-1e-3, which="LM", v0=rng.standard_normal(len(idx)))
        order = np.argsort(vals)[1 : m + 1]
        pe[idx, :m] = vecs[:, order]
    return torch.from_numpy(pe)
