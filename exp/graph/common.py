"""Shared helpers for the graph exploration: data, cold-start edges, hop buckets, metrics, result logging."""
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from src.data import load_products  # noqa: E402
from src.train import cold_start_edges, masked_metrics, set_seed  # noqa: E402,F401

OUT = REPO / "outputs" / "graph"
EXP = REPO / "exp" / "graph"
C = 47


def load():
    graph, _ = load_products()
    x = torch.from_numpy(np.load(REPO / "outputs/embeddings/original__all-MiniLM-L6-v2.npy")).float()
    return graph, x


def hops_to_train(graph):
    """Shortest hop distance from every node to its nearest train node on the full graph (-1 = unreachable)."""
    n = graph.num_nodes
    row, col = graph.edge_index
    dist = torch.full((n,), -1, dtype=torch.long)
    frontier = graph.train_mask.clone()
    dist[frontier] = 0
    d = 0
    while frontier.any():
        d += 1
        nxt = torch.zeros(n, dtype=torch.bool)
        nxt[col[frontier[row]]] = True
        nxt &= dist < 0
        dist[nxt] = d
        frontier = nxt
    return dist


def hop_buckets(graph, dist):
    t = graph.test_mask
    return {"1": t & (dist == 1), "2": t & (dist == 2), "3": t & (dist == 3), "4+": t & ((dist >= 4) | (dist < 0))}


def metrics(graph, prob_normal, prob_cold=None):
    """val acc, test acc, test macro-F1 (normal graph), cold-start test acc / F1."""
    pred = prob_normal.argmax(1)
    v = masked_metrics(graph.y, pred, graph.val_mask)
    t = masked_metrics(graph.y, pred, graph.test_mask)
    out = {"val_acc": v["acc"], "test_acc": t["acc"], "test_f1": t["macro_f1"]}
    if prob_cold is not None:
        tc = masked_metrics(graph.y, prob_cold.argmax(1), graph.test_mask)
        out.update(cold_acc=tc["acc"], cold_f1=tc["macro_f1"])
    return out


def log_rows(name, rows):
    """Append result rows (dicts) to exp/graph/results.jsonl."""
    with open(EXP / "results.jsonl", "a") as f:
        for r in rows:
            f.write(json.dumps({"exp": name, **r}) + "\n")


def knn_edges(x, k, nodes, device="cpu", batch=2048):
    """Directed edges node -> its k most similar nodes (cosine on x), for each node in `nodes`; no self edges."""
    import torch.nn.functional as F
    xn = F.normalize(x, dim=1).to(device)
    out = []
    for s in range(0, len(nodes), batch):
        q = nodes[s:s + batch].to(device)
        sim = xn[q] @ xn.T
        sim[torch.arange(len(q), device=device), q] = -2
        top = sim.topk(k, dim=1).indices
        out.append(torch.stack([q.repeat_interleave(k), top.flatten()]).cpu())
    return torch.cat(out, 1)


def augment(edge_index, x, k, mode, num_nodes, device="cpu"):
    """Co-purchase edges plus symmetric text-kNN edges, for all nodes or only nodes of degree <= 1."""
    from torch_geometric.utils import degree, to_undirected
    if k == 0:
        return edge_index
    deg = degree(edge_index[0], num_nodes)
    nodes = torch.arange(num_nodes) if mode == "all" else (deg <= 1).nonzero().flatten()
    return to_undirected(torch.cat([edge_index, knn_edges(x, k, nodes, device)], 1), num_nodes=num_nodes)
