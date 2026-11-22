"""Shared pieces for the graph-local prior experiments: a copy of scripts/run_fusion.py's fusion + BP, made
reusable so a per-node log prior ratio (or a per-node temperature) can be added to the fused log posterior.

Everything here is computed once and cached under outputs/g-localprior/.
"""
import csv
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from src.data import load_products  # noqa: E402
from src.improve import nearest_in_pool  # noqa: E402
from src.label_shift import K, compatibility, confusion, fit_temperature, neighbour_log_messages, temper  # noqa: E402,F401
from src.train import cold_start_edges, masked_metrics  # noqa: E402

OUT = REPO / "outputs/g-localprior"
OUT.mkdir(parents=True, exist_ok=True)
BETAS = [0.0, 0.25, 0.5, 1.0]
GAMMAS = [0.0, 0.25, 0.5, 1.0]
KNN = 5
RARE = {12: "Electronics", 18: "Office", 21: "Tools", 16: "Automotive"}

graph, _ = load_products()
y, N = graph.y.numpy(), graph.num_nodes
tr, va, te = graph.train_mask.numpy(), graph.val_mask.numpy(), graph.test_mask.numpy()
yt = torch.from_numpy(y)

llm = {"gpt": np.array([int(r[0]) for r in csv.reader(open(REPO / "outputs/tape_gpt_preds/ogbn-products.csv"))])}
for name, path in [("qwen", REPO / "outputs/llm/qwen35_9b.npy")] + [
        (p.stem, p) for p in sorted((REPO / "outputs/llm_answers").glob("*.npy"))]:
    if path.exists():
        llm[name] = np.load(path)
C = {k: confusion(a, y, tr | va) for k, a in llm.items()}
COMBOS = [("gpt",)] + [("gpt", k) for k in llm if k != "gpt"]
H = compatibility(graph.edge_index.numpy(), y, tr)
EDGES = {"normal": graph.edge_index.numpy(), "cold": cold_start_edges(graph).numpy()}


def llm_loglik(combo):
    """Σ_k log P(g_k | y): the LLM answers' log likelihood, [N, K]."""
    out = np.zeros((N, K))
    for k in combo:
        out += np.log(C[k][llm[k] % (K + 1)])
    return out


def evidence(combo):
    e = np.ones((N, K))
    for k in combo:
        e *= C[k][llm[k] % (K + 1)]
    e /= e.sum(1, keepdims=True)
    e[tr] = np.eye(K)[y[tr]]
    return e


MSGS = {(s, c): neighbour_log_messages(EDGES[s], N, evidence(c), H) for s in EDGES for c in COMBOS}
_x = torch.nn.functional.normalize(torch.from_numpy(
    np.load(REPO / "outputs/embeddings/original__all-MiniLM-L6-v2.npy")).float(), dim=1)
KNN_EDGES = nearest_in_pool(torch.arange(N), _x, torch.arange(N), KNN).numpy()  # j -> i
KNN_MSGS = {c: neighbour_log_messages(KNN_EDGES, N, evidence(c), compatibility(KNN_EDGES, y, tr)) for c in COMBOS}
LLMLOG = {c: llm_loglik(c) for c in COMBOS}


def adjacency(edge_index, sym=True):
    A = sp.coo_matrix((np.ones(edge_index.shape[1]), (edge_index[1], edge_index[0])), shape=(N, N)).tocsr()
    if sym:
        A = A + A.T
    A.data[:] = 1
    A.setdiag(0)
    A.eliminate_zeros()
    return A


def hop_distance(A, sources):
    """BFS hop distance from the nearest source node (inf if unreachable)."""
    d = np.full(N, np.inf)
    d[sources] = 0
    frontier = sources.copy()
    h = 0
    while frontier.any():
        h += 1
        nxt = (A @ frontier.astype(np.float32)) > 0
        nxt &= np.isinf(d)
        d[nxt] = h
        frontier = nxt
    return d


# Report buckets: hop distance to train on the full co-purchase graph (a fixed node attribute, analysis only).
A_CO = adjacency(EDGES["normal"])
HOP_REPORT = hop_distance(A_CO, tr)
BUCKETS = {"1": HOP_REPORT == 1, "2": HOP_REPORT == 2, "3": HOP_REPORT == 3, "4+": HOP_REPORT >= 4}
# Graphs the methods may use: cold start keeps no co-purchase edge touching a test node; kNN edges stay.
A_KNN = adjacency(KNN_EDGES)
GRAPH = {"normal": ((A_CO + A_KNN) > 0).astype(np.float64).tocsr(),
         "cold": ((adjacency(EDGES["cold"]) + A_KNN) > 0).astype(np.float64).tocsr()}
GRAPH_CO = {"normal": A_CO, "cold": adjacency(EDGES["cold"])}


def load_base(tag, setting, seed):
    p = np.load(REPO / f"outputs/probs/{tag}_{setting}_s{seed}.npy").astype(np.float64)
    return p


def calibrated_logp(p):
    """Log of the temperature-scaled probabilities (one temperature fitted on val NLL), as run_fusion.py."""
    return np.log(np.clip(temper(p, fit_temperature(p, y, va)), 1e-12, 1))


def normalise_log(z):
    z = z - z.max(1, keepdims=True)
    return z - np.log(np.exp(z).sum(1, keepdims=True))


def score(z):
    pred = torch.from_numpy(z.argmax(1))
    v, t = masked_metrics(yt, pred, torch.from_numpy(va)), masked_metrics(yt, pred, torch.from_numpy(te))
    return {"val_acc": v["acc"], "test_acc": t["acc"], "test_f1": t["macro_f1"]}


def detail(z):
    """Hop-bucket accuracy and recall on the train-rare classes (test, analysis only)."""
    pred = z.argmax(1)
    out = {f"hop{b}": float((pred[m & te] == y[m & te]).mean()) for b, m in BUCKETS.items()}
    for c, name in RARE.items():
        m = te & (y == c)
        out[f"rec_{name}"] = float((pred[m] == c).mean())
    return out


def grid(logp_fused_fn, setting, extra=None):
    """Yield (config, z) over the fusion+BP(+kNN) grid. logp_fused_fn(combo) -> fused log posterior [N, K]."""
    for combo in COMBOS:
        lf = logp_fused_fn(combo)
        for beta in BETAS:
            zb = lf + beta * MSGS[(setting, combo)]
            for gamma in GAMMAS:
                yield {"llms": "+".join(combo), "beta": beta, "gamma": gamma, **(extra or {})}, zb + gamma * KNN_MSGS[combo]


def select(df, keys):
    """Val selection as run_fusion.py: best mean val acc (over GNN seeds) among configs that use BP (β>0 or γ>0)."""
    d = df[(df.beta > 0) | (df.gamma > 0)]
    g = d.groupby(keys, dropna=False)[["val_acc", "test_acc", "test_f1"]].mean().reset_index()
    return g.loc[g.val_acc.idxmax()]
