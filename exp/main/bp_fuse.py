"""Graph-aware LLM fusion: one round of belief propagation of the neighbours' LLM answers (and train labels).

b_i(y) ∝ p(y | x_i) · P(g_i | y) · Π_{j∈N(i)} m_j(y)^β,   m_j(y) = Σ_{y'} H[y, y'] · e_j(y'),
e_j = onehot(y_j) for a train neighbour, P(g_j | ·) (normalised) for an unlabelled one.
H[y, y'] = P(neighbour has y' | node has y), counted on train-train edges, smoothed toward 0.8·I + 0.2·uniform.
β (one scalar) is chosen on val accuracy.
"""
import csv, sys
from pathlib import Path
import numpy as np, scipy.sparse as sp, torch
REPO = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(REPO))
from src.data import load_products
from src.label_shift import K, confusion, fit_temperature, fuse, temper
from src.train import masked_metrics, cold_start_edges
graph, _ = load_products(); y = graph.y.numpy(); N = len(y)
tr, va, te = graph.train_mask.numpy(), graph.val_mask.numpy(), graph.test_mask.numpy()
g1 = np.array([int(r[0]) for r in csv.reader(open(REPO / "outputs/tape_gpt_preds/ogbn-products.csv"))])
C = confusion(g1, y, tr | va)
ei = graph.edge_index.numpy()
both = tr[ei[0]] & tr[ei[1]]
H = np.zeros((K, K)); np.add.at(H, (y[ei[0][both]], y[ei[1][both]]), 1)
prior_H = 0.8 * np.eye(K) + 0.2 / K
H = (H + 5 * prior_H) / (H.sum(1, keepdims=True) + 5)  # 5 pseudo-edges per row toward the homophily prior
print("H diagonal mean (train classes):", H.diagonal()[np.bincount(y[tr], minlength=K) > 20].mean().round(3))
# evidence of every node about its own label, as seen by a neighbour (val labels are NOT used as evidence)
ev = C[g1 % (K + 1)]; ev = ev / ev.sum(1, keepdims=True)
ev[tr] = np.eye(K)[y[tr]]
msg = ev @ H.T  # msg[j, y] = Σ_y' H[y, y'] e_j(y')
logmsg = np.log(np.clip(msg, 1e-9, None))
yt = torch.from_numpy(y)
def met(p, mask): return masked_metrics(yt, torch.from_numpy(p.argmax(1)), torch.from_numpy(mask))
for setting in ["normal", "cold"]:
    e = graph.edge_index if setting == "normal" else cold_start_edges(graph)
    e = e.numpy()
    A = sp.coo_matrix((np.ones(e.shape[1]), (e[1], e[0])), shape=(N, N)).tocsr(); A.data[:] = 1
    deg = np.asarray(A.sum(1)).ravel()
    S = A @ logmsg  # Σ_j log m_j
    Smean = S / np.maximum(deg, 1)[:, None]
    for model in ["MLP", "GCN", "SAGE", "GT"]:
        if model == "MLP" and setting == "cold": continue
        res = {}
        for s in range(3):
            f = REPO / f"outputs/probs/{model}_{setting}_s{s}.npy"
            if not f.exists(): f = REPO / f"outputs/probs/{model}_normal_s{s}.npy"
            p = np.load(f).astype(np.float64); p = temper(p, fit_temperature(p, y, va))
            base = np.log(fuse(p, [g1], [C]))
            for agg, SS in [("sum", S), ("mean", Smean)]:
                for beta in [0, 0.25, 0.5, 1.0]:
                    z = base + beta * SS; q = np.exp(z - z.max(1, keepdims=True))
                    v, t = met(q, va), met(q, te)
                    res.setdefault((agg, beta), []).append((v["acc"], t["acc"], t["macro_f1"]))
        best = max(res, key=lambda k: np.mean([r[0] for r in res[k]]))
        print(setting, model)
        for k, v in res.items():
            v = np.array(v); flag = " <- val-selected" if k == best else ""
            print(f"   {k[0]:4s} beta={k[1]:<4} val {v[:,0].mean():.4f} test {v[:,1].mean():.4f} f1 {v[:,2].mean():.4f}{flag}")
