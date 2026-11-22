"""Collective inference: iterate belief updates so that test nodes inform each other over several rounds.

b_i^{t+1}(y) ∝ u_i(y) · Π_j m_{j}(y)^β,  m_j(y) = Σ_y' H[y, y'] b_j^t(y'),  u_i = fused unary (model × LLM likelihood);
train nodes are clamped to their labels; damping 0.5. One round from b^0 = LLM-only evidence is the BP used so far.
Rounds T, β and γ (kNN edges) are chosen on val.
"""
import csv, sys
from pathlib import Path
import numpy as np, scipy.sparse as sp, torch
REPO = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(REPO))
from src.data import load_products
from src.improve import nearest_in_pool
from src.label_shift import K, compatibility, confusion, fit_temperature, fuse, temper
from src.train import cold_start_edges
g, _ = load_products(); y, N = g.y.numpy(), g.num_nodes
tr, va, te = g.train_mask.numpy(), g.val_mask.numpy(), g.test_mask.numpy()
gpt = np.array([int(r[0]) for r in csv.reader(open(REPO / "outputs/tape_gpt_preds/ogbn-products.csv"))])
C = confusion(gpt, y, tr | va)
ev = C[gpt % (K + 1)]; ev = ev / ev.sum(1, keepdims=True); ev[tr] = np.eye(K)[y[tr]]
x = torch.nn.functional.normalize(torch.from_numpy(np.load(REPO / "outputs/embeddings/original__all-MiniLM-L6-v2.npy")), dim=1)
kn = nearest_in_pool(torch.arange(N), x, torch.arange(N), 5).numpy()
def adj(e):
    A = sp.coo_matrix((np.ones(e.shape[1]), (e[1], e[0])), shape=(N, N)).tocsr(); A.data[:] = 1; return A
Ak, Hk = adj(kn), compatibility(kn, y, tr)
H = compatibility(g.edge_index.numpy(), y, tr)
tag = sys.argv[1] if len(sys.argv) > 1 else "GT@ft-gte-large_s0"
for setting, e in [("normal", g.edge_index.numpy()), ("cold", cold_start_edges(g).numpy())]:
    A = adj(e)
    res = {}
    for s in range(3):
        p = np.load(REPO / f"outputs/probs/{tag}_{setting}_s{s}.npy").astype(float)
        u = np.log(fuse(temper(p, fit_temperature(p, y, va)), [gpt], [C]))
        for beta in [0.25, 0.5]:
            for gamma in [0.0, 0.25]:
                b = ev.copy()
                for T in range(1, 6):
                    z = u + beta * (A @ np.log(np.clip(b @ H.T, 1e-9, None))) + gamma * (Ak @ np.log(np.clip(b @ Hk.T, 1e-9, None)))
                    q = np.exp(z - z.max(1, keepdims=True)); q /= q.sum(1, keepdims=True)
                    pr = q.argmax(1)
                    res.setdefault((beta, gamma, T), []).append(((pr[va] == y[va]).mean(), (pr[te] == y[te]).mean()))
                    b = 0.5 * b + 0.5 * q; b[tr] = np.eye(K)[y[tr]]
    best = max(res, key=lambda k: np.mean([r[0] for r in res[k]]))
    print(tag, setting)
    for k, v in sorted(res.items()):
        v = np.array(v); print(f"  beta={k[0]} gamma={k[1]} T={k[2]}  val {v[:,0].mean():.4f} test {v[:,1].mean():.4f}{'  <- val-selected' if k == best else ''}")
