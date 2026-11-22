"""Degree-normalised BP over the restored graph: β · mean_j log m_j instead of β · Σ_j log m_j (so a node with 25
neighbours is not 25 times more confident). GPT only; β, temperature on val. Test labels only for reporting."""
import csv
import sys
from pathlib import Path
import numpy as np
import scipy.sparse as sp
import torch
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from src.data import load_products
from src.label_shift import K, compatibility, confusion, fit_temperature, fuse, temper
graph, _ = load_products()
y = graph.y.numpy(); N = len(y)
tr, va, te = graph.train_mask.numpy(), graph.val_mask.numpy(), graph.test_mask.numpy()
gpt = np.array([int(r[0]) for r in csv.reader(open(REPO / "outputs/tape_gpt_preds/ogbn-products.csv"))])
C = confusion(gpt, y, tr | va)
e = C[gpt % (K + 1)]; e = e / e.sum(1, keepdims=True); e[tr] = np.eye(K)[y[tr]]
for gname in ["given", "induced"]:
    ei = graph.edge_index.numpy() if gname == "given" else np.load(REPO / "outputs/ogb_full/induced_edges.npy")
    H = compatibility(ei, y, tr)
    L = np.log(np.clip(e @ H.T, 1e-9, None))
    for setting in ["normal", "cold"]:
        eis = ei if setting == "normal" else ei[:, ~(te[ei[0]] | te[ei[1]])]
        A = sp.csr_matrix((np.ones(eis.shape[1]), (eis[1], eis[0])), shape=(N, N)); A.data[:] = 1
        deg = np.maximum(np.asarray(A.sum(1)).ravel(), 1)
        msum = A @ L
        for tag in ["GT@ft-gte-large_s0", f"grestore-{gname}-GT@ft-gte-large_s0"]:
            res = {}
            for seed in range(3):
                p = np.load(REPO / f"outputs/probs/{tag}_{setting}_s{seed}.npy").astype(np.float64)
                lf = np.log(fuse(temper(p, fit_temperature(p, y, va)), [gpt], [C]))
                for norm in ["sum", "mean", "sqrt"]:
                    m = msum if norm == "sum" else msum / (deg[:, None] if norm == "mean" else np.sqrt(deg)[:, None])
                    for beta in [0, 0.25, 0.5, 1, 2, 4]:
                        pr = (lf + beta * m).argmax(1)
                        res.setdefault((norm, beta), []).append(((pr[va] == y[va]).mean(), (pr[te] == y[te]).mean()))
            r = {k: np.mean(v, 0) for k, v in res.items()}
            for norm in ["sum", "mean", "sqrt"]:
                best = max([k for k in r if k[0] == norm], key=lambda k: r[k][0])
                print(f"{gname:8s} {setting:6s} {tag:38s} {norm:4s} val-best β={best[1]:<4} val {r[best][0]:.4f} test {r[best][1]:.4f}"
                      f"   (β=0 test {r[(norm, 0)][1]:.4f}; best test over β {max(r[k][1] for k in r if k[0]==norm):.4f})")
