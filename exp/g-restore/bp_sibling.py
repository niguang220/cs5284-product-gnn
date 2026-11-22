"""BP with an extra, degree-normalised channel over 2-hop "sibling" pairs of the given graph (two products bought
with the same product). Uses only the given TAPE graph. GPT only; β (direct), γ2 (siblings) and temperature on val."""
import csv
import sys
from pathlib import Path
import numpy as np
import scipy.sparse as sp
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from src.data import load_products
from src.label_shift import K, compatibility, confusion, fit_temperature, fuse, temper, neighbour_log_messages
graph, _ = load_products()
y = graph.y.numpy(); N = len(y)
tr, va, te = graph.train_mask.numpy(), graph.val_mask.numpy(), graph.test_mask.numpy()
gpt = np.array([int(r[0]) for r in csv.reader(open(REPO / "outputs/tape_gpt_preds/ogbn-products.csv"))])
C = confusion(gpt, y, tr | va)
e = C[gpt % (K + 1)]; e = e / e.sum(1, keepdims=True); e[tr] = np.eye(K)[y[tr]]
ei = graph.edge_index.numpy()
A = sp.csr_matrix((np.ones(ei.shape[1]), (ei[0], ei[1])), shape=(N, N)); A.data[:] = 1
M1 = neighbour_log_messages(ei, N, e, compatibility(ei, y, tr))
S = (A @ A).tocsr(); S.setdiag(0); S = S - S.multiply(A); S.eliminate_zeros(); S.data[:] = 1
s2 = np.stack(S.nonzero())
L2 = np.log(np.clip(e @ compatibility(s2, y, tr).T, 1e-9, None))
sdeg = np.maximum(np.asarray(S.sum(1)).ravel(), 1)
M2 = {"mean": (S @ L2) / sdeg[:, None], "sqrt": (S @ L2) / np.sqrt(sdeg)[:, None]}
rows = []
for enc in range(3):
    for seed in range(3):
        p = np.load(REPO / f"outputs/probs/GT@ft-gte-large_s{enc}_normal_s{seed}.npy").astype(np.float64)
        lf = np.log(fuse(temper(p, fit_temperature(p, y, va)), [gpt], [C]))
        for beta in [0, 0.25, 0.5, 1]:
            for norm in ["mean", "sqrt"]:
                for g2 in [0, 0.25, 0.5, 1, 2]:
                    pr = (lf + beta * M1 + g2 * M2[norm]).argmax(1)
                    rows.append((enc, seed, beta, norm, g2, (pr[va] == y[va]).mean(), (pr[te] == y[te]).mean()))
import pandas as pd
df = pd.DataFrame(rows, columns=["enc", "seed", "beta", "norm", "g2", "val", "test"])
g = df.groupby(["enc", "beta", "norm", "g2"])[["val", "test"]].mean().reset_index()
for enc, d in g.groupby("enc"):
    base = d[d.g2 == 0].loc[lambda x: x.val.idxmax()]
    best = d.loc[d.val.idxmax()]
    print(f"enc {enc}: BP only val {base.val:.4f} test {base.test:.4f} | +sibling ({best.norm}, β={best.beta}, γ2={best.g2}) val {best.val:.4f} test {best.test:.4f}")
df.to_csv(REPO / "exp/g-restore/bp_sibling_runs.csv", index=False)
