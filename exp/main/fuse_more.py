"""Variants of LLM fusion: GPT's 2nd answer, graph smoothing of the fused posterior. Choices on val only."""
import csv, sys
from pathlib import Path
import numpy as np, scipy.sparse as sp, torch
REPO = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(REPO))
from src.data import load_products
from src.label_shift import K, confusion, fit_temperature, fuse, smooth, temper
from src.train import masked_metrics
graph, _ = load_products(); y = graph.y.numpy(); N = len(y)
tr, va, te = graph.train_mask.numpy(), graph.val_mask.numpy(), graph.test_mask.numpy()
rows = [[int(v) for v in r] for r in csv.reader(open(REPO / "outputs/tape_gpt_preds/ogbn-products.csv"))]
g1 = np.array([r[0] for r in rows])
g2 = np.array([next((v for v in r if v != r[0]), -1) for r in rows])  # first different later answer
C1, C2 = confusion(g1, y, tr | va), confusion(g2, y, tr | va)
ei = graph.edge_index.numpy()
A = sp.coo_matrix((np.ones(ei.shape[1]), (ei[0], ei[1])), shape=(N, N)).tocsr(); A.data[:] = 1
d = np.asarray(A.sum(1)).ravel(); dinv = np.where(d > 0, 1 / np.sqrt(np.maximum(d, 1)), 0)
An = sp.diags(dinv) @ A @ sp.diags(dinv)
yt = torch.from_numpy(y)
def m(p, mask): return masked_metrics(yt, torch.from_numpy(p.argmax(1)), torch.from_numpy(mask))
for model in ["SAGE", "GT"]:
    res = {}
    for s in range(3):
        p = np.load(REPO / f"outputs/probs/{model}_normal_s{s}.npy").astype(np.float64)
        p = temper(p, fit_temperature(p, y, va))
        f1 = fuse(p, [g1], [C1])
        cand = {"fuse g1": f1, "fuse g1+g2": fuse(p, [g1, g2], [C1, C2])}
        for a in [0.3, 0.5, 0.7, 0.9]:
            cand[f"fuse g1 -> smooth a={a}"] = smooth(f1, An, y, tr, alpha=a)
            cand[f"smooth a={a} -> fuse g1"] = fuse(smooth(p, An, y, tr, alpha=a), [g1], [C1])
        for k, q in cand.items():
            v, t = m(q, va), m(q, te)
            res.setdefault(k, []).append((v["acc"], t["acc"], t["macro_f1"]))
    print(model)
    for k, v in res.items():
        v = np.array(v); print(f"  {k:28s} val {v[:,0].mean():.4f} test {v[:,1].mean():.4f} f1 {v[:,2].mean():.4f}")
