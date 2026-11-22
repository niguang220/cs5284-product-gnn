"""Test accuracy by hop distance to the nearest train node, and on the far / no-title slices, for the val-selected stack.

Usage: python exp/main/breakdown.py TAG LLMS BETA GAMMA   (LLMS like gpt or gpt+qwen)
"""
import csv, sys
from pathlib import Path
import numpy as np, scipy.sparse as sp, torch
REPO = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(REPO))
from src.data import load_products
from src.improve import nearest_in_pool
from src.label_shift import K, compatibility, confusion, fit_temperature, fuse, neighbour_log_messages, temper
from src.text_clean import clean_text
tag, combo, beta, gamma = sys.argv[1], sys.argv[2].split("+"), float(sys.argv[3]), float(sys.argv[4])
g, text = load_products(); y = g.y.numpy(); N = len(y)
tr, va, te = g.train_mask.numpy(), g.val_mask.numpy(), g.test_mask.numpy()
llm = {"gpt": np.array([int(r[0]) for r in csv.reader(open(REPO / "outputs/tape_gpt_preds/ogbn-products.csv"))]),
       "qwen": np.load(REPO / "outputs/llm/qwen35_9b.npy")}
C = {k: confusion(llm[k], y, tr | va) for k in combo}
e = np.ones((N, K))
for k in combo: e *= C[k][llm[k] % (K + 1)]
e /= e.sum(1, keepdims=True); e[tr] = np.eye(K)[y[tr]]
ei = g.edge_index.numpy()
x = torch.nn.functional.normalize(torch.from_numpy(np.load(REPO / "outputs/embeddings/original__all-MiniLM-L6-v2.npy")), dim=1)
kn = nearest_in_pool(torch.arange(N), x, torch.arange(N), 5).numpy()
M = beta * neighbour_log_messages(ei, N, e, compatibility(ei, y, tr)) + gamma * neighbour_log_messages(kn, N, e, compatibility(kn, y, tr))
A = sp.coo_matrix((np.ones(ei.shape[1]), (ei[0], ei[1])), shape=(N, N)).tocsr()
d = np.full(N, 99); fr = np.where(tr)[0]; d[fr] = 0; k = 0
while len(fr):
    k += 1; nb = np.unique(A[fr].indices); nb = nb[d[nb] == 99]; d[nb] = k; fr = nb
notitle = text.title.map(clean_text).str.len().values == 0
slices = {"hop1": d == 1, "hop2": d == 2, "hop3": d == 3, "hop4+": (d >= 4) & (d < 99), "no title": notitle}
out = {}
for s in range(3):
    p = np.load(REPO / f"outputs/probs/{tag}_normal_s{s}.npy").astype(float); p = temper(p, fit_temperature(p, y, va))
    stages = {"base": p, "fuse": fuse(p, [llm[k] for k in combo], [C[k] for k in combo])}
    stages["full"] = np.exp(np.log(stages["fuse"]) + M)
    for st, q in stages.items():
        pr = q.argmax(1)
        for nm, m in slices.items():
            out.setdefault((st, nm), []).append((pr[te & m] == y[te & m]).mean())
print(tag, combo, beta, gamma)
print("slice      n      " + "  ".join(f"{st:>6s}" for st in ["base", "fuse", "full"]))
for nm, m in slices.items():
    print(f"{nm:9s} {int((te & m).sum()):6d}  " + "  ".join(f"{np.mean(out[(st, nm)]):.3f}" for st in ["base", "fuse", "full"]))
