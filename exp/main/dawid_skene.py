"""Dawid-Skene EM: re-estimate each LLM's confusion matrix on the unlabelled test nodes, with the model as the prior.

E: q_i(y) ∝ p_i(y) · Π_k C_k[g_ik, y]          (p_i = temperature-scaled model posterior)
M: C_k[g, y] ∝ s·C_k^lab[g, y] + Σ_{i∈test} q_i(y)·[g_ik = g]   (labelled-node counts as a Dirichlet prior of strength s)
s (prior strength) and the iteration count are chosen on val; val nodes are only scored, never used in the M-step.
"""
import csv, sys
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(REPO))
from src.data import load_products
from src.label_shift import K, confusion, fit_temperature, temper
g, _ = load_products(); y = g.y.numpy(); tr, va, te = g.train_mask.numpy(), g.val_mask.numpy(), g.test_mask.numpy()
lab = tr | va
ans = {"gpt": np.array([int(r[0]) for r in csv.reader(open(REPO / "outputs/tape_gpt_preds/ogbn-products.csv"))]),
       "qwen": np.load(REPO / "outputs/llm_answers/qwen35_9b_knowndefs.npy")}
counts = {}
for k, a in ans.items():
    c = np.zeros((K + 1, K)); np.add.at(c, (a[lab] % (K + 1), y[lab]), 1); counts[k] = c
tag = sys.argv[1] if len(sys.argv) > 1 else "GT@ft-gte-large_s0"
for combo in [("gpt",), ("gpt", "qwen")]:
    res = {}
    for s in range(3):
        p = np.load(REPO / f"outputs/probs/{tag}_normal_s{s}.npy").astype(float); p = temper(p, fit_temperature(p, y, va))
        for strength in [0.25, 1.0, 4.0]:
            Cs = {k: confusion(ans[k], y, lab) for k in combo}
            for it in range(6):
                L = np.ones_like(p)
                for k in combo: L *= Cs[k][ans[k] % (K + 1)]
                q = p * L; q /= q.sum(1, keepdims=True)
                pr = q.argmax(1)
                res.setdefault((strength, it), []).append(((pr[va] == y[va]).mean(), (pr[te] == y[te]).mean()))
                for k in combo:  # M-step on test nodes only
                    c = np.zeros((K + 1, K)); np.add.at(c, ans[k][te] % (K + 1), q[te])
                    n_lab = counts[k].sum(0, keepdims=True); n_te = c.sum(0, keepdims=True)
                    # scale the labelled counts to `strength` x the test mass of each class, plus Laplace 1
                    prior = counts[k] / np.maximum(n_lab, 1) * strength * np.maximum(n_te, 1)
                    cc = prior + c + 1.0
                    Cs[k] = cc / cc.sum(0, keepdims=True)
    best = max(res, key=lambda k: np.mean([r[0] for r in res[k]]))
    print(tag, combo)
    for k, v in sorted(res.items()):
        v = np.array(v); print(f"  strength={k[0]:<5} iter={k[1]}  val {v[:,0].mean():.4f} test {v[:,1].mean():.4f}{'  <- val' if k == best else ''}")
