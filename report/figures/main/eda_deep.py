"""Deeper EDA for the report: accuracy decomposition under label shift, ceilings, shift by hop, val power, LLM as a
prior-free signal. Test labels are used descriptively only."""
import csv, json, os, sys
from pathlib import Path
import numpy as np, pandas as pd, scipy.sparse as sp
HERE = Path(__file__).resolve().parent; REPO = HERE.parents[2]; sys.path.insert(0, str(REPO)); os.chdir(REPO)
from src.data import load_products
g, text = load_products()
y = g.y.numpy(); tr, va, te = [m.numpy() for m in (g.train_mask, g.val_mask, g.test_mask)]
K = 47; N = len(y)
names = pd.read_csv("data/TAPE/labelidx2productcategory.csv.gz")["product category"].tolist()
pri = lambda m: np.bincount(y[m], minlength=K) / m.sum()
p_tr, p_va, p_te = pri(tr), pri(va), pri(te)
tv = lambda a, b: 0.5 * np.abs(a - b).sum()
out = {}

# ---- 1. ceilings
ntr = np.bincount(y[tr], minlength=K)
unseen = te & (ntr[y] == 0)
out["ceiling"] = {"unseen_share": unseen.mean() / te.mean(), "n_unseen": int(unseen.sum()),
                  "missing_share": (te & (y == 24)).sum() / te.sum(),
                  "max_acc_unseen": 1 - unseen.sum() / te.sum(),
                  "max_acc_unseen_missing": 1 - (unseen | (te & (y == 24))).sum() / te.sum(),
                  "lt50_share": (te & (ntr[y] < 50)).sum() / te.sum(), "lt100_share": (te & (ntr[y] < 100)).sum() / te.sum()}

# ---- predictions
def avg_pred(paths):
    return [np.load(p).argmax(1) for p in paths]
gpt = np.array([int(r[0]) for r in csv.reader(open("outputs/tape_gpt_preds/ogbn-products.csv"))])
gpt = np.where((gpt >= 0) & (gpt < K), gpt, -1)
models = {
    "MiniLM + MLP": avg_pred([f"outputs/probs/MLP_normal_s{s}.npy" for s in range(3)]),
    "MiniLM + GCN": avg_pred([f"outputs/probs/GCN_normal_s{s}.npy" for s in range(3)]),
    "MiniLM + GT": avg_pred([f"outputs/probs/GT_normal_s{s}.npy" for s in range(3)]),
    "gte-large + NAGphormer": avg_pred([f"outputs/probs/NAGphormer@ft-gte-large-r_s{e}_normal_s{s}.npy" for e in range(3) for s in range(3)]),
    "GPT-3.5 answer": [gpt],
    "Final pipeline": avg_pred([f"outputs/final/NAGphormer@ft-gte-large-r_s{e}_normal_s{s}.npy" for e in range(3) for s in range(3)]),
}
def recall(pred, m):
    r = np.full(K, np.nan)
    for c in range(K):
        mm = m & (y == c)
        if mm.sum(): r[c] = (pred[mm] == c).mean()
    return r
dec = []
for name, preds in models.items():
    rows = []
    for pr in preds:
        acc_va = (pr[va] == y[va]).mean(); acc_te = (pr[te] == y[te]).mean()
        r_te, r_va = recall(pr, te), recall(pr, va)
        # test products, reweighted to the val class mix (classes present in both)
        ok = ~np.isnan(r_te) & (p_va > 0)
        te_at_val_mix = np.sum(p_va[ok] * r_te[ok]) / p_va[ok].sum()
        # predicted test acc from val recall and the test class mix (unseen-in-val classes: recall 0)
        pred_te = np.nansum(p_te * np.nan_to_num(r_va, nan=0.0))
        # test acc on train mix
        ok2 = ~np.isnan(r_te) & (p_tr > 0)
        te_at_train_mix = np.sum(p_tr[ok2] * r_te[ok2]) / p_tr[ok2].sum()
        pd_te = np.bincount(pr[te][pr[te] >= 0], minlength=K) / te.sum()
        rows.append(dict(val=acc_va, test=acc_te, test_at_val_mix=te_at_val_mix, test_at_train_mix=te_at_train_mix,
                         pred_from_val=pred_te, tv_pred_test=tv(pd_te, p_te), tv_pred_train=tv(pd_te, p_tr)))
    d = pd.DataFrame(rows).mean(); d["model"] = name; dec.append(d)
dec = pd.DataFrame(dec).set_index("model")
print(dec.round(4).to_string())
out["decomposition"] = dec.round(4).reset_index().to_dict(orient="records")

# ---- 3. by hop
ei = g.edge_index.numpy()
A = sp.csr_matrix((np.ones(ei.shape[1]), (ei[0], ei[1])), shape=(N, N))
dist = np.full(N, np.inf); dist[tr] = 0; fr = tr.copy(); h = 0
while fr.any():
    h += 1; fr = ((A @ fr.astype(float)) > 0) & np.isinf(dist); dist[fr] = h
shifted = np.isin(y, [12, 18, 21, 16])
hop = []
for nm, m in {"1": dist == 1, "2": dist == 2, "3": dist == 3, "4+": (dist >= 4) & np.isfinite(dist)}.items():
    mm = te & m
    hop.append(dict(hop=nm, n=int(mm.sum()), share=mm.sum() / te.sum(), tv=tv(pri(mm), p_tr),
                    shifted_share=shifted[mm].mean(), unseen_share=unseen[mm].mean(), gpt=(gpt[mm] == y[mm]).mean(),
                    deg=np.asarray(A.sum(1)).ravel()[mm].mean()))
hop = pd.DataFrame(hop); print(hop.round(3).to_string()); out["hop"] = hop.round(4).to_dict(orient="records")

# ---- 4. graph signal: homophily by split pair, train-neighbour vote accuracy on val vs test
r, c = ei[0], ei[1]
split = np.where(tr, 0, np.where(va, 1, 2))
hom = {}
for a, b, nm in [(0, 0, "train-train"), (0, 2, "train-test"), (2, 2, "test-test"), (0, 1, "train-val")]:
    m = ((split[r] == a) & (split[c] == b)) | ((split[r] == b) & (split[c] == a))
    hom[nm] = (float((y[r][m] == y[c][m]).mean()), int(m.sum() // (1 if a != b else 2)))
print(hom); out["homophily"] = hom
Y = sp.csr_matrix((np.ones(tr.sum()), (np.where(tr)[0], y[tr])), shape=(N, K))
votes = A @ Y
has = np.asarray(votes.sum(1)).ravel() > 0
vote = np.asarray(votes.argmax(1)).ravel()
vv = {}
for nm, m in {"val": va, "test": te}.items():
    vv[nm] = dict(share_with_train_nb=float(has[m].mean()), vote_acc=float((vote[m & has] == y[m & has]).mean()))
# label propagation-ish: majority of all labelled neighbours within 2 hops
print(vv); out["train_vote"] = vv

# ---- 5. val power for the shifted classes
pw = []
for cls in [12, 18, 21, 16]:
    pw.append(dict(cls=names[cls], n_train=int(ntr[cls]), n_val=int((va & (y == cls)).sum()), n_test=int((te & (y == cls)).sum()),
                   share_test=float(p_te[cls]), share_val=float(p_va[cls])))
pw = pd.DataFrame(pw); print(pw.to_string()); out["val_power"] = pw.to_dict(orient="records")
print("shifted classes: val n", int((va & shifted).sum()), "test share", float(shifted[te].mean()), "val share", float(shifted[va].mean()))
out["shifted_val_n"] = int((va & shifted).sum()); out["shifted_test_share"] = float(shifted[te].mean()); out["shifted_val_share"] = float(shifted[va].mean())
# SE on a weighted-val estimate of test accuracy restricted to these classes, at p = 0.6
w = np.where(va, p_te[y] / np.maximum(p_va[y], 1e-12), 0)
ess = w[va].sum() ** 2 / (w[va] ** 2).sum()
out["weighted_val_ess"] = float(ess); print("ESS of test-weighted val:", ess)

# ---- 6. LLM as a prior-free signal
gp = {}
for nm, m in {"train": tr, "val": va, "test": te}.items():
    gp[nm] = float((gpt[m] == y[m]).mean())
gd_te = np.bincount(gpt[te][gpt[te] >= 0], minlength=K) / te.sum()
gp["tv_gptdist_test"] = tv(gd_te, p_te)
print("GPT acc by split", gp); out["gpt"] = gp
notitle = np.array([not isinstance(t, str) or not t.strip() for t in text["title"].tolist()])
nt = {nm: dict(missing_share=float((y[m & notitle] == 24).mean()), gpt=float((gpt[m & notitle] == y[m & notitle]).mean()), n=int((m & notitle).sum()))
      for nm, m in {"train": tr, "val": va, "test": te}.items()}
print("no title", nt); out["notitle"] = nt
json.dump(out, open(HERE / "eda_deep.json", "w"), indent=1, default=float)
