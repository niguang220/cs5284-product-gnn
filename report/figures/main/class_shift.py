"""Per-class recall / precision change of the final pipeline vs NAGphormer against the class's test/train prior ratio."""
import os, sys, json
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr
HERE = Path(__file__).resolve().parent; REPO = HERE.parents[2]; sys.path.insert(0, str(REPO)); os.chdir(REPO)
from src.data import load_products
g, text = load_products()
y = g.y.numpy(); tr, te = g.train_mask.numpy(), g.test_mask.numpy(); K = 47
def runs(pat): return [np.load(pat.format(e=e, s=s)).argmax(1) for e in range(3) for s in range(3)]
base = runs("outputs/probs/NAGphormer@ft-gte-large-r_s{e}_normal_s{s}.npy")
fin = runs("outputs/final/NAGphormer@ft-gte-large-r_s{e}_normal_s{s}.npy")
ntr = np.bincount(y[tr], minlength=K); nte = np.bincount(y[te], minlength=K)
ptr, pte = ntr / ntr.sum(), nte / nte.sum()
def stats(ps):
    rec, prec, npred = [], [], []
    for p in ps:
        pt, yt = p[te], y[te]
        rec.append([(pt[yt == c] == c).mean() if nte[c] else np.nan for c in range(K)])
        prec.append([(yt[pt == c] == c).mean() if (pt == c).sum() else np.nan for c in range(K)])
        npred.append(np.bincount(pt, minlength=K) / len(pt))
    return np.nanmean(rec, 0), np.nanmean(prec, 0), np.mean(npred, 0)
rb, pb, qb = stats(base); rf, pf, qf = stats(fin)
cls = [c for c in range(K) if ntr[c] >= 5 and nte[c] >= 30 and c != 24]
lr = np.log2(pte[cls] / ptr[cls])
dr = 100 * (rf[cls] - rb[cls]); dp = 100 * (pf[cls] - pb[cls])
out = {"n": len(cls), "rho_recall": float(spearmanr(lr, dr).statistic), "rho_prec": float(spearmanr(lr, dp).statistic),
       "pred_share_tv_base": float(0.5 * np.abs(qb - pte).sum()), "pred_share_tv_final": float(0.5 * np.abs(qf - pte).sum()),
       "pred_share_tv_train": float(0.5 * np.abs(ptr - pte).sum())}
names = pd.read_csv("data/TAPE/labelidx2productcategory.csv.gz") if Path("data/TAPE/labelidx2productcategory.csv.gz").exists() else None
rows = [{"c": int(c), "log2ratio": float(l), "d_recall": float(a), "d_prec": float(b), "test_share": float(pte[c]),
         "rec_b": float(rb[c]), "rec_f": float(rf[c]), "prec_b": float(pb[c]), "prec_f": float(pf[c])} for c, l, a, b in zip(cls, lr, dr, dp)]
pd.DataFrame(rows).to_csv(HERE / "class_shift.csv", index=False)
print(json.dumps(out, indent=1)); print(pd.DataFrame(rows).sort_values("log2ratio").round(2).to_string())
