"""Extra slices for the report: homophily by class prevalence, accuracy by text length. Test labels descriptive only."""
import csv, os, sys, json
from pathlib import Path
import numpy as np, pandas as pd, scipy.sparse as sp
from scipy.stats import spearmanr
HERE = Path(__file__).resolve().parent; REPO = HERE.parents[2]; sys.path.insert(0, str(REPO)); os.chdir(REPO)
from src.data import load_products
g, text = load_products()
y = g.y.numpy(); tr, va, te = [m.numpy() for m in (g.train_mask, g.val_mask, g.test_mask)]
N, K = len(y), 47
ei = g.edge_index.numpy()
A = sp.csr_matrix((np.ones(ei.shape[1]), (ei[0], ei[1])), shape=(N, N)); A.data[:] = 1
deg = np.asarray(A.sum(1)).ravel()
# node homophily: share of neighbours with the same label
Y = sp.csr_matrix((np.ones(N), (np.arange(N), y)), shape=(N, K))
nb_same = np.asarray((A @ Y).multiply(Y).sum(1)).ravel()
h = np.where(deg > 0, nb_same / np.maximum(deg, 1), np.nan)
share = np.bincount(y, minlength=K) / N
cls = [c for c in range(K) if (y == c).sum() >= 20]
hc = np.array([np.nanmean(h[y == c]) for c in cls]); sc = share[cls]
rho = spearmanr(sc, hc)
out = {"homophily_spearman": [float(rho.statistic), float(rho.pvalue)], "n_cls": len(cls)}
rare = [12, 18, 21, 16]
out["rare_homophily"] = {int(c): float(np.nanmean(h[y == c])) for c in rare}
out["books_homophily"] = float(np.nanmean(h[y == 4]))
# test-node homophily with train vs any neighbours for rare classes
out["rare_test_homophily"] = float(np.nanmean(h[te & np.isin(y, rare)]))
out["other_test_homophily"] = float(np.nanmean(h[te & ~np.isin(y, rare)]))
# accuracy by text length (words of title + content)
t = (text["title"].fillna("").astype(str) + " " + text["content"].fillna("").astype(str)).str.split().str.len().values
bins = [0, 10, 30, 80, 200, 10 ** 9]; lab = ["<10", "10–29", "30–79", "80–199", "200+"]
b = np.digitize(t, bins[1:-1])
gpt = np.array([int(r[0]) for r in csv.reader(open("outputs/tape_gpt_preds/ogbn-products.csv"))])
def runs(pat): return [np.load(pat.format(e=e, s=s)).argmax(1) for e in range(3) for s in range(3)]
models = {"MiniLM + GT": runs("outputs/probs/garch-GT@original__all-MiniLM-L6-v2_normal_s{s}.npy".replace("{e}", "")) if False else [np.load(f"outputs/probs/garch-GT@original__all-MiniLM-L6-v2_normal_s{s}.npy").argmax(1) for s in range(3)],
          "NAGphormer": runs("outputs/probs/NAGphormer@ft-gte-large-r_s{e}_normal_s{s}.npy"),
          "final": runs("outputs/final/NAGphormer@ft-gte-large-r_s{e}_normal_s{s}.npy"), "GPT-3.5": [gpt]}
rows = []
for k in range(len(lab)):
    m = te & (b == k)
    r = {"bin": lab[k], "n": int(m.sum()), "share": float(m.sum() / te.sum()),
         "rare_share": float(np.isin(y[m], rare).mean()), "notitle": float(text["title"].fillna("").astype(str).str.strip().eq("").values[m].mean())}
    for name, ps in models.items():
        if ps: r[name] = float(np.mean([(p[m] == y[m]).mean() for p in ps]))
    rows.append(r)
out["length"] = rows
print(json.dumps(out, indent=1))
json.dump(out, open(HERE / "extra_insights.json", "w"), indent=1)

# the group's 34-arm fusion evaluation: base vs neighbour-aware fusion (normal) and blend (cold, the only rule there)
d = pd.read_csv("results/4_fusion/final_test_summary.csv")
p = d.pivot_table(index=["arm", "setting", "subset"], columns="method", values="acc").reset_index()
rows = []
for s, m in [("normal", "cond"), ("cold", "blend")]:
    for sub in ["all", "no_train_nb", "one_train_nb", "two_plus_train_nb"]:
        x = p[(p.setting == s) & (p.subset == sub)].dropna(subset=["model", m])
        for _, r in x.iterrows():
            rows.append({"arm": r.arm, "setting": s, "subset": sub, "base": r.model, "fused": r[m]})
pd.DataFrame(rows).to_csv(HERE / "arms_fusion.csv", index=False)
