"""Accuracy slices of the saved final test predictions, for the analysis section of the report.

No model is re-run and nothing is selected. Reads results/4_fusion/final_test_predictions.npz (argmax predictions per
run, written by scripts/run_final_eval.py) and writes report/tables/report_slices.csv: mean accuracy over runs and its
standard deviation (sd, empty for a single run) per (arm, setting, method, slice). Slices: hops to the nearest training product, degree, title present or not, seen
classes, and per-class recall for a few classes.

Usage: python report/scripts/report_slices.py
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.csgraph import shortest_path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from src.data import load_products  # noqa: E402

g, t = load_products()
N = g.num_nodes
y = g.y.numpy()
tr, te = g.train_mask.numpy(), g.test_mask.numpy()
ei = g.edge_index.numpy()
A = sp.csr_matrix((np.ones(ei.shape[1]), (ei[0], ei[1])), shape=(N, N)); A.data[:] = 1
deg = np.asarray(A.sum(1)).ravel()
# hops to the nearest train node (multi-source BFS through a virtual node)
S = sp.vstack([A, sp.csr_matrix((np.ones(tr.sum()), (np.zeros(tr.sum(), int), np.where(tr)[0])), shape=(1, N))])
S = sp.hstack([S, sp.csr_matrix((N + 1, 1))]).tocsr(); S = S + S.T
hops = shortest_path(S, unweighted=True, indices=[N], directed=False)[0][:N] - 1
no_title = (t["title"].isna() | (t["title"].astype(str).str.strip() == "")).to_numpy()
seen = np.isin(y, np.unique(y[tr]))
names = pd.read_csv(REPO / "data/TAPE/labelidx2productcategory.csv.gz")["product category"].tolist()
cid = {n: i for i, n in enumerate(names)}

z = np.load(REPO / "results" / "4_fusion" / "final_test_predictions.npz")
nodes = z["nodes"]
yt = y[nodes]
slices = {
    "all": np.ones(len(nodes), bool),
    "hop1": hops[nodes] == 1, "hop2": hops[nodes] == 2, "hop3": hops[nodes] == 3,
    "hop4+": (hops[nodes] >= 4) & np.isfinite(hops[nodes]),
    "deg1": deg[nodes] == 1, "deg2": deg[nodes] == 2, "deg3-5": (deg[nodes] >= 3) & (deg[nodes] <= 5),
    "deg6+": deg[nodes] >= 6,
    "titled": ~no_title[nodes], "no_title": no_title[nodes],
    "seen": seen[nodes], "unseen_or_missing": ~seen[nodes] | (yt == 24),
}
for n in ["Books", "Electronics", "Automotive", "Office Products", "Tools & Home Improvement", "Cell Phones & Accessories"]:
    slices["recall:" + n] = yt == cid[n]
rows = []
for key in z.files:
    if key == "nodes":
        continue
    arm, lm, seed, setting, method = key.split("|")
    pred = z[key].astype(int)
    ok = pred == yt
    for s, m in slices.items():
        rows.append((arm, lm, seed, setting, method, s, int(m.sum()), float(ok[m].mean()) if m.any() else np.nan))
df = pd.DataFrame(rows, columns=["arm", "lm_seed", "seed", "setting", "method", "slice", "n", "acc"])
out = df.groupby(["arm", "setting", "method", "slice"], sort=False).agg(n=("n", "first"), runs=("acc", "size"),
                                                                       acc=("acc", "mean"), sd=("acc", "std")).reset_index()
out.to_csv(REPO / "report" / "tables" / "report_slices.csv", index=False)

show = [("MiniLM + MLP", "normal", "model"), ("MiniLM + GCN", "normal", "model"), ("MiniLM + GraphSAGE", "normal", "model"),
        ("MiniLM + GCN", "cold", "model"), ("MiniLM + GraphSAGE", "cold", "model"),
        ("DeBERTa + MLP", "normal", "model"), ("DeBERTa + GCN", "normal", "model"), ("DeBERTa + GCN", "normal", "blend"),
        ("DeBERTa + GCN", "normal", "cond"), ("DeBERTa + GCN", "cold", "model"), ("DeBERTa + GCN", "cold", "blend"),
        ("ensemble + GraphSAGE", "normal", "model"), ("ensemble + GraphSAGE", "normal", "cond"), ("GPT answer", "normal", "model")]
p = out.set_index(["arm", "setting", "method", "slice"])["acc"].unstack("slice")
cols = list(slices)
pd.set_option("display.width", 250)
print("slice sizes:", {s: int(m.sum()) for s, m in slices.items()})
print((p.loc[show, cols[:13]] * 100).round(1).to_string())
print((p.loc[show, cols[13:]] * 100).round(1).to_string())
