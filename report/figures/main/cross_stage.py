"""Cross-stage comparison on common slices: best model of each stage, from saved predictions (no retraining).
Stage 1: GCN on OGB features (3 seeds); Stage 2: fine-tuned gte-large + MLP (3 enc x 3 seeds);
Stage 3: NAGphormer on fine-tuned gte-large (released encoders, 3 x 3); Stage 4: final pipeline (3 x 3); GPT-3.5 for reference.
Test labels are used descriptively only."""
import csv, json, os, sys
from pathlib import Path
import numpy as np, pandas as pd, scipy.sparse as sp
HERE = Path(__file__).resolve().parent; REPO = HERE.parents[2]; sys.path.insert(0, str(REPO)); os.chdir(REPO)
from src.data import load_products
from src.train import masked_metrics
import torch
g, text = load_products()
y = g.y.numpy(); tr, va, te = [m.numpy() for m in (g.train_mask, g.val_mask, g.test_mask)]
N, K = len(y), 47

def ogb_preds():
    out = []
    for sd in ["seed_5284", "seed_5285", "seed_5286"]:
        d = pd.read_csv(f"results/1_gnn_ogb/02_multi_seed/GCN/per_seed/{sd}/1_ogb_predictions.csv")
        p = np.full(N, -1); p[d.node_id.values] = d.predicted_label.values
        assert (y[d.node_id.values] == d.true_label.values).all()
        out.append(p)
    return out
probs = lambda pat: [np.load(pat.format(e=e, s=s)).argmax(1) for e in range(3) for s in range(3)]
gpt = np.array([int(r[0]) for r in csv.reader(open("outputs/tape_gpt_preds/ogbn-products.csv"))])
models = {
    "S1 GCN, OGB features": ogb_preds(),
    "S2 gte-large + MLP": probs("outputs/probs/garch-MLP@ft-gte-large_s{e}_normal_s{s}.npy"),
    "S3 gte-large + NAGphormer": probs("outputs/probs/NAGphormer@ft-gte-large-r_s{e}_normal_s{s}.npy"),
    "S4 final pipeline": probs("outputs/final/NAGphormer@ft-gte-large-r_s{e}_normal_s{s}.npy"),
    "GPT-3.5": [gpt],
}
ei = g.edge_index.numpy()
A = sp.csr_matrix((np.ones(ei.shape[1]), (ei[0], ei[1])), shape=(N, N))
deg = np.asarray(A.sum(1)).ravel()
dist = np.full(N, np.inf); dist[tr] = 0; fr = tr.copy(); h = 0
while fr.any():
    h += 1; fr = ((A @ fr.astype(float)) > 0) & np.isinf(dist); dist[fr] = h
notitle = np.array([not isinstance(t, str) or not t.strip() for t in text["title"].tolist()])
rare = np.isin(y, [12, 18, 21, 16])
ntr = np.bincount(y[tr], minlength=K)
unl = (ntr[y] == 0) | (y == 24)
pri = lambda m: np.bincount(y[m], minlength=K) / m.sum()
p_va = pri(va)
slices = {"hop 1": te & (dist == 1), "hop 2": te & (dist == 2), "hop 3": te & (dist == 3),
          "hop 4+": te & (dist >= 4) & np.isfinite(dist),
          "degree 1": te & (deg == 1), "degree 2+": te & (deg >= 2),
          "with title": te & ~notitle, "no title": te & notitle,
          "train-rare classes": te & rare, "other learnable": te & ~rare & ~unl, "unlearnable": te & unl}
rows = []
for name, preds in models.items():
    acc = {k: [] for k in ["val", "test", "test_at_val_mix", "macro_f1"] + list(slices)}
    for p in preds:
        acc["val"].append((p[va] == y[va]).mean()); acc["test"].append((p[te] == y[te]).mean())
        r = np.full(K, np.nan)
        for c in range(K):
            m = te & (y == c)
            if m.sum(): r[c] = (p[m] == c).mean()
        ok = ~np.isnan(r) & (p_va > 0); acc["test_at_val_mix"].append(np.sum(p_va[ok] * r[ok]) / p_va[ok].sum())
        pp = torch.from_numpy(np.where(p < 0, 0, p))
        acc["macro_f1"].append(masked_metrics(torch.from_numpy(y), pp, torch.from_numpy(te))["macro_f1"])
        for k, m in slices.items():
            acc[k].append((p[m] == y[m]).mean())
    row = {k: float(np.mean(v)) for k, v in acc.items()}; row["model"] = name; rows.append(row)
df = pd.DataFrame(rows).set_index("model")
df["within"] = 100 * (df.val - df.test_at_val_mix); df["classmix"] = 100 * (df.test_at_val_mix - df.test)
pd.set_option("display.width", 250)
print((df * 1).round(3).T.to_string())
print({k: int(m.sum()) for k, m in slices.items()})
df.to_csv(HERE / "cross_stage.csv")
