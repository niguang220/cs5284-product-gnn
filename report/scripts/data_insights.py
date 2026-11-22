"""Dataset statistics for the report (descriptive only; nothing is selected).

Usage: python report/scripts/data_insights.py > report/tables/data_insights.txt
Covers how the subset was sampled (hop layers from the 128 seed nodes), degree by split, hops from test products to
the nearest training product and the class-mix distance per hop, edges by split pair, title-less products and the
MISSING class, and classes absent from training. Test labels are used only for these descriptive counts.
"""
import sys
from pathlib import Path
import numpy as np
import scipy.sparse as sp
import torch
from scipy.sparse.csgraph import shortest_path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from src.data import load_products  # noqa: E402

g, t = load_products()
raw = torch.load(REPO / "data/TAPE/ogbn-products_subset.pt")
N = g.num_nodes
y = g.y.numpy()
tr, va, te = (m.numpy() for m in (g.train_mask, g.val_mask, g.test_mask))
ei = g.edge_index.numpy()
A = sp.csr_matrix((np.ones(ei.shape[1]), (ei[0], ei[1])), shape=(N, N))
A.data[:] = 1
deg = np.asarray(A.sum(1)).ravel()

# 1. hop layer from the 128 seeds, node-id order, split share per layer
hop = shortest_path(A, unweighted=True, indices=np.arange(128), directed=False).min(0).astype(int)
print("1. hop layer sizes:", np.bincount(hop).tolist())
bounds = [int(np.where(hop == h)[0].min()) for h in range(4)] + [N]
ordered = all((hop[bounds[h]:bounds[h + 1]] == h).all() for h in range(4))
print("   node ids sorted by hop layer:", ordered, "| first id of each layer:", bounds[:4])
for h in range(4):
    m = hop == h
    print(f"   hop {h}: n={m.sum():6d} train {tr[m].mean():.1%} val {va[m].mean():.1%} test {te[m].mean():.1%}"
          f" mean degree {deg[m].mean():.2f}")

# 2. degree by split
for name, m in [("train", tr), ("val", va), ("test", te)]:
    print(f"2. degree {name}: mean {deg[m].mean():.2f}, median {np.median(deg[m]):.0f}, degree-1 share {(deg[m] == 1).mean():.1%}")

# 3. distance to the nearest train node, for test nodes: BFS from a virtual node linked to every train node
S = sp.vstack([A, sp.csr_matrix((np.ones(tr.sum()), (np.zeros(tr.sum(), int), np.where(tr)[0])), shape=(1, N))])
S = sp.hstack([S, sp.csr_matrix((N + 1, 1))]).tocsr()
S = S + S.T
dist = shortest_path(S, unweighted=True, indices=[N], directed=False)[0][:N] - 1
dt = dist[te]
buckets = {"1": (dt == 1).sum(), "2": (dt == 2).sum(), "3": (dt == 3).sum(), "4+": ((dt >= 4) & np.isfinite(dt)).sum(),
           "unreachable": (~np.isfinite(dt)).sum()}
print("3. test nodes by hops to nearest train node:", {k: int(v) for k, v in buckets.items()})
print(f"   test nodes with a train neighbour: {(dt == 1).mean():.1%}")



def mix(m):
    c = np.bincount(y[m], minlength=47).astype(float)
    return c / c.sum()


for name, m in [("1", te & (dist == 1)), ("2", te & (dist == 2)), ("3", te & (dist == 3)), ("4+", te & (dist >= 4) & np.isfinite(dist))]:
    print(f"   label TV to train, test at {name} hops: {0.5 * np.abs(mix(m) - mix(tr)).sum():.2f}")

# 4. edges by split pair, cold start
s = np.where(tr, 0, np.where(va, 1, 2))
u = ei[0] < ei[1]
pairs = {}
for a, b in zip(s[ei[0][u]], s[ei[1][u]]):
    k = tuple(sorted((a, b)))
    pairs[k] = pairs.get(k, 0) + 1
names = ["train", "val", "test"]
print("4. undirected edges by split pair:", {f"{names[a]}-{names[b]}": v for (a, b), v in sorted(pairs.items())})
touch = te[ei[0][u]] | te[ei[1][u]]
print(f"   edges touching a test node: {touch.sum()} of {u.sum()} ({touch.mean():.1%})")

# 5. title-less products and MISSING (descriptive)
nt = (t["title"].isna() | (t["title"].astype(str).str.strip() == "")).to_numpy()
for name, m in [("train", tr), ("val", va), ("test", te)]:
    k = m & nt
    print(f"5. {name}: title-less {k.sum()} ({k.sum() / m.sum():.1%}); MISSING among them {(y[k] == 24).mean():.1%};"
          f" MISSING total {(m & (y == 24)).sum()}, of which title-less {(m & (y == 24) & nt).sum()}")
print("   class 42 (Baby) by split:", int((tr & (y == 42)).sum()), int((va & (y == 42)).sum()), int((te & (y == 42)).sum()))
unseen = ~np.isin(np.arange(47), np.unique(y[tr]))
print(f"   test nodes in classes absent from train: {np.isin(y[te], np.where(unseen)[0]).sum()}"
      f" ({np.isin(y[te], np.where(unseen)[0]).mean():.1%}); with MISSING: "
      f"{(np.isin(y[te], np.where(unseen)[0]) | (y[te] == 24)).mean():.1%}")
