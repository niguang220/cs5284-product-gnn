"""Comparison table over the 3 fine-tuned encoder seeds (each = mean of 3 GNN seeds): mean ± std over encoders.

Run summarize.py for each encoder first. Prints a markdown table.
"""
import pandas as pd

EMBS = ["ft-gte-large_s0", "ft-gte-large_s1", "ft-gte-large_s2"]
b = pd.concat([pd.read_csv(f"exp/g-arch/summary_base_{e}.csv", index_col=0).assign(emb=e) for e in EMBS])
f = pd.concat([pd.read_csv(f"exp/g-arch/summary_fusion_{e}.csv", index_col=0).assign(emb=e) for e in EMBS])
n = b.groupby(level=0).size()
models = [m for m in n.index if n[m] == 3]
cols_b = ["val_acc", "test_acc", "test_f1", "cold_test_acc", "hop4+"]
cols_f = ["val", "test", "f1", "cold_test", "hop4+"]
B = b.loc[models].groupby(level=0)[cols_b].agg(["mean", "std"])
F = f.loc[models].groupby(level=0)[cols_f].agg(["mean", "std"])
B = B.sort_values(("test_acc", "mean"))
fmt = lambda t, c: f"{t[(c, 'mean')]:.4f} ± {t[(c, 'std')]:.4f}"
print("| Model | val | test acc | test F1 | cold acc | hop 4+ | fused val | fused test | fused F1 | fused cold | fused hop 4+ |")
print("|---|---|---|---|---|---|---|---|---|---|---|")
for m in B.index:
    r, q = B.loc[m], F.loc[m]
    print(f"| {m} | {r[('val_acc', 'mean')]:.4f} | {fmt(r, 'test_acc')} | {r[('test_f1', 'mean')]:.3f} | {fmt(r, 'cold_test_acc')} "
          f"| {r[('hop4+', 'mean')]:.3f} | {q[('val', 'mean')]:.4f} | {fmt(q, 'test')} | {q[('f1', 'mean')]:.3f} | {fmt(q, 'cold_test')} "
          f"| {q[('hop4+', 'mean')]:.3f} |")
