"""Markdown table from exp/g-arch/summary_{base,fusion}_{emb}.csv (run summarize.py first)."""
import sys
import pandas as pd

emb = sys.argv[1] if len(sys.argv) > 1 else "ft-gte-large_s0"
b = pd.read_csv(f"exp/g-arch/summary_base_{emb}.csv", index_col=0)
f = pd.read_csv(f"exp/g-arch/summary_fusion_{emb}.csv", index_col=0)
print("| Model | val acc | test acc | test F1 | cold acc | cold F1 | hop 1 / 2 / 3 / 4+ | fused val | fused test | fused F1 "
      "| fused cold | fused hop 4+ |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|")
for v, r in b.iterrows():
    s = (f"| {v} | {r.val_acc:.4f} | {r.test_acc:.4f} ± {r.test_acc_std:.4f} | {r.test_f1:.3f} | {r.cold_test_acc:.4f} "
         f"| {r.cold_test_f1:.3f} | {r.hop1:.3f} / {r.hop2:.3f} / {r.hop3:.3f} / {r['hop4+']:.3f} |")
    if v in f.index:
        q = f.loc[v]
        s += f" {q.val:.4f} | {q.test:.4f} ± {q.test_std:.4f} | {q.f1:.3f} | {q.cold_test:.4f} | {q['hop4+']:.3f} |"
    else:
        s += " | | | | | |"
    print(s)
