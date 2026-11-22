"""Architecture ensemble: average the softmax outputs of the k base models with the highest mean val acc
(exp/g-arch/summary_base_{emb}.csv), per GNN seed and setting. k is chosen afterwards on post-fusion val acc.

Usage: python exp/g-arch/ensemble.py EMB K [K ...]   -> outputs/probs/garch-ENS-top{K}@{EMB}_{normal,cold}_s{seed}.npy
       python exp/g-arch/ensemble.py EMB --members A,B,C NAME
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
P = HERE.parents[1] / "outputs" / "probs"
emb = sys.argv[1]
if "--members" in sys.argv:
    groups = {sys.argv[sys.argv.index("--members") + 2]: sys.argv[sys.argv.index("--members") + 1].split(",")}
else:
    base = pd.read_csv(HERE / f"summary_base_{emb}.csv", index_col=0).sort_values("val_acc", ascending=False)
    groups = {f"ENS-top{k}": list(base.index[:int(k)]) for k in sys.argv[2:]}
for name, members in groups.items():
    for s in ["normal", "cold"]:
        for seed in range(3):
            p = np.mean([np.load(P / f"garch-{m}@{emb}_{s}_s{seed}.npy") for m in members], 0)
            np.save(P / f"garch-{name}@{emb}_{s}_s{seed}.npy", p.astype(np.float32))
    (P / f"garch-{name}@{emb}.json").write_text(json.dumps({"members": members, "selection": "top-k by mean val acc"}))
    print(name, members)
