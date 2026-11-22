"""Aggregate exp/graph/results.jsonl into a mean ± std table (markdown) per variant."""
import json

import numpy as np
import pandas as pd

from common import EXP

rows = [json.loads(l) for l in open(EXP / "results.jsonl")]
d = pd.DataFrame(rows).drop(columns=["args"], errors="ignore").drop_duplicates(["variant", "seed"], keep="last")
cols = [c for c in ["val_acc", "val_cold_acc", "test_acc", "test_f1", "cold_acc", "cold_f1", "hop1", "hop2", "hop3", "hop4+",
                    "cold_hop3", "cold_hop4+"] if c in d]
g = d.groupby("variant", sort=False)[cols]
mean, std, n = g.mean(), g.std(ddof=0), g.size()
out = mean.applymap(lambda v: f"{v:.4f}" if v == v else "-")
for c in ["val_acc", "test_acc", "test_f1", "cold_acc"]:
    out[c] = [f"{m:.4f}±{s:.4f}" if m == m else "-" for m, s in zip(mean[c], std[c])]
out["n"] = n
out.to_csv(EXP / "summary.csv")
pd.set_option("display.width", 250, "display.max_columns", 30)
print(out.to_string())
