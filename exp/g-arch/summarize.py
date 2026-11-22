"""Summarise g-arch runs: base metrics (exp/g-arch/results.jsonl) and post-fusion metrics (exp/g-arch/fusion/*.csv).

Post-fusion row = val-selected fuse+BP or fuse+BP+kNN (whichever has the higher mean val acc), as in the main
pipeline; selected separately for normal and cold start. Writes exp/g-arch/summary_{base,fusion}.csv.
"""
import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
emb = sys.argv[1] if len(sys.argv) > 1 else "ft-gte-large_s0"
rows = [json.loads(l) for l in open(HERE / "results.jsonl")]
df = pd.DataFrame(rows)
df = df[df.emb == emb].drop_duplicates(["variant", "seed"], keep="last")
cols = ["val_acc", "test_acc", "test_f1", "cold_test_acc", "cold_test_f1", "hop1", "hop2", "hop3", "hop4+",
        "cold_hop4+", "rec_Electronics", "rec_Office", "rec_Tools", "rec_Automotive"]
m = df.groupby("variant")[cols].mean()
s = df.groupby("variant")[["val_acc", "test_acc", "cold_test_acc"]].std()
base = m.copy()
for c in s:
    base[c + "_std"] = s[c]
base["cfg"] = df.groupby("variant")["cfg"].first().astype(str)
base = base.sort_values("val_acc", ascending=False)
base.round(4).to_csv(HERE / f"summary_base_{emb}.csv")

fus = []
for v in base.index:
    f = HERE / "fusion" / f"garch-{v}@{emb}_runs.csv"
    if not f.exists():
        continue
    r = pd.read_csv(f).fillna({"llms": ""})
    key = ["setting", "method", "llms", "beta", "gamma"]
    g = r.drop(columns=["model", "seed"]).groupby(key, dropna=False).agg(["mean", "std"])
    g.columns = ["_".join(c) for c in g.columns]
    g = g.reset_index()
    out = {"variant": v}
    for st in ["normal", "cold"]:
        cand = g[(g.setting == st) & g.method.isin(["fuse+BP", "fuse+BP+kNN"])]
        b = cand.loc[cand.val_acc_mean.idxmax()]
        p = "" if st == "normal" else "cold_"
        out.update({p + "val": b.val_acc_mean, p + "test": b.test_acc_mean, p + "test_std": b.test_acc_std,
                    p + "f1": b.test_f1_mean, p + "sel": f"{b.method} {b.llms} b={b.beta} g={b.gamma}"})
        if st == "normal":
            out.update({k: b[k + "_mean"] for k in ["hop1", "hop2", "hop3", "hop4+", "rec_Electronics",
                                                     "rec_Office", "rec_Tools", "rec_Automotive"]})
    fus.append(out)
fus = pd.DataFrame(fus).set_index("variant") if fus else pd.DataFrame()
fus.round(4).to_csv(HERE / f"summary_fusion_{emb}.csv")
with pd.option_context("display.width", 250, "display.max_columns", 40, "display.max_colwidth", 60):
    print(base[["val_acc", "val_acc_std", "test_acc", "test_acc_std", "test_f1", "cold_test_acc", "cold_test_f1",
                "hop1", "hop2", "hop3", "hop4+", "cfg"]].round(4))
    if len(fus):
        print(fus.round(4))
