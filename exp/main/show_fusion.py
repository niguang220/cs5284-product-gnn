"""Print the val-selected configuration per (model, setting, method) from a fusion runs csv."""
import sys
import pandas as pd
s = pd.read_csv(sys.argv[1]).fillna({"llms": ""})
g = s.groupby(["model", "setting", "method", "llms", "beta", "gamma"], dropna=False)[["val_acc", "test_acc", "test_f1"]].agg(["mean", "std"])
g.columns = ["_".join(c) for c in g.columns]
g = g.reset_index()
sel = g.loc[g.groupby(["model", "setting", "method"])["val_acc_mean"].idxmax()]
print(sel[["model", "setting", "method", "llms", "beta", "gamma", "val_acc_mean", "test_acc_mean", "test_acc_std", "test_f1_mean"]].round(4).to_string(index=False))
