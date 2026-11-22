"""Val-selected results per variant family: for each (encoder seed, setting, variant) pick the (params, LLMs, β, γ)
with the best mean val acc over the 3 GNN seeds among configs that use BP (β>0 or γ>0), as run_fusion.py does; then
mean ± std over encoder seeds. Also a per-params table (selection over LLMs/β/γ only) for sensitivity.
Usage: python summarize.py [runs_dir]  -> exp/g-localprior/summary[_dir].csv, summary[_dir]_params.csv
"""
import glob
import sys
from pathlib import Path

import pandas as pd

D = Path(__file__).parent
RUNS = sys.argv[1] if len(sys.argv) > 1 else "runs"  # runs, runs_ppr, ...
files = sorted(glob.glob(str(D / RUNS / "*.csv")))
SUF = "" if RUNS == "runs" else "_" + RUNS.replace("runs_", "")
df = pd.concat([pd.read_csv(f) for f in files])
df = df[(df.beta > 0) | (df.gamma > 0)]
df["params"] = df.params.astype(str).str.replace(r",w=.*", "", regex=True)  # fitted weights differ per seed
metrics = [c for c in df.columns if c in ("val_acc", "test_acc", "test_f1") or c.startswith(("hop", "rec_"))]
cfg = ["params", "llms", "beta", "gamma"]
g = df.groupby(["enc", "setting", "variant"] + cfg)[metrics].mean().reset_index()
sel = g.loc[g.groupby(["enc", "setting", "variant"]).val_acc.idxmax()]
sel.to_csv(D / f"summary{SUF}_selected_per_enc.csv", index=False)
agg = sel.groupby(["setting", "variant"])[metrics].agg(["mean", "std"])
out = pd.DataFrame(index=agg.index)
for m in metrics:
    out[m] = agg[(m, "mean")].round(4)
out["test_acc_std"] = agg[("test_acc", "std")].round(4)
out["n_enc"] = sel.groupby(["setting", "variant"]).size()
out.to_csv(D / f"summary{SUF}.csv")
with pd.option_context("display.width", 250, "display.max_rows", 500, "display.max_columns", 30):
    print(out[["n_enc", "val_acc", "test_acc", "test_acc_std", "test_f1", "hop1", "hop2", "hop3", "hop4+",
               "rec_Electronics", "rec_Office", "rec_Tools", "rec_Automotive"]].to_string())
    print(sel[["enc", "setting", "variant", "params", "llms", "beta", "gamma"]].to_string(index=False))
# sensitivity: per params, selection over LLMs / β / γ, mean over encoder seeds
p = g.loc[g.groupby(["enc", "setting", "variant", "params"]).val_acc.idxmax()]
p = p.groupby(["setting", "variant", "params"])[["val_acc", "test_acc", "test_f1", "hop3", "hop4+"]].mean().round(4)
p.to_csv(D / f"summary{SUF}_params.csv")
