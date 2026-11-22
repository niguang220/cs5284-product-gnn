"""Stage 2 (text-only classifiers): numbers for the report table and the appendix figure fig_text_only.png/.pdf.

Two sources, kept apart in text_only.json:
  table   the run behind the report table (notebooks/2_text_only/text_only_analysis.ipynb, outputs of cells 29-31).
          Its results/2_text_only/text_only_runs.csv and ogb_mlp_runs.csv are not in the repo, so the values are copied here.
  repo    recomputed from what is in the repo: results/3_gnn_text/baselines_runs.csv and results/2_text_only/{tape,gpt_pred,text_transformer}_runs.csv
          (same arms, same seeds), results/3_gnn_text/tape_repro_ours_lm*_runs.csv (DeBERTa + MLP, val) and
          results/4_fusion/final_test_runs.csv (test),
          outputs/probs/garch-MLP@ft-gte-large_s{e}_normal_s{s}.npy (gte-large + MLP, 3 encoders x 3 seeds), and an
          MLP retrained here on the OGB features (3 seeds, 4 threads; standardised with train statistics).
Slices (src/train.py test_subsets): test_seen = class occurs in train; test_dedup = text does not duplicate a training
product. Macro-F1 averages over the classes present in the evaluated slice (masked_metrics). Test labels are only used
descriptively; nothing is selected here. The figure uses the table run where it has the value, the repo otherwise.
Usage: python report/figures/main/fig_text_only.py [--plot]   (--plot redraws from text_only.json, no retraining)"""
import json, os, sys
from pathlib import Path
import numpy as np, pandas as pd, torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent; REPO = HERE.parents[2]; sys.path.insert(0, str(REPO)); os.chdir(REPO)
from src.data import load_products  # noqa: E402
from src.models import MLP  # noqa: E402
from src.train import masked_metrics, set_seed, test_subsets, train  # noqa: E402

# ---------------- names (exactly the report table's rows, plus K and P) ----------------
ROWS = [("ogb", "OGB bag-of-words"), ("A", "MiniLM, frozen"), ("K", "MiniLM + 384 random dims"),
        ("G", "MiniLM + GPT explanation"), ("P", "MiniLM + GPT top-3 answers (one-hot)"),
        ("T", "Text Transformer from scratch"), ("deb_t", "DeBERTa fine-tuned on text"),
        ("deb_e", "DeBERTa fine-tuned on explanation"), ("gte", "gte-large fine-tuned (ours)"),
        ("gpt", "GPT-3.5 zero-shot answer (no training)")]
NAME = dict(ROWS)
# The table run (notebooks/2_text_only/text_only_analysis.ipynb outputs): mean (std) over seeds 0-2.
TABLE = {
    "ogb": dict(val_acc=.6771, val_acc_std=.0068, val_f1=.3833, val_f1_std=.0090, test_acc=.5109, test_acc_std=.0050,
                test_f1=.2055, test_f1_std=.0022, test_seen_f1=.2707, test_dedup_f1=.2053),
    "A": dict(val_acc=.8191, val_acc_std=.0007, val_f1=.4806, val_f1_std=.0102, test_acc=.6586, test_acc_std=.0006,
              test_f1=.2662, test_f1_std=.0006, test_seen_f1=.3475, test_dedup_f1=.2660),
    "K": dict(val_acc=.8206, val_acc_std=.0050, val_f1=.4819, val_f1_std=.0117, test_acc=.6550, test_acc_std=.0009,
              test_f1=.2670, test_f1_std=.0004, test_seen_f1=.3486, test_dedup_f1=.2669),
    "G": dict(val_acc=.8743, val_acc_std=.0026, val_f1=.5648, val_f1_std=.0100, test_acc=.7191, test_acc_std=.0013,
              test_f1=.3161, test_f1_std=.0020, test_seen_f1=.4143, test_dedup_f1=.3160),
    "P": dict(val_acc=.8757, val_acc_std=.0026, val_f1=.5491, val_f1_std=.0142, test_acc=.7230, test_acc_std=.0161,
              test_f1=.3224, test_f1_std=.0139, test_seen_f1=.4229, test_dedup_f1=.3223),
    "T": dict(val_acc=.8217, val_acc_std=.0080, val_f1=.5082, val_f1_std=.0313, test_acc=.6368, test_acc_std=.0161,
              test_f1=.2597, test_f1_std=.0160, test_seen_f1=.3423, test_dedup_f1=.2594),
    "gpt": dict(val_acc=.7831, val_f1=.5409, test_acc=.7235, test_f1=.3811, test_seen_f1=.4927),
}

PLOT_ONLY = "--plot" in sys.argv


def compute():
    g, text = load_products()
    y = g.y
    subsets = test_subsets(g, text)
    SLICES = {"val": g.val_mask, "test": subsets["test"], "test_seen": subsets["test_seen"],
              "test_dedup": subsets["test_dedup"]}

    def from_runs(df):
        """Per-run metrics from a *_runs.csv (rows per seed x eval)."""
        out = {}
        for s in SLICES:
            d = df[df["eval"] == s]
            if len(d):
                out[s] = {"acc": d.acc.tolist(), "f1": d.macro_f1.tolist()}
        return out

    def from_preds(preds):
        out = {s: {"acc": [], "f1": []} for s in SLICES}
        for p in preds:
            for s, m in SLICES.items():
                r = masked_metrics(y, torch.as_tensor(p), m); out[s]["acc"].append(r["acc"]); out[s]["f1"].append(r["macro_f1"])
        return out

    R2, R3, R4 = (REPO / "results" / s for s in ("2_text_only", "3_gnn_text", "4_fusion"))
    runs = {}
    b = pd.read_csv(R3 / "baselines_runs.csv"); b = b[b.setting == "normal"]
    runs["A"], runs["K"] = from_runs(b[b.arm == "A"]), from_runs(b[b.arm == "K"])
    t = pd.read_csv(R2 / "tape_runs.csv"); runs["G"] = from_runs(t[(t.arm == "G") & (t.setting == "normal")])
    gp = pd.read_csv(R2 / "gpt_pred_runs.csv"); gp = gp[gp.setting == "normal"]
    runs["P"], runs["gpt"] = from_runs(gp[gp.arm == "P_MLP"]), from_runs(gp[gp.arm == "GPT"])
    tt = pd.read_csv(R2 / "text_transformer_runs.csv"); runs["T"] = from_runs(tt[tt.setting == "normal"])
    # DeBERTa + MLP (3 DeBERTa seeds x 3 MLP seeds): val from run_tape_repro.py, test from run_final_eval.py.
    # Their predictions (outputs/final_eval/, outputs/tape_lm/) are not on this machine, so there is no dedup slice.
    rep = pd.concat(pd.read_csv(R3 / f"tape_repro_ours_lm{i}_runs.csv") for i in range(3))
    rep = rep[(rep.model == "MLP") & (rep.setting == "normal") & (rep["eval"] == "val")]
    ft = pd.read_csv(R4 / "final_test_runs.csv"); ft = ft[(ft.setting == "normal") & (ft.method == "model")]
    for key, feat, arm in [("deb_t", "orig", "DeBERTa + MLP"), ("deb_e", "expl", "DeBERTa(explanation) + MLP")]:
        v = rep[rep.feature == feat]; a = ft[ft.arm == arm]
        runs[key] = {"val": {"acc": v.acc.tolist(), "f1": v.macro_f1.tolist()}}
        for s, sub in [("test", "all"), ("test_seen", "seen")]:
            d = a[a.subset == sub]; runs[key][s] = {"acc": d.acc.tolist(), "f1": d.macro_f1.tolist()}
    runs["gte"] = from_preds([np.load(f"outputs/probs/garch-MLP@ft-gte-large_s{e}_normal_s{s}.npy").argmax(1)
                              for e in range(3) for s in range(3)])
    # OGB features + the same MLP (2 layers, hidden 256, dropout 0.5, lr 0.01, wd 5e-4, patience 50), seeds 0-2.
    torch.set_num_threads(4)
    x = torch.load("data/TAPE/ogbn-products_subset.pt").x.float()  # file already checked by load_products()
    mu, sd = x[g.train_mask].mean(0), x[g.train_mask].std(0).clamp(min=1e-8)
    x = (x - mu) / sd
    ogb_preds = []
    for seed in range(3):
        set_seed(seed); model = MLP(x.size(1), 256, 47); model, _ = train(model, x, g)
        model.eval()
        with torch.no_grad():
            ogb_preds.append(model(x).argmax(1))
    runs["ogb"] = from_preds(ogb_preds)

    def summ(r):
        o = {}
        for s, d in r.items():
            for k in ["acc", "f1"]:
                o[f"{s}_{k}"] = float(np.mean(d[k])); o[f"{s}_{k}_std"] = float(np.std(d[k], ddof=1)) if len(d[k]) > 1 else None
            o["runs"] = len(r["test"]["acc"])
        return o

    repo = {k: summ(runs[k]) for k, _ in ROWS}
    sizes = {s: int(m.sum()) for s, m in SLICES.items()}

    # ---------------- combined numbers (table run first) and checks ----------------
    def pick(k, field):
        tab = {"val_acc": "val_acc", "val_f1": "val_f1", "test_acc": "test_acc", "test_f1": "test_f1",
               "test_seen_f1": "test_seen_f1", "test_dedup_f1": "test_dedup_f1"}.get(field)
        if k in TABLE and tab in TABLE[k]:
            return TABLE[k][tab], "table"
        return repo[k].get(field), "repo"

    rows, diffs = [], []
    for k, name in ROWS:
        r = {"key": k, "name": name}
        for f in ["val_acc", "val_f1", "test_acc", "test_f1", "test_seen_f1", "test_dedup_f1"]:
            r[f], r[f + "_src"] = pick(k, f)
            if k in TABLE and f in TABLE[k] and repo[k].get(f) is not None and abs(TABLE[k][f] - repo[k][f]) > 0.002:
                diffs.append({"row": name, "metric": f, "table_run": TABLE[k][f], "repo": round(repo[k][f], 4)})
        for f in ["val_f1", "test_acc", "test_f1", "val_acc"]:
            s = TABLE.get(k, {}).get(f + "_std") if k in TABLE else None
            r[f + "_std"] = s if s is not None else repo[k].get(f + "_std")
        r["drop"] = r["val_acc"] - r["test_acc"]
        for f in ["test_seen_acc", "test_dedup_acc"]:
            r[f + "_repo"] = repo[k].get(f)
        r["runs"] = repo[k]["runs"]
        rows.append(r)
    df = pd.DataFrame(rows).set_index("key")

    def order(col, src):
        return [NAME[k] for k in src[col].sort_values(ascending=False).index]

    def spearman(a, b):
        return float(pd.Series(a).rank().corr(pd.Series(b).rank()))

    R2 = pd.DataFrame(repo).T  # repo numbers only, so each comparison uses one source
    checks = {"seen_f1_gain": {NAME[k]: round(df.loc[k, "test_seen_f1"] - df.loc[k, "test_f1"], 4) for k in df.index}}
    checks["seen_f1_gain_range"] = [min(checks["seen_f1_gain"].values()), max(checks["seen_f1_gain"].values())]
    checks["seen_f1_gain_range_trained_only"] = [min(v for n, v in checks["seen_f1_gain"].items() if "GPT-3.5" not in n),
                                                 max(v for n, v in checks["seen_f1_gain"].items() if "GPT-3.5" not in n)]
    for slice_ in ["test_seen", "test_dedup"]:
        for m in ["acc", "f1"]:
            have = R2[f"{slice_}_{m}"].notna()
            sub = R2[have]
            checks[f"{slice_}_{m}"] = {
                "rows_without_slice": [NAME[k] for k in R2.index[~have]],
                "order_full_test": order(f"test_{m}", sub), "order_slice": order(f"{slice_}_{m}", sub),
                "same_order": order(f"test_{m}", sub) == order(f"{slice_}_{m}", sub),
                "spearman": spearman(sub[f"test_{m}"], sub[f"{slice_}_{m}"]),
                "max_abs_change": float((sub[f"{slice_}_{m}"] - sub[f"test_{m}"]).abs().max()) if slice_ == "test_dedup" else None}
    out = {"slice_sizes": sizes, "rows": df.reset_index().to_dict("records"), "repo": repo, "table_run": TABLE,
           "table_vs_repo_over_0.002": diffs, "checks": checks,
           "notes": {"gte": "garch-MLP: same 2-layer MLP (hidden 256, wd 5e-4) but lr/dropout chosen on val (0.01 / 0.2)",
                     "deberta": "3 DeBERTa seeds x 3 MLP seeds; dedup slice unavailable (predictions not on this machine)",
                     "ogb": "repo column retrained here; table run from the notebook"}}
    (HERE / "text_only.json").write_text(json.dumps(out, indent=1, default=float) + "\n")
    pd.set_option("display.width", 250)
    print(df[["val_acc", "test_acc", "test_f1", "drop", "val_f1", "test_seen_f1", "test_dedup_f1", "test_dedup_acc_repo"]].round(4))
    print(pd.DataFrame(repo).T[["val_acc", "test_acc", "test_f1", "val_f1", "test_seen_f1", "test_dedup_acc", "test_dedup_f1"]].round(4))
    print("diffs:", diffs); print(json.dumps(checks, indent=1))
    return out


out = json.loads((HERE / "text_only.json").read_text()) if PLOT_ONLY else compute()
df = pd.DataFrame(out["rows"]).set_index("key")
# The same ranking checks on the table-consistent numbers (macro-F1 only: the table run has no slice accuracy).
tc = {}
for slice_ in ["test_seen", "test_dedup"]:
    sub = df[df[f"{slice_}_f1"].notna()]
    full = [NAME[k] for k in sub.test_f1.sort_values(ascending=False).index]
    sl = [NAME[k] for k in sub[f"{slice_}_f1"].sort_values(ascending=False).index]
    tc[f"{slice_}_f1"] = {"rows_without_slice": [NAME[k] for k in df.index if k not in sub.index],
                          "order_full_test": full, "order_slice": sl, "same_order": full == sl,
                          "spearman": float(sub.test_f1.rank().corr(sub[f"{slice_}_f1"].rank()))}
out["checks_table_consistent"] = tc
(HERE / "text_only.json").write_text(json.dumps(out, indent=1, default=float) + "\n")
print(json.dumps(tc, indent=1))

# ---------------- figure (style of figs_final.py) ----------------
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
VAL, TEST = "#6250d6", "#2a78d6"
BASELINE, BASE, GPT = "#8a8984", "#1baf7a", "#eb6834"
F_TICK, F_LAB, F_LEG, F_ANN, F_TIT = 6.5, 7.0, 6.2, 6.0, 7.6
plt.rcParams.update({"mathtext.fontset": "dejavusans", "font.family": "Liberation Sans", "font.size": F_LAB,
                     "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "xtick.labelsize": F_TICK, "ytick.labelsize": F_TICK, "axes.labelsize": F_LAB,
                     "legend.fontsize": F_LEG, "legend.frameon": False, "legend.handlelength": 1.4,
                     "legend.handletextpad": 0.4, "legend.borderaxespad": 0.3, "legend.labelspacing": 0.3,
                     "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titlesize": F_TIT, "axes.titleweight": "bold", "axes.titlelocation": "left",
                     "axes.titlepad": 4, "xtick.major.size": 2.5, "ytick.major.size": 2.5, "xtick.major.pad": 2,
                     "ytick.major.pad": 2, "axes.labelpad": 2, "figure.dpi": 300, "savefig.dpi": 300,
                     "axes.linewidth": 0.7, "hatch.linewidth": 0.6})

fig, (a, b) = plt.subplots(1, 2, figsize=(6.3, 3.1), gridspec_kw={"width_ratios": [1.0, 1.0], "wspace": 0.14})
# (a) macro-F1, horizontal grouped bars so the full names fit
keys = [k for k, _ in ROWS]
yy = np.arange(len(keys))[::-1].astype(float)
h = 0.26
a.barh(yy + h, df.loc[keys, "val_f1"], h, xerr=df.loc[keys, "val_f1_std"].fillna(0), color=VAL,
       error_kw=dict(lw=0.6, capsize=1.2, ecolor=INK), label="validation")
a.barh(yy, df.loc[keys, "test_f1"], h, xerr=df.loc[keys, "test_f1_std"].fillna(0), color=TEST,
       error_kw=dict(lw=0.6, capsize=1.2, ecolor=INK), label="test")
a.barh(yy - h, df.loc[keys, "test_seen_f1"], h, color="white", edgecolor=TEST, linewidth=0.6, hatch="//////",
       label="test, categories seen in training")
a.set_yticks(yy, [NAME[k] for k in keys]); a.tick_params(axis="y", length=0)
a.set_xlim(0, 0.75); a.set_xticks([0, 0.2, 0.4, 0.6]); a.set_xlabel("macro-F1")
a.set_ylim(-0.6, len(keys) - 0.4)
a.grid(axis="x", color=GRID, linewidth=0.6); a.set_axisbelow(True)
a.legend(loc="upper left", bbox_to_anchor=(-0.02, -0.13), ncol=3, columnspacing=1.0)
a.set_title("(a) Macro-F1", x=-0.62)

# (b) accuracy, validation -> test, labels spread so they do not overlap
col = {k: BASELINE for k in keys}; col.update({"gte": BASE, "gpt": GPT})
for k in keys:
    v, t = df.loc[k, "val_acc"], df.loc[k, "test_acc"]
    b.plot([0, 1], [v, t], color=col[k], lw=1.5 if k in ("gte", "gpt") else 0.9, ls=(0, (3, 1.5)) if k == "gpt" else "-",
           marker="o", ms=2.4, zorder=3 if k in ("gte", "gpt") else 2, solid_capstyle="round")
ts = df.loc[keys, "test_acc"].sort_values()
pos, gap = {}, 0.0185
for k in ts.index:  # bottom-up placement with a minimum gap
    pos[k] = max(ts[k], max(pos.values()) + gap) if pos else ts[k]
shift = max(0.0, max(pos.values()) - 0.805)  # keep the top label inside the axes
for k in pos:
    pos[k] -= shift * (pos[k] - min(pos.values())) / max(1e-9, max(pos.values()) - min(pos.values()))
for k in keys:
    t, p = df.loc[k, "test_acc"], pos[k]
    b.plot([1.02, 1.1], [t, p], color=GRID if col[k] == BASELINE else col[k], lw=0.5, zorder=1)
    b.text(1.12, p, f"{NAME[k]}  $-${100 * df.loc[k, 'drop']:.1f}", va="center", fontsize=F_ANN,
           color=INK if col[k] == BASELINE else col[k])
b.set_xlim(-0.12, 2.62); b.set_xticks([0, 1], ["validation\n(best-sellers)", "test\n(long tail)"])
b.tick_params(axis="x", length=0)
b.set_ylim(0.49, 0.94); b.set_ylabel("accuracy"); b.spines["bottom"].set_visible(False)
for v in np.arange(0.5, 0.95, 0.05):  # grid only under the lines, not under the labels
    b.plot([-0.12, 1.06], [v, v], color=GRID, lw=0.6, zorder=0)
for s in [0, 1]:
    b.axvline(s, color=GRID, lw=0.6, zorder=0)
b.text(1.12, 0.925, "drop, pp", fontsize=F_ANN, color=MUTED, va="center")
b.set_title("(b) Accuracy, validation to test")
fig.savefig(HERE / "fig_text_only.png", bbox_inches="tight", pad_inches=0.02, facecolor="white")
fig.savefig(HERE / "fig_text_only.pdf", bbox_inches="tight", pad_inches=0.02, facecolor="white")
plt.close(fig)
