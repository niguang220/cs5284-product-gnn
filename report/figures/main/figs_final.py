"""Report figures, drawn at print size (inserted at 12-16 cm, so 1 pt in the figure is about 1 pt on paper).
One style for all: fixed font sizes, and colours that mean the same thing everywhere:
  splits:  train = TRAIN (grey), validation = VAL (violet), test = TEST (blue)
  models:  MiniLM baselines = BASELINE (light grey), NAGphormer base = BASE (aqua), GPT-3.5 = GPT (orange),
           final pipeline = FINAL (blue)
Numbers come from results/, exp/ and the analysis scripts in this folder."""
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Patch
from matplotlib.lines import Line2D

HERE = Path(__file__).parent
OUT = HERE  # figures are written next to this script
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
TRAIN, VAL, TEST = "#8a8984", "#6250d6", "#2a78d6"
BASELINE, BASE, GPT, FINAL = "#b3b2ac", "#1baf7a", "#eb6834", "#2a78d6"
LIGHT = "#d9d8d3"
F_TICK, F_LAB, F_LEG, F_ANN, F_TIT = 6.5, 7.0, 6.2, 6.2, 7.6
LW, MS = 1.6, 3.4
plt.rcParams.update({"mathtext.fontset": "dejavusans", "font.family": "Liberation Sans", "font.size": F_LAB,
                     "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "xtick.labelsize": F_TICK, "ytick.labelsize": F_TICK, "axes.labelsize": F_LAB,
                     "legend.fontsize": F_LEG, "legend.frameon": False, "legend.handlelength": 1.4,
                     "legend.handletextpad": 0.4, "legend.borderaxespad": 0.3, "legend.labelspacing": 0.3,
                     "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titlesize": F_TIT, "axes.titleweight": "bold", "axes.titlelocation": "left",
                     "axes.titlepad": 4, "xtick.major.size": 2.5, "ytick.major.size": 2.5, "xtick.major.pad": 2,
                     "ytick.major.pad": 2, "axes.labelpad": 2, "figure.dpi": 300, "savefig.dpi": 300,
                     "axes.linewidth": 0.7})

def grid(ax, axis="y"):
    ax.grid(axis=axis, color=GRID, linewidth=0.6); ax.set_axisbelow(True)

def ann(ax, x, y, s, **kw):
    ax.text(x, y, s, fontsize=F_ANN, color=kw.pop("color", MUTED), **kw)

def save(fig, name):
    fig.savefig(OUT / name, bbox_inches="tight", pad_inches=0.02, facecolor="white"); plt.close(fig)

# ================= Figure 1: sampling layers and class shift =================
sh = pd.read_csv(HERE / "class_share.csv")
fig, (a, b) = plt.subplots(1, 2, figsize=(5.2, 1.75), gridspec_kw={"width_ratios": [0.85, 1.3], "wspace": 0.75})
layers = ["seeds", "hop 1", "hop 2", "hop 3"]; n = [128, 1590, 11814, 40493]; trs = [6.2, 30.6, 26.8, 27.3]
x = np.arange(4)
a.bar(x, trs, 0.62, color=TRAIN)
a.axhline(8, color=INK, lw=0.8, ls=(0, (3, 2)))
ann(a, 0, 9.3, "OGB\n8%", ha="center", va="bottom", color=INK, linespacing=1.0)
ann(a, 0, 3.1, "6%", ha="center", va="center", color="white")
for i in range(1, 4):
    ann(a, i, trs[i] + 0.8, f"{trs[i]:.0f}%", ha="center")
a.set_xticks(x, [f"{l}\n{n[i] / 1000:.1f}k" for i, l in enumerate(layers)])
a.set_ylabel("train share (%)"); a.set_ylim(0, 36); grid(a); a.set_title("(a) Sampling layers")
keep = ["Books", "Cell Phones & Accessories", "Health & Personal Care", "Movies & TV", "MISSING", "Baby", "Automotive",
        "Office Products", "Sports & Outdoors", "Tools & Home Improvement", "Electronics"]
s = sh[sh.name.isin(keep)].drop_duplicates("name", keep="first")
s = s[~((s.name == "MISSING") & (s.ntr == 0))].sort_values("d")
yy = np.arange(len(s))
for i, (_, r) in enumerate(s.iterrows()):
    b.plot([100 * r.train, 100 * r.test], [i, i], color=LIGHT, lw=1.6, zorder=1, solid_capstyle="round")
b.scatter(100 * s.train, yy, s=16, color=TRAIN, zorder=2, label="train (best-sellers)")
b.scatter(100 * s.test, yy, s=16, color=TEST, zorder=2, label="test (long tail)")
short = {"Tools & Home Improvement": "Tools & Home Impr.", "Health & Personal Care": "Health & Pers. Care",
         "Cell Phones & Accessories": "Cell Phones & Acc."}
b.set_yticks(yy, [short.get(m, m) for m in s.name]); b.set_xlim(-1, 29)
b.set_xlabel("share of split (%)"); grid(b, "x"); b.legend(loc="center right", bbox_to_anchor=(1.0, 0.42))
b.set_title("(b) Class share, train vs test")
save(fig, "fig1_data.png")

# ================= Figure 2: what the class mix does to accuracy (C1, C2) =================
fig, (a, b, c) = plt.subplots(1, 3, figsize=(5.6, 1.6), gridspec_kw={"wspace": 0.55, "width_ratios": [1.4, 0.82, 1.05]})
rows = [("MiniLM + MLP", .8227, .8007, .6611, BASELINE), ("MiniLM + GCN", .8707, .8550, .7120, BASELINE),
        ("MiniLM + GT", .9039, .8811, .7648, BASELINE), ("gte + NAGphormer", .9288, .9101, .8138, BASE),
        ("GPT-3.5", .7824, .7871, .7230, GPT)]
yy = np.arange(len(rows))[::-1]
for yv, (nm, v, m, t, col) in zip(yy, rows):
    w1, w2 = 100 * (v - m), 100 * (m - t)
    a.barh(yv, w1, color=LIGHT, height=0.62, edgecolor="white", lw=0.5)
    a.barh(yv, w2, left=max(w1, 0), color=col, height=0.62, edgecolor="white", lw=0.5)
    if w1 > 0:
        ann(a, w1 + w2 + 0.4, yv, f"{w2 / (w1 + w2) * 100:.0f}%", va="center")
a.set_yticks(yy, [r[0] for r in rows]); a.set_xlim(-0.8, 19.5); a.axvline(0, color=MUTED, lw=0.7)
a.set_xlabel("val − test accuracy (points)"); grid(a, "x")
a.legend(handles=[Patch(color=LIGHT, label="within-class"), Patch(color=BASELINE, label="class mix")],
         loc="lower right")
a.set_title("(a) Val → test drop")
pred = [68.5, 72.4, 78.9, 83.2]; act = [66.1, 71.2, 76.5, 81.4]
b.plot([62, 86], [62, 86], color=MUTED, lw=0.7, ls=(0, (3, 2)))
b.scatter(pred, act, s=16, color=[BASELINE, BASELINE, BASELINE, BASE], edgecolor=MUTED, lw=0.3, zorder=3)
for p, t_, nm in zip(pred, act, ["MLP", "GCN", "GT", "NAG"]):
    ann(b, p + 0.9, t_ - 0.3, nm, va="top")
b.set_xlim(62, 86); b.set_ylim(62, 86); b.set_xticks([65, 75, 85]); b.set_yticks([65, 75, 85]); grid(b, "both")
b.set_xlabel("predicted test acc. (%)"); b.set_ylabel("actual test acc. (%)")
b.set_title("(b) Predicted vs actual")
cls = ["Elec.", "Off.", "Tools", "Auto"]; nv, nt = [25, 8, 6, 1], [3088, 1518, 1794, 968]
xx = np.arange(4)
bv = c.bar(xx - 0.2, nv, 0.38, color=VAL, label="validation"); c.bar(xx + 0.2, nt, 0.38, color=TEST, label="test")
c.set_yscale("log"); c.set_ylim(0.7, 3e4); c.bar_label(bv, fontsize=F_ANN, color=MUTED, padding=1)
c.set_xticks(xx, cls, fontsize=6.2); c.set_ylabel("products (log scale)")
c.legend(loc="upper right", ncol=2, columnspacing=0.6, handlelength=1.0, bbox_to_anchor=(1.05, 1.02))
c.set_title("(c) Shifted-class support")
save(fig, "fig_eda_shift.png")

# ================= Figure 3: the shift is structured on the graph (C3, O1) =================
fig, (a, b, c) = plt.subplots(1, 3, figsize=(5.6, 1.6), gridspec_kw={"wspace": 0.5, "width_ratios": [1.1, 1, 0.8]})
x = np.arange(4)
tv, rare = [0.139, 0.232, 0.465, 0.782], [0.091, 0.163, 0.391, 0.714]
a.plot(x, tv, color=INK, lw=LW, marker="o", ms=MS, label="TV to train class mix")
a.plot(x, rare, color=TEST, lw=LW, marker="o", ms=MS, label="share of 4 train-rare classes")
ann(a, 3.15, tv[3] + 0.02, f"{tv[3]:.2f}", va="center", color=INK); ann(a, 3.15, rare[3] - 0.05, f"{rare[3]:.2f}", va="center", color=TEST)
a.set_xticks(x, ["1\n41%", "2\n40%", "3\n12%", "4+\n6%"]); a.set_xlim(-0.25, 3.7); a.set_ylim(0, 1.12); grid(a)
a.set_xlabel("hops to train (share of test)"); a.legend(loc="upper left"); a.set_title("(a) Shift by distance")
for nm, v, col, ls in [("MiniLM + MLP", [75.1, 67.4, 51.3, 23.9], BASELINE, (0, (3, 1.5))),
                       ("MiniLM + GCN", [79.9, 73.2, 58.9, 30.4], BASELINE, "-"),
                       ("GPT-3.5", [74.8, 71.6, 68.1, 69.8], GPT, "-")]:
    b.plot(x, v, color=col, ls=ls, lw=LW, marker="o", ms=MS, label=nm)
ann(b, 3.15, 69.8, "70", va="center"); ann(b, 3.15, 31.5, "30", va="center"); ann(b, 3.15, 22.5, "24", va="center")
b.set_xticks(x, ["1", "2", "3", "4+"]); b.set_xlim(-0.25, 3.6); b.set_ylim(15, 100); grid(b)
b.set_xlabel("hops to train"); b.set_ylabel("test accuracy (%)"); b.legend(loc="upper right")
b.set_title("(b) Accuracy by distance")
xx = np.arange(3)
for i, (nm, v, col) in enumerate([("train", [51, 74, 91], TRAIN), ("test", [43, 63, 83], TEST)]):
    bars = c.bar(xx + (i - 0.5) * 0.38, v, 0.35, color=col, label=nm); c.bar_label(bars, fontsize=F_ANN, color=MUTED, padding=1)
c.set_xticks(xx, ["none", "some", "all"]); c.set_ylim(0, 112); c.set_yticks(range(0, 101, 25)); grid(c)
c.set_ylabel("GPT-3.5 accuracy (%)"); c.set_xlabel("neighbours with\nthe same answer")
c.legend(loc="upper left", ncol=2, columnspacing=0.6, handlelength=1.0); c.set_title("(c) LLM reliability")
save(fig, "fig_eda_hop.png")

# ================= Figure 4: what cannot be learned (C5) =================
fig, (a, b, c) = plt.subplots(1, 3, figsize=(5.6, 1.5), gridspec_kw={"wspace": 0.55, "width_ratios": [1.05, 0.7, 1.0]})
labs = ["≥100 train products", "1–99 train products", "MISSING (metadata)", "no train product"]
vals = [87.9, 7.8, 2.0, 2.3]; cols = [TEST, "#86b6ef", TRAIN, INK]
yy = np.arange(4)[::-1]
a.barh(yy, vals, 0.6, color=cols)
for yv, v in zip(yy, vals):
    ann(a, v + 1.5, yv, f"{v}%", va="center")
a.set_yticks(yy, labs); a.set_xlim(0, 112); a.set_xticks([0, 50, 100]); grid(a, "x")
a.set_xlabel("share of test products (%)"); a.set_title("(a) Test by label status")
bars = b.bar(range(3), [0.8, 3.1, 24.1], 0.6, color=[TRAIN, VAL, TEST])
b.bar_label(bars, fmt="%.1f", fontsize=F_ANN, color=MUTED, padding=1)
b.set_xticks(range(3), ["train", "val", "test"]); b.set_ylim(0, 29); grid(b)
b.set_ylabel("MISSING among\ntitle-less (%)"); b.set_title("(b) Group shift")
cp = [90.8, 13.7, 99.5, 94.0]; el = [8.3, 81.6, 0.2, 3.5]
xx = np.array([0, 1, 2.3, 3.3])
c.bar(xx, cp, 0.7, color=BASE, label="Cell Phones")
c.bar(xx, el, 0.7, bottom=cp, color="#e87ba4", label="Electronics")
c.set_xticks(xx, ["train", "test", "train", "test"])
for xc, lab in [(0.5, "tablet acc."), (2.8, "phone acc.")]:
    c.text(xc, -34, lab, ha="center", fontsize=F_TICK, color=MUTED)
c.set_ylim(0, 128); c.set_yticks(range(0, 101, 25)); grid(c); c.set_ylabel("label share (%)")
c.legend(loc="upper center", ncol=2, columnspacing=0.6, handlelength=1.0, bbox_to_anchor=(0.5, 1.03))
c.set_title("(c) A moving boundary")
save(fig, "fig_eda_labels.png")

# ================= Figure 5: method =================
plt.rcParams["mathtext.fontset"] = "cm"  # LaTeX-style maths for the equations of this figure
fig, ax = plt.subplots(figsize=(6.3, 2.1)); ax.set_xlim(-0.5, 100.5); ax.set_ylim(-2.5, 36); ax.axis("off")
def box(x0, title, body, col, w=17.9):
    ax.add_patch(FancyBboxPatch((x0, 15.5), w, 18, boxstyle="round,pad=0.25,rounding_size=1.0", fc=col + "1f", ec=col, lw=0.9))
    ax.text(x0 + w / 2, 31.6, title, ha="center", va="top", fontsize=7.2, fontweight="bold")
    ax.text(x0 + w / 2, 27.4, body, ha="center", va="top", fontsize=6.2, color=MUTED, linespacing=1.3)
box(0.5, "1  Text encoder", "gte-large fine-tuned\non 14.7k train labels\n→ [CLS] features $X$", BASE)
box(20.75, "2  NAGphormer", "hop tokens\n$X,\\ \\hat{A}X,\\ \\ldots,\\ \\hat{A}^{6}X$\n$\\rightarrow\\ p(y \\mid x)$, $T$ on val", BASE)
box(41.0, "3  LLM fusion", "GPT-3.5 answer $g$\n$\\times$ confusion $P(g \\mid y)$\nweight 1 (Bayes)", GPT)
box(61.25, "4  PPR local prior", "PPR-diffused beliefs\n$+\\lambda\\,\\log\\,\\hat{\\pi}_i(y)\\,/\\,\\pi_{\\mathrm{tr}}(y)$\n$\\lambda = 1$ (Saerens)", FINAL)
box(81.5, "5  One-round BP", "messages via $H$ from\nco-purchase, kNN\nand sibling edges", FINAL)
for x0 in [18.65, 38.9, 59.15, 79.4]:
    ax.annotate("", (x0 + 1.8, 24.5), (x0, 24.5), arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=0.8, shrinkA=0, shrinkB=0, mutation_scale=7))
ax.plot([0.5, 38.65], [12.6, 12.6], color=BASE, lw=1.6); ax.text(19.6, 10.6, "trained once (GPU)", ha="center", va="top", fontsize=6.4, color=BASE)
ax.plot([41.0, 99.4], [12.6, 12.6], color=FINAL, lw=1.6)
ax.text(70.2, 10.6, "post-hoc on CPU in seconds (only $T$, PPR teleport and $\\beta$ tuned on val)", ha="center", va="top",
        fontsize=6.4, color=FINAL)
ax.text(50, 1.2, r"$b_i(y)\;\propto\;p(y \mid x_i)\,P(g_i \mid y)\,\left[\hat{\pi}_i(y)\,/\,\pi_{\mathrm{tr}}(y)\right]^{\lambda}\;\prod_{c}\,\prod_{j \in N_c(i)} m_j(y)^{\beta_c}, \qquad m_j(y) = \sum_{y^{\prime}}\, H_c[y, y^{\prime}]\; e_j(y^{\prime})$", ha="center", va="center", fontsize=8.5)
save(fig, "fig3_method.png")
plt.rcParams["mathtext.fontset"] = "dejavusans"

# ================= Figure 6: validation cannot see the correction (C2) =================
fig, axs = plt.subplots(1, 3, figsize=(5.6, 1.55), gridspec_kw={"width_ratios": [1.2, 1, 1], "wspace": 0.35})
st = ["base", "+ LLM\nfusion", "+ PPR\nprior", "+ BP\n(final)"]
val = np.array([0.9288, 0.9188, 0.9166, 0.9184]); tst = np.array([0.8138, 0.8358, 0.8415, 0.8420])
def deltas(ax, xs, v, t, xt, title, mark=None):
    ax.axhline(0, color=MUTED, lw=0.7)
    if mark is not None:
        ax.axvline(mark, color=MUTED, lw=0.7, ls=(0, (1, 1.5)))
    ax.plot(xs, 100 * (t - t[0]), color=TEST, lw=LW, marker="o", ms=MS, label="test")
    ax.plot(xs, 100 * (v - v[0]), color=VAL, lw=LW, marker="o", ms=MS, label="validation")
    ax.set_xticks(xs, xt); grid(ax); ax.set_title(title)
deltas(axs[0], np.arange(4), val, tst, st, "(a) Pipeline steps")
axs[0].set_ylabel("Δ accuracy (points)"); axs[0].set_ylim(-2.1, 3.6); axs[0].set_xlim(-0.3, 3.4)
for i in range(1, 4):
    ann(axs[0], i, 100 * (tst[i] - tst[0]) + 0.3, f"+{100 * (tst[i] - tst[0]):.1f}", ha="center", color=TEST)
    ann(axs[0], i, 100 * (val[i] - val[0]) - 0.75, f"{100 * (val[i] - val[0]):.1f}", ha="center", color=VAL)
axs[0].legend(loc="center right", bbox_to_anchor=(1.0, 0.58))
# Typed in from earlier exploration runs; these values are not saved as a results file in this repo.
lam = np.array([0, 0.25, 0.5, 1, 1.5, 2])
deltas(axs[1], lam, np.array([.9226, .9222, .9224, .9184, .9154, .9122]), np.array([.8382, .8413, .8427, .8433, .8423, .8407]),
       ["0", "", "0.5", "1", "1.5", "2"], "(b) Prior strength λ", mark=1)
axs[1].set_xlabel("λ (with w = 1)")
w = np.array([0.5, 0.75, 1, 1.25])
deltas(axs[2], w, np.array([.9302, .9269, .9232, .9194]), np.array([.8351, .8387, .8409, .8355]),
       ["0.5", "0.75", "1", "1.25"], "(c) LLM weight w", mark=1)
axs[2].set_xlabel("w (λ chosen on val)")
for ax in axs[1:]:
    ax.set_ylim(-1.25, 0.8)
save(fig, "fig4_valshift.png")

# ================= Figure 7: where the gain lands (C1, C3) =================
fig, (a, b, c3) = plt.subplots(1, 3, figsize=(6.6, 1.65), gridspec_kw={"width_ratios": [1, 1.2, 1.1], "wspace": 0.42})
series = [("MiniLM + GCN", [79.9, 73.2, 58.9, 30.4], BASELINE), ("GPT-3.5", [74.8, 71.6, 68.1, 69.8], GPT),
          ("NAGphormer (base)", [86.1, 81.6, 74.1, 63.7], BASE), ("final pipeline", [87.0, 83.5, 80.1, 78.2], FINAL)]
for nm, v, col in series:
    a.plot(range(4), v, color=col, lw=LW, marker="o", ms=MS, label=nm)
for v, dy in [(78.2, 1.5), (69.8, 0.5), (63.7, -1.5), (30.4, 0)]:
    ann(a, 3.15, v + dy, f"{v:.0f}", va="center")
a.set_xticks(range(4), ["1", "2", "3", "4+"]); a.set_xlim(-0.25, 3.6); a.set_ylim(20, 95); grid(a)
a.set_xlabel("hops to train"); a.set_ylabel("test accuracy (%)"); a.legend(loc="lower left")
a.set_title("(a) Accuracy by distance")
cls = ["Elec.", "Office", "Tools", "Auto", "Books"]
rec = [("GPT-3.5", [48.2, 86.4, 45.0, 70.1, 93.0], GPT), ("NAGphormer (base)", [64.9, 68.9, 66.9, 29.5, 99.6], BASE),
       ("final pipeline", [72.3, 88.2, 80.5, 71.9, 99.0], FINAL)]
x = np.arange(5); wd = 0.26
for i, (nm, v, col) in enumerate(rec):
    bars = b.bar(x + (i - 1) * wd, v, wd - 0.03, color=col, label=nm)
    if nm == "final pipeline":
        b.bar_label(bars, fmt="%.0f", fontsize=F_ANN, color=MUTED, padding=1)
b.set_xticks(x, cls); b.set_ylabel("test recall (%)"); b.set_ylim(0, 112); b.set_yticks(range(0, 101, 25)); grid(b)
b.set_title("(b) Recall on train-rare classes")
cs3 = pd.read_csv(HERE / "class_shift.csv"); cs3 = cs3[cs3.c != 28]
sz = 6 + 400 * cs3.test_share
c3.axhline(0, color=MUTED, lw=0.6); c3.axvline(0, color=GRID, lw=0.6)
c3.scatter(cs3.log2ratio, cs3.d_recall, s=sz, color=FINAL, alpha=0.9, lw=0, label="recall", zorder=2)
c3.scatter(cs3.log2ratio, cs3.d_prec, s=sz, facecolor="white", edgecolor=FINAL, lw=0.7, label="precision", zorder=2)
for nm, x0, y0, ha in [("Automotive", 2.95, 42.3, "right"), ("Office", 2.6, 24.5, "right"), ("Tools", 2.3, 13.6, "right"),
                       ("Home & Kitchen\n(precision)", -0.29, 24, "center")]:
    ann(c3, x0, y0, nm, va="center", ha=ha, linespacing=0.95)
c3.set_xlabel(r"class prior ratio $\log_2\,\pi_{\mathrm{test}}/\pi_{\mathrm{train}}$"); c3.set_ylabel("change vs base (points)")
c3.set_ylim(-10, 48); grid(c3); c3.legend(loc="upper left", markerscale=0.6)
c3.set_title("(c) Per-class change")
save(fig, "fig5_where.png")

# ================= Figure 8: backbones converge; accuracy vs coverage =================
fig, (a, b, c9) = plt.subplots(1, 3, figsize=(6.6, 1.6), gridspec_kw={"wspace": 0.85, "width_ratios": [1.1, 1, 0.85]})
bk = [("GAT", 78.78, 82.49), ("GCN", 79.15, 82.49), ("MLP", 79.40, 83.02), ("SAGE", 80.84, 83.38), ("APPNP", 81.03, 83.13),
      ("GT", 81.13, 83.53), ("NAGphormer", 81.69, 83.74)]
yy = np.arange(len(bk))
for yv, (nm, b0, b1) in zip(yy, bk):
    a.plot([b0, b1], [yv, yv], color=LIGHT, lw=1.6, zorder=1, solid_capstyle="round")
a.scatter([r[1] for r in bk], yy, s=16, color=BASE, zorder=2, label="base")
a.scatter([r[2] for r in bk], yy, s=16, color=FINAL, zorder=2, label="+ fusion + BP")
a.set_yticks(yy, [r[0] for r in bk]); a.set_xlim(78, 84.6); a.set_ylim(-0.6, 6.6); grid(a, "x")
a.set_xlabel("test accuracy (%)"); a.legend(loc="upper left", handletextpad=0.1)
a.set_title("(a) Backbones after fusion")
# Typed in from earlier exploration runs; these values are not saved as a results file in this repo.
cov = [100, 90, 80, 70, 60, 50]
b.plot(cov, [84.2, 88.42, 91.05, 92.89, 94.36, 95.23], color=FINAL, lw=LW, marker="o", ms=MS, label="final pipeline")
b.plot(cov, [81.38, 85.8, 89.24, 92.65, 94.99, 96.97], color=BASE, lw=LW, marker="o", ms=MS, label="NAGphormer (base)")
b.set_xlim(103, 47); b.set_xticks([100, 80, 60]); b.set_ylim(80, 99); grid(b, "both")
b.set_xlabel("coverage (% most confident kept)"); b.set_ylabel("test accuracy (%)"); b.legend(loc="lower right", bbox_to_anchor=(1.04, -0.02))
b.set_title("(b) Accuracy vs coverage")
elab = ["Baby", "MISSING", "Elec.→Cell Ph.", "no same-class nb.", "other"]
# Typed in from earlier exploration runs; these values are not saved as a results file in this repo.
ev = [2.14, 2.01, 1.60, 5.99, 4.07]; ecol = [TRAIN, TRAIN, FINAL, FINAL, FINAL]
ey = np.arange(5)[::-1]
c9.barh(ey, ev, 0.6, color=ecol)
for yv, t in zip(ey, ev):
    ann(c9, t + 0.12, yv, f"{t:.1f}", va="center")
c9.set_yticks(ey, elab); c9.set_xlim(0, 7.4); c9.set_xticks([0, 2, 4, 6]); grid(c9, "x"); c9.set_xlabel("% of test misclassified")
c9.set_title("(c) Remaining errors")
save(fig, "fig_results_extra.png")

# ================= Figure 9: decomposition revisited (C1) =================
rows = [("MiniLM + MLP", .8227, .8007, .6611, BASELINE), ("MiniLM + GCN", .8707, .8550, .7120, BASELINE),
        ("MiniLM + GT", .9039, .8811, .7648, BASELINE), ("gte + NAGphormer", .9288, .9101, .8138, BASE),
        ("GPT-3.5", .7824, .7871, .7230, GPT), ("Final pipeline", .9184, .9086, .8420, FINAL)]
fig, ax = plt.subplots(figsize=(4.4, 1.45))
yy = np.arange(len(rows))[::-1]
for yv, (nm, v, m, t, col) in zip(yy, rows):
    w1, w2 = 100 * (v - m), 100 * (m - t)
    ax.barh(yv, w1, color=LIGHT, height=0.62, edgecolor="white", lw=0.5)
    ax.barh(yv, w2, left=max(w1, 0), color=col, height=0.62, edgecolor="white", lw=0.5)
    ann(ax, max(w1, 0) + w2 + 0.3, yv, f"{100 * (v - t):.1f} = {w1:.1f} + {w2:.1f}", va="center")
ax.set_yticks(yy, [r[0] for r in rows]); ax.set_xlim(-0.8, 21.5); ax.axvline(0, color=MUTED, lw=0.7)
ax.set_xlabel("val − test accuracy (points)"); grid(ax, "x")
ax.legend(handles=[Patch(color=LIGHT, label="within-class"), Patch(color=BASELINE, label="class mix (model colour)")],
          loc="lower right")
save(fig, "fig_decomp.png")

print("ok")

# ================= Ladder: the final pipeline step by step =================
steps = ["GCN\n(OGB)", "GT\n(MiniLM)", "GT\n(ft gte)", "NAG-\nphormer", "+LLM\nfusion", "+PPR\nprior", "+BP\n(final)"]
acc = [68.1, 76.5, 81.25, 81.38, 83.58, 84.15, 84.20]; cold = [51.2, 70.9, 80.32, 79.87, 82.48, 83.11, 83.11]
f1 = [28.4, 36.5, 40.2, 39.8, 42.7, 43.6, 43.7]
fig, (a, b) = plt.subplots(1, 2, figsize=(5.6, 1.75), gridspec_kw={"width_ratios": [1.45, 1], "wspace": 0.3})
x = np.arange(len(steps))
for ax in (a, b):
    for lo, hi, lab in [(-0.5, 0.5, "Stage 1"), (0.5, 3.5, "Stage 3"), (3.5, 6.5, "Stage 4")]:
        ax.axvspan(lo, hi, color="#f4f3f0" if lab != "Stage 3" else "white", zorder=0, lw=0)
a.plot(x, acc, color=FINAL, lw=LW, marker="o", ms=MS, label="test accuracy", zorder=3)
a.plot(x, cold, color=FINAL, lw=1.1, ls=(0, (3, 1.5)), marker="o", ms=2.6, mfc="white", label="cold start", zorder=3)
for i in range(1, len(x)):
    d = acc[i] - acc[i - 1]
    ann(a, i, acc[i] + 1.4, f"+{d:.1f}" if d >= 0.1 else f"+{d:.2f}", ha="center", color=FINAL)
a.set_xticks(x, steps, fontsize=5.8); a.set_xlim(-0.5, 6.5); a.set_ylim(44, 91); grid(a)
a.set_ylabel("test accuracy (%)"); a.legend(loc="center right", bbox_to_anchor=(1.0, 0.45)); a.set_title("(a) Accuracy, step by step")
for lab, xc in [("Stage 1", 0), ("Stage 3", 2), ("Stage 4", 5)]:
    ann(a, xc, 45.5, lab, ha="center", color=MUTED, style="italic")
b.plot(x, f1, color=FINAL, lw=LW, marker="o", ms=MS, zorder=3)
ann(b, 0.2, f1[0], f"{f1[0]:.1f}", va="center", color=FINAL)
for i in [1, 2, 6]:
    ann(b, i - 0.15, f1[i] + 1.0, f"{f1[i]:.1f}", ha="right" if i < 6 else "center", color=FINAL)
b.set_xticks(x, ["1", "2", "3", "4", "5", "6", "7"]); b.set_xlim(-0.5, 6.5); b.set_ylim(25, 48); grid(b)
b.set_xlabel("step (as in a)"); b.set_ylabel("test macro-F1 (%)"); b.set_title("(b) Macro-F1")
save(fig, "fig_ladder.png")

# ================= Cross-stage: where each upgrade helps =================
cs = pd.read_csv(HERE / "cross_stage.csv").set_index("model")
st = ["S1 GCN, OGB features", "S2 gte-large + MLP", "S3 gte-large + NAGphormer", "S4 final pipeline"]
fig, (a, b) = plt.subplots(1, 2, figsize=(5.6, 1.95), gridspec_kw={"width_ratios": [0.9, 1.3], "wspace": 0.75})
lab = ["Stage 1: GCN, OGB", "Stage 2: gte + MLP", "Stage 3: gte + NAG", "Stage 4: final", "GPT-3.5"]
cols = [BASELINE, BASELINE, BASE, FINAL, GPT]
yy = np.arange(5)[::-1]
for yv, m, col in zip(yy, st + ["GPT-3.5"], cols):
    w1, w2 = cs.loc[m, "within"], cs.loc[m, "classmix"]
    a.barh(yv, max(w1, 0), color=LIGHT, height=0.62, edgecolor="white", lw=0.5)
    a.barh(yv, w2, left=max(w1, 0), color=col, height=0.62, edgecolor="white", lw=0.5)
    ann(a, max(w1, 0) + w2 + 0.3, yv, f"{w1 + w2:.1f}", va="center")
a.set_yticks(yy, lab); a.set_xlim(0, 18.5); grid(a, "x"); a.set_xlabel("val − test accuracy (points)")
a.legend(handles=[Patch(color=LIGHT, label="within-class"), Patch(color=BASELINE, label="class mix")], loc="lower right")
a.set_title("(a) Drop by stage")
sl = ["hop 1", "hop 2", "hop 3", "hop 4+", "degree 1", "degree 2+", "with title", "no title", "train-rare classes", "other learnable"]
sl_lab = ["1 hop", "2 hops", "3 hops", "4+ hops", "degree 1", "degree ≥ 2", "with title", "no title", "train-rare cls", "other classes"]
M = np.array([[100 * (cs.loc[st[j + 1], s] - cs.loc[st[j], s]) for j in range(3)] for s in sl])
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
cmap = LinearSegmentedColormap.from_list("gain", ["#eb6834", "#f7f6f2", "#86b6ef", "#2a78d6", "#104281"])
im = b.imshow(M.T, cmap=cmap, norm=TwoSlopeNorm(vmin=-5, vcenter=0, vmax=36), aspect="auto")
for i in range(M.shape[0]):
    for j in range(3):
        v = M[i, j]
        b.text(i, j, f"{v:+.0f}" if abs(v) >= 0.95 else f"{v:+.1f}", ha="center", va="center", fontsize=5.8,
               color="white" if v > 18 else INK)
b.set_yticks(range(3), ["S1→S2 text", "S2→S3 graph", "S3→S4 LLM+prior"], fontsize=6.2)
b.set_xticks(range(len(sl)), sl_lab, rotation=40, ha="right", fontsize=6.0)
b.tick_params(length=0); [sp_.set_visible(False) for sp_ in b.spines.values()]
b.set_title("(b) Gain per upgrade (points)")
save(fig, "fig_stages.png")

# ================= Figure: what the group's 34-arm evaluation and text length add =================
import json
af = pd.read_csv(HERE / "arms_fusion.csv"); xi = json.load(open(HERE / "extra_insights.json"))
fig, (a, b, c) = plt.subplots(1, 3, figsize=(6.4, 1.7), gridspec_kw={"wspace": 0.42, "width_ratios": [1.05, 0.9, 1.1]})
for s, mk, lab in [("normal", "o", "normal (neighbour-aware)"), ("cold", "o", "cold start (blend)")]:
    x = af[(af.setting == s) & (af.subset == "all")]
    bx, gy = 100 * x.base.values, 100 * (x.fused - x.base).values
    fc = FINAL if s == "normal" else "white"
    a.scatter(bx, gy, s=10, marker=mk, facecolor=fc, edgecolor=FINAL, linewidth=0.7, zorder=2, label=lab)
    k, c0 = np.polyfit(bx, gy, 1); xs = np.array([bx.min(), bx.max()])
    a.plot(xs, k * xs + c0, color=FINAL, lw=0.9, ls="-" if s == "normal" else "--", zorder=1)
a.axhline(0, color=MUTED, lw=0.6); grid(a, "both")
a.set_xlabel("base test accuracy (%)"); a.set_ylabel("gain from fusion (points)"); a.set_ylim(-1.5, 17)
a.legend(loc="upper right", handletextpad=0.2); a.set_title("(a) Gain vs base accuracy")
sl = [("no_train_nb", "none"), ("one_train_nb", "one"), ("two_plus_train_nb", "two+")]
xx = np.arange(3); wd = 0.36
for i, (s, col, hatch) in enumerate([("normal", FINAL, None), ("cold", "white", "////")]):
    v = [100 * (af[(af.setting == s) & (af.subset == k)].fused - af[(af.setting == s) & (af.subset == k)].base).mean() for k, _ in sl]
    bars = b.bar(xx + (i - 0.5) * wd, v, wd - 0.04, color=col, edgecolor=FINAL, linewidth=0.7, hatch=hatch,
                 label="normal" if s == "normal" else "cold start")
    b.bar_label(bars, fmt="%.1f", fontsize=F_ANN, color=MUTED, padding=1)
b.set_xticks(xx, [l for _, l in sl]); b.set_xlabel("labelled neighbours"); b.set_ylabel("mean gain (points)")
b.set_ylim(0, 7.2); grid(b); b.legend(loc="upper right"); b.set_title("(b) Gain by neighbourhood")
L = xi["length"]; bins = [r["bin"] for r in L]
for nm, col, lab in [("MiniLM + GT", BASELINE, "MiniLM + GT"), ("GPT-3.5", GPT, "GPT-3.5"),
                     ("NAGphormer", BASE, "NAGphormer (base)"), ("final", FINAL, "final pipeline")]:
    c.plot(range(len(L)), [100 * r[nm] for r in L], color=col, lw=LW, marker="o", ms=MS, label=lab)
c.set_xticks(range(len(L)), bins); c.set_xlabel("words in title + description"); c.set_ylabel("test accuracy (%)")
c.set_ylim(45, 95); grid(c); c.legend(loc="lower right", ncol=1); c.set_title("(c) Accuracy by text length")
save(fig, "fig_arms.png")
