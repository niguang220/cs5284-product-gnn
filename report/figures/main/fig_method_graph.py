"""Figure 5 (method), relabelled for the collective-inference framing: steps 3-5 are inference on the product graph.
Same drawing as the Figure 5 block of figs_final.py; only the labels change."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT = Path(__file__).parent
MUTED = "#52514e"
BASE, GPT, FINAL = "#1baf7a", "#eb6834", "#2a78d6"
plt.rcParams.update({"font.family": "Liberation Sans", "text.color": "#0b0b0b", "mathtext.fontset": "cm",
                     "figure.dpi": 300, "savefig.dpi": 300})

fig, ax = plt.subplots(figsize=(6.3, 2.1)); ax.set_xlim(-0.5, 100.5); ax.set_ylim(-2.5, 36); ax.axis("off")
def box(x0, title, body, col, w=17.9):
    ax.add_patch(FancyBboxPatch((x0, 15.5), w, 18, boxstyle="round,pad=0.25,rounding_size=1.0", fc=col + "1f", ec=col, lw=0.9))
    ax.text(x0 + w / 2, 31.6, title, ha="center", va="top", fontsize=7.2, fontweight="bold")
    ax.text(x0 + w / 2, 27.4, body, ha="center", va="top", fontsize=6.2, color=MUTED, linespacing=1.3)
box(0.5, "1  Text encoder", "gte-large fine-tuned\non 14.7k train labels\n→ [CLS] features $X$", BASE)
box(20.75, "2  NAGphormer", "hop tokens\n$X,\\ \\hat{A}X,\\ \\ldots,\\ \\hat{A}^{6}X$\n$\\rightarrow\\ p(y \\mid x)$, $T$ on val", BASE)
box(41.0, "3  LLM evidence", "GPT-3.5 answer $g$\n$\\times$ confusion $P(g \\mid y)$\nweight 1 (Bayes)", GPT)
box(61.25, "4  Local prior", "PPR over the graph\n$+\\lambda\\,\\log\\,\\hat{\\pi}_i(y)\\,/\\,\\pi_{\\mathrm{tr}}(y)$\n$\\lambda = 1$ (Saerens)", FINAL)
box(81.5, "5  Messages", "one BP round via $H_c$\nco-purchase, kNN\nand sibling edges", FINAL)
for x0 in [18.65, 38.9, 59.15, 79.4]:
    ax.annotate("", (x0 + 1.8, 24.5), (x0, 24.5), arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=0.8, shrinkA=0, shrinkB=0, mutation_scale=7))
ax.plot([0.5, 38.65], [12.6, 12.6], color=BASE, lw=1.6)
ax.text(19.6, 10.6, "training: graph inside the GNN (GPU)", ha="center", va="top", fontsize=6.4, color=BASE)
ax.plot([41.0, 99.4], [12.6, 12.6], color=FINAL, lw=1.6)
ax.text(70.2, 10.6, "inference: collective inference on the graph (CPU, seconds;\nonly $T$, PPR teleport and $\\beta$ chosen on val)",
        ha="center", va="top", fontsize=6.4, color=FINAL, linespacing=1.2)
ax.text(50, 0.4, r"$b_i(y)\;\propto\;p(y \mid x_i)\,P(g_i \mid y)\,\left[\hat{\pi}_i(y)\,/\,\pi_{\mathrm{tr}}(y)\right]^{\lambda}\;\prod_{c}\,\prod_{j \in N_c(i)} m_j(y)^{\beta_c}, \qquad m_j(y) = \sum_{y^{\prime}}\, H_c[y, y^{\prime}]\; e_j(y^{\prime})$", ha="center", va="center", fontsize=8.5)
fig.savefig(OUT / "fig_method_graph.png", bbox_inches="tight", pad_inches=0.02, facecolor="white")
