"""Does truncating product text at 256 tokens hurt? Accuracy on truncated vs non-truncated products.

Usage: python scripts/analyze_truncation.py [--splits val test] [--seeds 0 1 2] [--arms A B SAGE G H]
Needs the weights from scripts/run_baselines.py and scripts/run_tape.py (outputs/weights/).

MiniLM reads at most 256 tokens of build_original_text(title, content); longer texts are cut. Products are
split into "long" (> 256 tokens, truncated) and "short" (<= 256). For each arm and seed:
  raw gap        acc(long) - acc(short)
  matched gap    acc(long) - acc(short) reweighted to the long group's class mix, so the gap is not just
                 long products belonging to harder (or easier) categories
  vs GPT         the same gap for GPT's first answer. GPT never sees MiniLM's truncation, so its gap
                 measures how hard long products are anyway. gap(arm) - gap(GPT) is the part that
                 truncation could explain (a difference in differences).
Also accuracy by length bin. Note: --splits test reads test labels; the later scripts in this project
are test-blind, so use --splits val if test should stay untouched.
Writes results/2_text_only/truncation_{summary,bins}.csv.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from src.arms import Features, load_base  # noqa: E402
from src.data import load_products  # noqa: E402
from src.encode import load_encoder  # noqa: E402
from src.text_clean import build_original_text  # noqa: E402

MAX_LEN = 256
BINS = [0, 64, 128, 256, 512, np.inf]
BIN_LABELS = ["<=64", "65-128", "129-256", "257-512 (truncated)", ">512 (truncated)"]

parser = argparse.ArgumentParser()
parser.add_argument("--splits", nargs="+", default=["val", "test"], choices=["train", "val", "test"])
parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
parser.add_argument("--arms", nargs="+", default=["A", "B", "SAGE", "G", "H"])
args = parser.parse_args()
torch.set_num_threads(4)

graph, text = load_products()
N = graph.num_nodes
y = graph.y.view(-1).numpy()

# Token length of every product's text, as MiniLM's tokenizer counts it (special tokens included).
cache = REPO / "outputs" / "token_lengths_original.npy"
if cache.exists():
    lengths = np.load(cache)
else:
    tok = load_encoder().tokenizer
    texts = [build_original_text(t, c) for t, c in zip(text.title, text.content)]
    lengths = np.array([len(ids) for ids in tok(texts, add_special_tokens=True, verbose=False)["input_ids"]])
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, lengths)
assert len(lengths) == N
long = lengths > MAX_LEN
print(f"truncated: {long.mean():.1%} of all products; median length {np.median(lengths):.0f} tokens")

# Predictions: GPT's first answer (no training) and each trained arm, per seed.
features = Features(N)
predictions = {("GPT", -1): features["rank"][:, 0].numpy()}
for arm in args.arms:
    for seed in args.seeds:
        try:
            model, x = load_base(arm, seed, features)
        except (FileNotFoundError, KeyError) as e:
            print(f"skip {arm} seed {seed}: {type(e).__name__} {e} (no weights, or arm unknown to src.arms)")
            continue
        model.eval()
        with torch.no_grad():
            predictions[(arm, seed)] = model(x, graph.edge_index).argmax(1).numpy()


def acc(pred, mask):
    return float((pred[mask] == y[mask]).mean()) if mask.any() else np.nan


def class_matched_acc(pred, source, target):
    """Accuracy on `source`, with classes weighted as in `target` (classes missing from source dropped)."""
    total, weight = 0.0, 0
    for c, n in zip(*np.unique(y[target], return_counts=True)):
        in_class = source & (y == c)
        if in_class.any():
            total += n * (pred[in_class] == c).mean()
            weight += n
    return total / weight if weight else np.nan


rows, bin_rows = [], []
bin_of = pd.cut(lengths, BINS, labels=BIN_LABELS, right=True)
for split in args.splits:
    in_split = graph[f"{split}_mask"].numpy()
    m_long, m_short = in_split & long, in_split & ~long
    for (arm, seed), pred in predictions.items():
        a_long, a_short = acc(pred, m_long), acc(pred, m_short)
        a_short_matched = class_matched_acc(pred, m_short, m_long)
        # Normal-approximation standard error of the raw gap (two independent proportions).
        se = np.sqrt(a_long * (1 - a_long) / m_long.sum() + a_short * (1 - a_short) / m_short.sum())
        rows.append({"split": split, "arm": arm, "seed": seed, "n_long": int(m_long.sum()),
                     "n_short": int(m_short.sum()), "acc_long": a_long, "acc_short": a_short,
                     "acc_short_class_matched": a_short_matched, "gap_raw": a_long - a_short,
                     "gap_raw_ci95": 1.96 * se, "gap_class_matched": a_long - a_short_matched})
        for label in BIN_LABELS:
            m = in_split & (bin_of == label)
            bin_rows.append({"split": split, "arm": arm, "seed": seed, "length_bin": label,
                             "n": int(m.sum()), "acc": acc(pred, m)})

df = pd.DataFrame(rows)
gpt_gap = df[df.arm == "GPT"].set_index("split")["gap_raw"]
df["gap_minus_gpt_gap"] = df["gap_raw"] - df["split"].map(gpt_gap)

out = REPO / "results" / "2_text_only"
df.to_csv(out / "truncation_summary.csv", index=False)
bins = pd.DataFrame(bin_rows)
bins.to_csv(out / "truncation_bins.csv", index=False)

cols = ["n_long", "n_short", "acc_long", "acc_short", "gap_raw", "gap_raw_ci95", "gap_class_matched",
        "gap_minus_gpt_gap"]
summary = df.groupby(["split", "arm"], sort=False)[cols].mean().round(4)
summary[["n_long", "n_short"]] = summary[["n_long", "n_short"]].astype(int)
table = bins.groupby(["split", "arm", "length_bin"], sort=False, observed=True)["acc"].mean().unstack("length_bin")
with pd.option_context("display.width", 200, "display.max_columns", 20):
    print("\nMean over seeds (gap = long minus short; negative = truncated products are classified worse)")
    print(summary)
    print("\nAccuracy by token-length bin (mean over seeds)")
    print(table[[b for b in BIN_LABELS if b in table.columns]].round(4))
    print("\nn per bin:", bins[(bins.arm == "GPT")].set_index(["split", "length_bin"])["n"].to_dict())