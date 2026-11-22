"""Checks that go with the final evaluation, all decided before it ran (in a plan written on 7 Nov; not in this repository).

Usage: python scripts/run_final_checks.py --eval {val,test}   (val is a dry run; test labels are overwritten)

  1 gpt_by_agreement   GPT's accuracy by the share of neighbours giving the same answer (0, between, 1), per split
  2 edge_same_label    share of edges whose ends share a category, for edges whose GPT answers agree or differ
  3 answer_shift       P(g | y) on eval nodes vs train: per-class GPT accuracy and total variation distance
  4 call_curve         query GPT only for the least confident eval products (by max softmax), then fuse; a random
                       order as the control. Fusion: blend (w = 0.25) and poe (gamma = 0.5)
  5 blind              the label-free predictions in results/4_fusion/blind_prediction.csv against the actual
                       accuracy (test only; needs results/4_fusion/final_test_runs.csv)
  6 drop               the val-to-eval accuracy drop, split into class mix, within-class change and unseen classes:
                       mix = sum_c pi_eval(c | seen) a_val(c) - a_val; within = a_eval(seen) - that; unseen =
                       a_eval - a_eval(seen). A class with no val nodes uses the method's overall val accuracy.
Writes results/4_fusion/final_checks_{eval}.json and results/4_fusion/final_checks_{eval}_call_curve.csv.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from src.arms import NUM_CLASSES, Features, load_base, load_repro  # noqa: E402
from src.data import load_products  # noqa: E402
from src.improve import agreement_buckets, answer_given_class, blend_gpt, gpt_one_hot, reliability_fusion  # noqa: E402

K = NUM_CLASSES
W, GAMMA = 0.25, 0.5
FRACTIONS = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]
BLIND_SHA256 = "1690994088ce1bbb478f2a3ee0e21f3f564d325c23c83359a0695a217f9c5aca"

parser = argparse.ArgumentParser()
parser.add_argument("--eval", choices=["val", "test"], required=True)
args = parser.parse_args()
torch.set_num_threads(4)

graph, _ = load_products()
N, ei = graph.num_nodes, graph.edge_index
y = graph.y.clone()
if args.eval == "val":
    y[graph.test_mask] = -999
eval_mask = graph.val_mask if args.eval == "val" else graph.test_mask
train, val = graph.train_mask, graph.val_mask
features = Features(N)
gpt = features["rank"][:, 0]
gpt_hot = gpt_one_hot(gpt, K)
out = {"eval": args.eval}


def acc(pred, mask):
    return float((pred[mask] == y[mask]).float().mean())


# 1. GPT's accuracy by neighbour agreement.
bucket = agreement_buckets(ei, gpt, N)
names = ["none agree", "some agree", "all agree"]
out["gpt_by_agreement"] = {
    split: {names[b]: {"n": int((mask & (bucket == b)).sum()), "gpt_acc": acc(gpt, mask & (bucket == b))}
            for b in range(3)}
    for split, mask in [("train", train), (args.eval, eval_mask)]}

# 2. Same-label rate of edges, by GPT agreement. An edge counts for a split if both ends are labelled and at least
# one end is in the split; train edges have both ends in train.
src, dst = ei
has = gpt >= 0
agree = (gpt[src] == gpt[dst]) & has[src] & has[dst]
labelled = train | eval_mask
both = labelled[src] & labelled[dst]
edge_sets = {"train": train[src] & train[dst], args.eval: both & (eval_mask[src] | eval_mask[dst])}
out["edge_same_label"] = {}
for split, keep in edge_sets.items():
    row = {}
    for tag, m in [("agree", keep & agree), ("differ", keep & ~agree)]:
        row[tag] = {"edges": int(m.sum()), "same_label": float((y[src[m]] == y[dst[m]]).float().mean())}
    out["edge_same_label"][split] = row

# 3. Shift of P(g | y): train vs eval, on classes present in both.
p_train = answer_given_class(y, gpt, train, K, smooth=0.0 + 1e-9)
p_eval = answer_given_class(y, gpt, eval_mask, K, smooth=0.0 + 1e-9)
n_train = torch.bincount(y[train], minlength=K)
n_eval = torch.bincount(y[eval_mask], minlength=K)
common = (n_train > 0) & (n_eval > 0)
tv = 0.5 * (p_train - p_eval).abs().sum(0)
weight = n_eval.float() * common / (n_eval.float() * common).sum()
diag = torch.arange(K)
out["answer_shift"] = {
    "classes_compared": int(common.sum()),
    "tv_weighted_by_eval_share": float((tv * weight).sum()),
    "gpt_acc_on_own_class_train_weighted": float((p_train[diag, diag] * weight).sum()),
    "gpt_acc_on_own_class_eval_weighted": float((p_eval[diag, diag] * weight).sum()),
    "per_class": [{"class": int(c), "n_train": int(n_train[c]), "n_eval": int(n_eval[c]),
                   "gpt_acc_train": float(p_train[c, c]), "gpt_acc_eval": float(p_eval[c, c]), "tv": float(tv[c])}
                  for c in common.nonzero().flatten()],
}

# 4. LLM call curve.
p_g_y = answer_given_class(y, gpt, train, K)
eval_nodes = eval_mask.nonzero().flatten()


def call_curve(name, prob):
    rows = []
    conf = prob.max(1).values
    orders = {"least confident": eval_nodes[conf[eval_nodes].argsort()],
              "random": eval_nodes[torch.randperm(len(eval_nodes), generator=torch.Generator().manual_seed(0))]}
    fused_blend = blend_gpt(prob, gpt_hot, W).argmax(1)
    fused_poe = reliability_fusion(prob, gpt, p_g_y, GAMMA).argmax(1)
    for order, nodes in orders.items():
        for f in FRACTIONS:
            asked = torch.zeros(N, dtype=torch.bool)
            asked[nodes[:int(round(f * len(nodes)))]] = True
            for fusion, fused in [("blend", fused_blend), ("poe", fused_poe)]:
                pred = torch.where(asked, fused, prob.argmax(1))
                rows.append({"arm": name, "order": order, "fraction": f, "fusion": fusion, "acc": acc(pred, eval_mask)})
    return rows


curve = []
for code, name in [("B", "MiniLM + GCN"), ("H", "MiniLM + explanation + GCN")]:
    for seed in [0, 1, 2]:
        model, x = load_base(code, seed, features)
        with torch.no_grad():
            curve += [{**r, "seed": seed} for r in call_curve(name, model(x, ei).softmax(1))]
lm_features = Features(N, lm_seed=0)
for seed in [0, 1, 2]:
    model, x = load_repro("ours", "orig", "GCN", seed, lm_features)
    with torch.no_grad():
        curve += [{**r, "seed": seed} for r in call_curve("DeBERTa + GCN", model(x, ei).softmax(1))]
curve = pd.DataFrame(curve)
curve.to_csv(REPO / "results" / "4_fusion" / f"final_checks_{args.eval}_call_curve.csv", index=False)
table = curve.groupby(["arm", "fusion", "order", "fraction"]).acc.mean().unstack("fraction")
out["call_curve_mean_acc"] = {f"{a} | {fu} | {o}": {str(k): float(v) for k, v in r.items()}
                              for (a, fu, o), r in table.iterrows()}

# 5 and 6 read the final evaluation's runs on the same split.
runs_path = REPO / "results" / "4_fusion" / f"final_{args.eval}_runs.csv"
runs = pd.read_csv(runs_path)
val_runs = pd.read_csv(REPO / "results" / "4_fusion" / "final_val_runs.csv")

# 5. Blind prediction (test only).
blind_path = REPO / "results" / "4_fusion" / "blind_prediction.csv"
if args.eval == "test" and blind_path.exists():
    # Written on Windows with CRLF line ends; git stores LF and may check out either, so hash the CRLF form.
    crlf = blind_path.read_bytes().replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")
    assert hashlib.sha256(crlf).hexdigest() == BLIND_SHA256, "blind prediction file changed"
    blind = pd.read_csv(blind_path)
    rename = {"TAPE ensemble MLP": "TAPE ensemble + MLP", "TAPE ensemble GCN": "TAPE ensemble + GCN",
              "TAPE ensemble SAGE": "TAPE ensemble + GraphSAGE"}
    actual = runs[(runs.subset == "all") & (runs.setting == "normal")]
    rows = []
    for _, b in blind.iterrows():
        arm, method = b.arm, "model"
        if arm.endswith(" + blend"):
            arm, method = arm[:-len(" + blend")], "blend"
        arm = rename.get(arm, arm)
        same_seed = True if arm == "DeBERTa alone" else actual.seed == b.seed  # the blind file stores seed 0 there
        hit = actual[(actual.arm == arm) & (actual.method == method) & (actual.lm_seed == b.lm_seed) & same_seed]
        assert len(hit) == 1, (b.arm, b.lm_seed, b.seed, len(hit))
        rows.append({"arm": b.arm, "blind": bool(b.blind), "actual": float(hit.acc.iloc[0]),
                     **{e: float(b[e]) for e in ["val", "gpt_shares", "confidence", "atc", "conf_bins", "agree+conf"]}})
    blind_df = pd.DataFrame(rows)
    err = {}
    for e in ["val", "gpt_shares", "confidence", "atc", "conf_bins", "agree+conf"]:
        for tag, sub in [("blind arms", blind_df[blind_df.blind]), ("known arms", blind_df[~blind_df.blind])]:
            d = sub[e] - sub.actual
            err[f"{e} | {tag}"] = {"mean_error": float(d.mean()), "mean_abs_error": float(d.abs().mean())}
    out["blind"] = {"errors": err,
                    "by_arm": blind_df.groupby("arm")[["actual", "gpt_shares", "atc", "val"]].mean().round(4)
                    .reset_index().to_dict("records")}

# 6. Drop decomposition, for GPT and a few arms.
preds = np.load(REPO / "results" / "4_fusion" / f"final_{args.eval}_predictions.npz")
val_preds = np.load(REPO / "results" / "4_fusion" / "final_val_predictions.npz")
seen = torch.isin(y, y[train].unique())
y_eval = y[eval_mask].numpy()
seen_eval = seen[eval_mask].numpy()
y_val = y[val].numpy()


def decompose(eval_pred, val_pred):
    a_val = float((val_pred == y_val).mean())
    per_class_val = {c: float((val_pred[y_val == c] == c).mean()) for c in np.unique(y_val)}
    classes, counts = np.unique(y_eval[seen_eval], return_counts=True)
    shares = counts / counts.sum()
    mix_val = float(sum(s * per_class_val.get(c, a_val) for c, s in zip(classes, shares)))
    a_eval = float((eval_pred == y_eval).mean())
    a_eval_seen = float((eval_pred[seen_eval] == y_eval[seen_eval]).mean())
    return {"val": a_val, "eval": a_eval, "drop": a_eval - a_val, "class_mix": mix_val - a_val,
            "within_class": a_eval_seen - mix_val, "unseen_classes": a_eval - a_eval_seen}


out["drop"] = {}
targets = [("GPT answer", "model"), ("MiniLM + MLP", "model"), ("MiniLM + GCN", "model"), ("DeBERTa alone", "model"),
           ("DeBERTa + GCN", "model"), ("DeBERTa + GCN", "blend"), ("DeBERTa + GCN", "edge+blend")]
for arm, method in targets:
    keys = runs[(runs.arm == arm) & (runs.method == method) & (runs.setting == "normal") & (runs.subset == "all")].key
    parts = [decompose(preds[k].astype(np.int64), val_preds[k].astype(np.int64)) for k in keys]
    out["drop"][f"{arm} | {method}"] = {k: float(np.mean([p[k] for p in parts])) for k in parts[0]}

path = REPO / "results" / "4_fusion" / f"final_checks_{args.eval}.json"
path.write_text(json.dumps(out, indent=2) + "\n")
for key in ["gpt_by_agreement", "edge_same_label", "drop"]:
    print(key, json.dumps(out[key], indent=1))
print("answer_shift", {k: v for k, v in out["answer_shift"].items() if k != "per_class"})
print(table.round(4).to_string())
if "blind" in out:
    print(json.dumps(out["blind"]["errors"], indent=1))
