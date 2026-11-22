"""Final evaluation of every trained arm and every fixed improvement, from saved weights (no retraining).

Usage: python scripts/run_final_eval.py --eval {val,test}
  val   dry run on validation. Test labels are overwritten right after loading, so nothing can read them.
  test  the single test evaluation, run once after the val dry run reproduces the earlier val results.

Arms (saved weights from run_baselines.py, run_tape.py, run_gpt_pred.py and run_tape_repro.py; seeds 0, 1, 2):
  MiniLM + MLP / GCN / GraphSAGE; MiniLM + explanation + MLP / GCN; MiniLM + GPT answer + MLP / GCN; GPT's answer
  DeBERTa alone (fine-tuned on the text, or on TAPE's explanations), DeBERTa seeds 0, 1, 2
  ours: MLP / GCN / GraphSAGE with our settings on DeBERTa(text), DeBERTa(explanation), GPT's top 3, and their ensemble
  tape: the same with TAPE's settings and GPT's top 5; "tape ensemble" is TAPE's method
  Under ours and tape, every DeBERTa seed is crossed with GNN seeds 0, 1, 2. The GPT-answer models do not depend on the
  DeBERTa seed: they have rows only under DeBERTa seed 0, and join the ensemble of every DeBERTa seed.

Methods. Parameters were fixed on train-only holdouts before this script was written (a plan written on 7 Nov, before the test run; not in this repository).
p = softmax of the model, g = GPT's first answer (src/gpt_parse.py). None of the methods is new.
  model        argmax p
  blend        (1 - w) p + w onehot(g), w = 0.25                     a standard ensemble
  poe          log p + 0.5 log P(g | y)                              reliability fusion (Kerrigan et al. 2021)
  cond         log p + 0.5 log P(g | y, b), kappa = 5                 b: share of neighbours with the same answer
                                                                     (0, between, 1)
  bp           log p + 0.5 log P(g | y) + 0.25 sum_u log M(u | y)    graph-free arms only; exploratory
  edge         weight 0.25 on edges whose two GPT answers differ     src.models.GCN arms only
  edge+blend   edge, then blend                                      main improvement
  edge+cond    edge, then cond                                       exploratory
P(g | y), P(g | y, b) and the messages M use train labels only. Settings: normal, and cold (edges touching a test
node removed) for model and blend on graph arms.

Pre-declared main comparison: edge+blend vs blend on every GCN arm. The other comparisons are listed in COMPARISONS.
Subsets: all eval nodes; seen (class occurs in train); by the number of neighbours that are train nodes: none, any, one,
two or more.
Bootstrap: 1000 resamples of the eval nodes within each subset (all and seen), the same resamples for every method and
arm; a comparison averages over the runs of each side, then takes percentiles of the difference.

Writes results/4_fusion/final_{eval}_{runs,summary,compare}.csv, _meta.json and _predictions.npz (the predicted class of
every eval node for every run and method).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from src.arms import BASE_ARMS, NUM_CLASSES, REPRO_FEATURES, Features, load_base, load_repro  # noqa: E402
from src.data import load_products  # noqa: E402
from src.improve import (agreement_buckets, agreement_edge_weights, answer_given_class,  # noqa: E402
                         answer_given_class_and_bucket, blend_gpt, gcn_logits_weighted, gpt_one_hot,
                         neighbour_aware_fusion, neighbour_evidence, neighbour_tables, reliability_fusion)
from src.models import GCN  # noqa: E402
from src.train import cold_start_edges, masked_metrics  # noqa: E402

THREADS = 4
SEEDS = [0, 1, 2]
LM_SEEDS = [0, 1, 2]
K = NUM_CLASSES
W, GAMMA, KAPPA, RHO, ETA = 0.25, 0.5, 5.0, 0.25, 0.25
N_BOOT, BOOT_SEED, BOOT_CHUNK = 1000, 0, 250
BASE_NAMES = {"A": "MiniLM + MLP", "B": "MiniLM + GCN", "SAGE": "MiniLM + GraphSAGE",
              "G": "MiniLM + explanation + MLP", "H": "MiniLM + explanation + GCN",
              "P_MLP": "MiniLM + GPT answer + MLP", "P_GCN": "MiniLM + GPT answer + GCN"}
KIND_NAMES = {"MLP": "MLP", "GCN": "GCN", "SAGE": "GraphSAGE"}
FEATURE_NAMES = {"orig": "DeBERTa", "expl": "DeBERTa(explanation)", "pred": "GPT ranks", "ensemble": "ensemble"}

parser = argparse.ArgumentParser()
parser.add_argument("--eval", choices=["val", "test"], required=True)
args = parser.parse_args()
torch.set_num_threads(THREADS)

graph, _ = load_products()
N, ei = graph.num_nodes, graph.edge_index
y = graph.y.clone()
if args.eval == "val":
    y[graph.test_mask] = -999  # the dry run cannot read test labels
eval_mask = graph.val_mask if args.eval == "val" else graph.test_mask
train = graph.train_mask
assert not (eval_mask & train).any()

# Subsets of the eval nodes. "seen" uses the train label set only.
n_train_nb = torch.bincount(ei[1][train[ei[0]]], minlength=N)
subsets = {
    "all": eval_mask,
    "seen": eval_mask & torch.isin(y, y[train].unique()),
    "no_train_nb": eval_mask & (n_train_nb == 0),
    "has_train_nb": eval_mask & (n_train_nb > 0),
    "one_train_nb": eval_mask & (n_train_nb == 1),
    "two_plus_train_nb": eval_mask & (n_train_nb >= 2),
}
BOOT_SUBSETS = ["all", "seen"]
print({k: int(v.sum()) for k, v in subsets.items()}, flush=True)

features = Features(N)
gpt_top1 = features["rank"][:, 0]
gpt_hot = gpt_one_hot(gpt_top1, K)

# Evidence tables, from train labels only.
p_g_y = answer_given_class(y, gpt_top1, train, K)
bucket = agreement_buckets(ei, gpt_top1, N)
p_g_yb = answer_given_class_and_bucket(y, gpt_top1, train, bucket, p_g_y, KAPPA)
edge_label, edge_answer = neighbour_tables(y, gpt_top1, train, ei, K)
log_messages = neighbour_evidence(y, gpt_top1, train, ei, edge_label, edge_answer)
edge_w = agreement_edge_weights(ei, gpt_top1, RHO)
cold_edges = cold_start_edges(graph)
print("edges with weight 1:", int((edge_w == 1).sum()), "of", ei.size(1), flush=True)

runs, preds = [], {}


def methods_for(prob, edge_prob, graph_free):
    out = {"model": prob.argmax(1), "blend": blend_gpt(prob, gpt_hot, W).argmax(1),
           "poe": reliability_fusion(prob, gpt_top1, p_g_y, GAMMA).argmax(1),
           "cond": neighbour_aware_fusion(prob, gpt_top1, bucket, p_g_yb, GAMMA).argmax(1)}
    if graph_free:
        out["bp"] = (reliability_fusion(prob, gpt_top1, p_g_y, GAMMA) + ETA * log_messages).argmax(1)
    if edge_prob is not None:
        out["edge"] = edge_prob.argmax(1)
        out["edge+blend"] = blend_gpt(edge_prob, gpt_hot, W).argmax(1)
        out["edge+cond"] = neighbour_aware_fusion(edge_prob, gpt_top1, bucket, p_g_yb, GAMMA).argmax(1)
    return out


def record(info, setting, method, pred):
    key = f"{info['arm']}|{info['lm_seed']}|{info['seed']}|{setting}|{method}"
    preds[key] = pred[eval_mask].numpy().astype(np.int8)
    for name, mask in subsets.items():
        runs.append({**info, "setting": setting, "method": method, "subset": name, "n": int(mask.sum()),
                     **masked_metrics(y, pred, mask), "key": key})


def add_arm(info, logits, edge_logits=None, cold_logits=None):
    """logits: (N, K) of the model; edge_logits: the same GCN with weighted edges; cold_logits: under cold start."""
    graph_free = info["kind"] in ("MLP", "none")
    prob = logits.softmax(1)
    edge_prob = None if edge_logits is None else edge_logits.softmax(1)
    for method, pred in methods_for(prob, edge_prob, graph_free).items():
        record(info, "normal", method, pred)
    if cold_logits is not None:
        cold = cold_logits.softmax(1)
        record(info, "cold", "model", cold.argmax(1))
        record(info, "cold", "blend", blend_gpt(cold, gpt_hot, W).argmax(1))


@torch.no_grad()
def forward(model, x, edges):
    model.eval()
    return model(x, edges)


t0 = time.time()
# GPT's answer alone: a node with no answer (-1) counts as wrong.
record({"arm": "GPT answer", "family": "gpt", "config": "-", "feature": "gpt", "kind": "none", "code": "GPT",
        "lm_seed": -1, "seed": -1}, "normal", "model", gpt_top1)

for code, (kind, feature) in BASE_ARMS.items():
    for seed in SEEDS:
        model, x = load_base(code, seed, features)
        info = {"arm": BASE_NAMES[code], "family": "minilm", "config": "ours", "feature": feature, "kind": kind,
                "code": code, "lm_seed": -1, "seed": seed}
        logits = forward(model, x, ei)
        edge_logits = gcn_logits_weighted(model, x, ei, edge_w) if isinstance(model, GCN) else None
        cold_logits = None if kind == "MLP" else forward(model, x, cold_edges)
        add_arm(info, logits, edge_logits, cold_logits)
    print(f"{BASE_NAMES[code]} done, {time.time() - t0:.0f}s", flush=True)

for lm_seed in LM_SEEDS:
    for variant in ["orig", "expl"]:
        logits = torch.from_numpy(np.load(REPO / "outputs" / "tape_lm" /
                                          f"tape_lm_{variant}_deberta-base_seed{lm_seed}.pred.npy").astype(np.float32))
        name = "DeBERTa alone" if variant == "orig" else "DeBERTa(explanation) alone"
        add_arm({"arm": name, "family": "deberta", "config": "lm", "feature": variant, "kind": "none",
                 "code": f"M_{variant}", "lm_seed": lm_seed, "seed": lm_seed}, logits)

for lm_seed in LM_SEEDS:
    lm_features = Features(N, lm_seed=lm_seed)
    lm_features.cache["rank"] = features["rank"]
    lm_features.cache["gpt3"] = features["gpt3"]
    for config in ["ours", "tape"]:
        for kind in ["MLP", "GCN", "SAGE"]:
            for seed in SEEDS:
                parts = {"normal": [], "edge": [], "cold": []}
                for feature in REPRO_FEATURES:
                    model, x = load_repro(config, feature, kind, seed, lm_features)
                    logits = forward(model, x, ei)
                    edge_logits = gcn_logits_weighted(model, x, ei, edge_w) if isinstance(model, GCN) else None
                    cold_logits = None if kind == "MLP" else forward(model, x, cold_edges)
                    parts["normal"].append(logits)
                    parts["edge"].append(edge_logits)
                    parts["cold"].append(cold_logits)
                    if feature != "pred" or lm_seed == 0:
                        prefix = "TAPE " if config == "tape" else ""
                        add_arm({"arm": f"{prefix}{FEATURE_NAMES[feature]} + {KIND_NAMES[kind]}", "family": "deberta",
                                 "config": config, "feature": feature, "kind": kind, "code": f"{config}_{feature}_{kind}",
                                 "lm_seed": lm_seed, "seed": seed}, logits, edge_logits, cold_logits)

                def mean(xs):
                    return None if xs[0] is None else sum(xs) / len(xs)
                prefix = "TAPE " if config == "tape" else ""
                add_arm({"arm": f"{prefix}ensemble + {KIND_NAMES[kind]}", "family": "deberta", "config": config,
                         "feature": "ensemble", "kind": kind, "code": f"{config}_ensemble_{kind}",
                         "lm_seed": lm_seed, "seed": seed},
                        mean(parts["normal"]), mean(parts["edge"]), mean(parts["cold"]))
            print(f"lm{lm_seed} {config} {kind} done, {time.time() - t0:.0f}s", flush=True)

stem = REPO / "results" / "4_fusion" / f"final_{args.eval}"
df = pd.DataFrame(runs)
df.to_csv(f"{stem}_runs.csv", index=False)
np.savez_compressed(f"{stem}_predictions.npz", nodes=eval_mask.nonzero().flatten().numpy(),
                    **{k: v for k, v in preds.items()})
print(f"{len(df)} metric rows, {len(preds)} prediction vectors, {time.time() - t0:.0f}s", flush=True)

# ---- Bootstrap: per prediction vector, accuracy and macro-F1 on each resample. ----
eval_nodes = eval_mask.nonzero().flatten()
position = torch.full((N,), -1, dtype=torch.long)
position[eval_nodes] = torch.arange(len(eval_nodes))


def bootstrap_metrics(subset):
    """{key: (acc[N_BOOT], f1[N_BOOT])} for every prediction vector, resampling the nodes of one subset."""
    idx = position[subsets[subset].nonzero().flatten()].numpy()
    true = y[subsets[subset]].numpy()
    n = len(idx)
    rng = np.random.default_rng(BOOT_SEED)
    counts = np.stack([np.bincount(rng.integers(0, n, n), minlength=n) for _ in range(N_BOOT)]).astype(np.uint8)
    out = {}
    for key, full_pred in preds.items():
        pred = full_pred[idx].astype(np.int64)
        pred = np.where(pred < 0, K, pred)  # no answer is its own column
        cells = sp.csr_matrix((np.ones(n, np.float32), (np.arange(n), true * (K + 1) + pred)), shape=(n, K * (K + 1)))
        accs, f1s = [], []
        for start in range(0, N_BOOT, BOOT_CHUNK):
            conf = (cells.T @ counts[start:start + BOOT_CHUNK].T.astype(np.float32)).T.reshape(-1, K, K + 1)
            tp = conf[:, np.arange(K), np.arange(K)]
            row = conf.sum(2)
            col = conf[:, :, :K].sum(1)
            present = row > 0
            f1 = np.where(present, 2 * tp / np.maximum(row + col, 1e-12), 0.0)
            f1s.append(f1.sum(1) / present.sum(1))
            accs.append(tp.sum(1) / n)
        out[key] = (np.concatenate(accs), np.concatenate(f1s))
    return out


boot = {}
for subset in BOOT_SUBSETS:
    boot[subset] = bootstrap_metrics(subset)
    print(f"bootstrap {subset} done, {time.time() - t0:.0f}s", flush=True)

# The bootstrap's macro-F1 on the original sample must equal masked_metrics' (same definition).
check = df[(df.subset == "all") & (df.method == "model")].iloc[0]
pred = preds[check.key].astype(np.int64)
true = y[eval_mask].numpy()
cm = np.bincount(true * (K + 1) + np.where(pred < 0, K, pred), minlength=K * (K + 1)).reshape(K, K + 1)
tp, row, col = np.diag(cm[:, :K]), cm.sum(1), cm[:, :K].sum(0)
assert abs((2 * tp / np.maximum(row + col, 1e-12))[row > 0].mean() - check.macro_f1) < 1e-9

summary_rows = []
for (arm, setting, method, subset), sub in df.groupby(["arm", "setting", "method", "subset"], sort=False):
    entry = {"arm": arm, "setting": setting, "method": method, "subset": subset, "runs": len(sub),
             "acc": sub.acc.mean(), "acc_std": sub.acc.std(ddof=1), "macro_f1": sub.macro_f1.mean(),
             "macro_f1_std": sub.macro_f1.std(ddof=1)}
    if subset in BOOT_SUBSETS:
        a = np.mean([boot[subset][k][0] for k in sub.key], axis=0)
        f = np.mean([boot[subset][k][1] for k in sub.key], axis=0)
        entry.update(acc_ci_low=np.percentile(a, 2.5), acc_ci_high=np.percentile(a, 97.5),
                     f1_ci_low=np.percentile(f, 2.5), f1_ci_high=np.percentile(f, 97.5))
    summary_rows.append(entry)
summary = pd.DataFrame(summary_rows)
summary.to_csv(f"{stem}_summary.csv", index=False)

# ---- Comparisons: (label, arm a, method a, arm b, method b). ----
arms = list(dict.fromkeys(df.arm))
gcn_arms = [a for a in arms if (df[(df.arm == a)].method == "edge+blend").any()]
free_arms = [a for a in arms if (df[(df.arm == a)].method == "bp").any()]
fusion_arms = [a for a in arms if a != "GPT answer"]
COMPARISONS = (
    [("main: edge+blend - blend", a, "edge+blend", a, "blend") for a in gcn_arms]
    + [("edge - model", a, "edge", a, "model") for a in gcn_arms]
    + [("edge+blend - model", a, "edge+blend", a, "model") for a in gcn_arms]
    + [("blend - model", a, "blend", a, "model") for a in fusion_arms]
    + [("cond - poe", a, "cond", a, "poe") for a in fusion_arms]
    + [("cond - blend", a, "cond", a, "blend") for a in fusion_arms]
    + [("poe - blend", a, "poe", a, "blend") for a in fusion_arms]
    + [("edge+cond - blend", a, "edge+cond", a, "blend") for a in gcn_arms]
    + [("bp - blend", a, "bp", a, "blend") for a in free_arms]
    + [("arms: " + label, a, "model", b, "model") for label, a, b in [
        ("graph, MiniLM", "MiniLM + GCN", "MiniLM + MLP"),
        ("graph, DeBERTa", "DeBERTa + GCN", "DeBERTa + MLP"),
        ("encoder, MLP", "DeBERTa + MLP", "MiniLM + MLP"),
        ("encoder, GCN", "DeBERTa + GCN", "MiniLM + GCN"),
        ("fine-tuned LM vs GPT", "DeBERTa alone", "GPT answer"),
        ("TAPE ensemble vs DeBERTa, GraphSAGE", "TAPE ensemble + GraphSAGE", "DeBERTa + GraphSAGE"),
        ("explanation, MiniLM GCN", "MiniLM + explanation + GCN", "MiniLM + GCN"),
        ("GPT answer vs explanation, MiniLM GCN", "MiniLM + GPT answer + GCN", "MiniLM + explanation + GCN"),
    ]]
)

compare_rows = []
for label, arm_a, method_a, arm_b, method_b in COMPARISONS:
    for subset in BOOT_SUBSETS:
        side = {}
        for tag, arm, method in [("a", arm_a, method_a), ("b", arm_b, method_b)]:
            sub = df[(df.arm == arm) & (df.method == method) & (df.setting == "normal") & (df.subset == subset)]
            side[tag] = sub
        if side["a"].empty or side["b"].empty:
            continue
        da = (np.mean([boot[subset][k][0] for k in side["a"].key], axis=0)
              - np.mean([boot[subset][k][0] for k in side["b"].key], axis=0))
        df1 = (np.mean([boot[subset][k][1] for k in side["a"].key], axis=0)
               - np.mean([boot[subset][k][1] for k in side["b"].key], axis=0))
        compare_rows.append({
            "comparison": label, "arm_a": arm_a, "method_a": method_a, "arm_b": arm_b, "method_b": method_b,
            "subset": subset, "runs_a": len(side["a"]), "runs_b": len(side["b"]),
            "acc_diff": side["a"].acc.mean() - side["b"].acc.mean(),
            "acc_ci_low": np.percentile(da, 2.5), "acc_ci_high": np.percentile(da, 97.5),
            "f1_diff": side["a"].macro_f1.mean() - side["b"].macro_f1.mean(),
            "f1_ci_low": np.percentile(df1, 2.5), "f1_ci_high": np.percentile(df1, 97.5)})
compare = pd.DataFrame(compare_rows)
compare.to_csv(f"{stem}_compare.csv", index=False)

meta = {"eval": args.eval, "n_eval": int(eval_mask.sum()), "subsets": {k: int(v.sum()) for k, v in subsets.items()},
        "w": W, "gamma": GAMMA, "kappa": KAPPA, "rho": RHO, "eta": ETA, "seeds": SEEDS, "lm_seeds": LM_SEEDS,
        "n_boot": N_BOOT, "boot_seed": BOOT_SEED, "threads": THREADS,
        "tables": "P(g|y), P(g|y,b) and BP messages from train labels; smoothing 0.5",
        "test_labels": "read" if args.eval == "test" else "overwritten with -999 before use",
        "parameters_fixed": "on train-only holdouts before this script was written",
        "minutes": round((time.time() - t0) / 60, 1), "code": "scripts/run_final_eval.py"}
Path(f"{stem}_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
with pd.option_context("display.width", 220, "display.max_rows", 400, "display.max_columns", 20):
    main = compare[compare.comparison.str.startswith("main")]
    print(main[["arm_a", "subset", "acc_diff", "acc_ci_low", "acc_ci_high", "f1_diff", "f1_ci_low", "f1_ci_high"]]
          .round(4).to_string(index=False))
print(f"done in {(time.time() - t0) / 60:.1f} min", flush=True)
