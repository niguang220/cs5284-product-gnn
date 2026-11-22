"""The improvements and the diagnosis on holdouts of train nodes that resemble test nodes (results/shared/holdouts.json).

Usage: python scripts/run_holdout.py [--max-epochs N] [--out-dir DIR]
Models are trained on the official train nodes minus the holdout, whose nodes stay in the graph without labels,
like test nodes. Only MiniLM arms run here: the DeBERTa arms were fine-tuned on these nodes, so they are not unseen.

Evaluated on the holdout nodes, whose labels are train labels:
  normal: all edges, overall and by number of neighbours that are training nodes (0, 1, 2+).
  cold_strict: edges touching a holdout, val or test node removed, with imputation zero, self, knn, both.
    Virtual neighbours come from the remaining train nodes.
  Each with the GPT blend at w = 0 and w = 0.25, the best w on this holdout, and cross-fitted w (each half of the
  holdout uses the w chosen on the other half). w = 0.25 was read off H on these same holdouts, so its gains are
  slightly in-sample; cross-fitted w is the honest estimate.
Early stopping: B and H are also trained with early stopping on half of the holdout and evaluated on the other half,
next to the same models stopped on val, to see whether a test-like set picks a better epoch.

Test labels are not read. Finished runs are reused from their saved weights, so the script can be restarted.
Writes results/4_fusion/holdout_{runs,summary}.csv.
"""
import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from src.arms import NUM_CLASSES, Features, base_model  # noqa: E402
from src.data import load_products  # noqa: E402
from src.improve import (blend_gpt, crossfit_blend, drop_edges_touching, gpt_one_hot, impute_edges,  # noqa: E402
                         split_in_two)
from src.train import masked_metrics, set_seed, train  # noqa: E402

THREADS = 4
SEEDS = [0, 1, 2, 3, 4]
STOP_SEEDS = [0, 1, 2]
ARMS = ["A", "B", "SAGE", "H"]
STOP_ARMS = ["B", "H"]
HOLDOUTS = ["testlike", "iw"]
W = 0.25
W_GRID = np.round(np.arange(0, 1.0001, 0.05), 2)
HOW = ["zero", "self", "knn", "both"]

parser = argparse.ArgumentParser()
parser.add_argument("--max-epochs", type=int, default=500)
parser.add_argument("--out-dir", default=None, help="for a smoke test: weights and tables go here")
args = parser.parse_args()
torch.set_num_threads(THREADS)

graph, _ = load_products()
N, y = graph.num_nodes, graph.y
features = Features(N)
x_norm = F.normalize(features["orig"], dim=1)
gpt_top1 = features["rank"][:, 0]
gpt_hot = gpt_one_hot(gpt_top1, NUM_CLASSES)
holdouts = json.loads((REPO / "results" / "shared" / "holdouts.json").read_text())
weights_dir = Path(args.out_dir) if args.out_dir else REPO / "outputs" / "holdout"
weights_dir.mkdir(parents=True, exist_ok=True)
stem = (Path(args.out_dir) if args.out_dir else REPO / "results" / "4_fusion") / "holdout"
rows = []


def mask_of(nodes):
    mask = torch.zeros(N, dtype=torch.bool)
    mask[nodes] = True
    return mask


def add_blend_rows(base, prob, mask, halves=None):
    """Metrics on mask at w = 0, w = W, the best w on mask and, given the two halves of mask, cross-fitted w."""
    assert not graph.test_mask[mask].any()
    rows.append({**base, "w_rule": "none", "w": 0.0, "n": int(mask.sum()), **masked_metrics(y, prob.argmax(1), mask)})
    rows.append({**base, "w_rule": "fixed", "w": W, "n": int(mask.sum()),
                 **masked_metrics(y, blend_gpt(prob, gpt_hot, W).argmax(1), mask)})
    curve = [float((blend_gpt(prob[mask], gpt_hot[mask], w).argmax(1) == y[mask]).float().mean()) for w in W_GRID]
    best = float(W_GRID[int(np.argmax(curve))])
    rows.append({**base, "w_rule": "best_on_set", "w": best, "n": int(mask.sum()),
                 **masked_metrics(y, blend_gpt(prob, gpt_hot, best).argmax(1), mask)})
    if halves is not None:
        pred, chosen = crossfit_blend(prob, gpt_hot, y, *halves, W_GRID)
        rows.append({**base, "w_rule": "crossfit", "w": sum(chosen) / 2, "n": int(mask.sum()),
                     **masked_metrics(y, pred, mask)})


def fit(arm, seed, g, tag):
    """Train with the shared settings, or load the weights of a finished run."""
    path = weights_dir / f"{tag}_{arm}_s{seed}.pt"
    model, x = base_model(arm, features)
    if path.exists():
        model.load_state_dict(torch.load(path, weights_only=True))
        return model.eval(), x
    set_seed(seed)
    model, x = base_model(arm, features)
    t0 = time.time()
    model, hist = train(model, x, g, max_epochs=args.max_epochs)
    torch.save(model.state_dict(), path)
    print(f"{tag} {arm} seed {seed}: {len(hist)} epochs, {(time.time() - t0) / 60:.1f} min", flush=True)
    return model.eval(), x


def probs(model, x, edges):
    with torch.no_grad():
        return model(x, edges).softmax(1)


setups = {}
for name in HOLDOUTS:
    nodes = torch.tensor(holdouts[name])
    assert graph.train_mask[nodes].all()
    mask = mask_of(nodes)
    g = copy.copy(graph)
    g.train_mask = graph.train_mask & ~mask
    ei = graph.edge_index
    n_train_neighbours = torch.bincount(ei[1][g.train_mask[ei[0]]], minlength=N)
    groups = {"all": mask, "train_nb_0": mask & (n_train_neighbours == 0), "train_nb_1": mask & (n_train_neighbours == 1),
              "train_nb_2+": mask & (n_train_neighbours >= 2)}
    pool = g.train_mask.nonzero().flatten()
    cold = drop_edges_touching(ei, mask | graph.val_mask | graph.test_mask)
    imputed = {how: impute_edges(cold, how, x_norm, pool, N) for how in HOW}
    half_a, half_b = split_in_two(nodes, N)
    g_stop = copy.copy(g)
    g_stop.val_mask = half_a
    setups[name] = {"mask": mask, "g": g, "groups": groups, "imputed": imputed, "half_a": half_a, "half_b": half_b,
                    "g_stop": g_stop}
    print(f"{name}: train {int(g.train_mask.sum())}, holdout {len(nodes)}, groups "
          f"{ {k: int(v.sum()) for k, v in groups.items()} }", flush=True)
    for group, group_mask in groups.items():
        rows.append({"holdout": name, "arm": "GPT", "seed": -1, "stop": "none", "setting": "normal", "impute": "zero",
                     "group": group, "w_rule": "none", "w": 0.0, "n": int(group_mask.sum()),
                     **masked_metrics(y, gpt_top1, group_mask)})

for seed in SEEDS:  # outer loop on seeds, so a partial run still has complete seeds
    for name, setup in setups.items():
        for arm in ARMS:
            model, x = fit(arm, seed, setup["g"], name)
            base = {"holdout": name, "arm": arm, "seed": seed, "stop": "val"}
            prob = probs(model, x, graph.edge_index)
            halves = (setup["half_a"], setup["half_b"])
            for group, group_mask in setup["groups"].items():
                add_blend_rows({**base, "setting": "normal", "impute": "zero", "group": group}, prob, group_mask,
                               halves if group == "all" else None)
            add_blend_rows({**base, "setting": "normal", "impute": "zero", "group": "half_b"}, prob, setup["half_b"])
            if arm != "A":
                for how, edges in setup["imputed"].items():
                    add_blend_rows({**base, "setting": "cold_strict", "impute": how, "group": "all"},
                                   probs(model, x, edges), setup["mask"], halves)
            if arm in STOP_ARMS and seed in STOP_SEEDS:
                model, x = fit(arm, seed, setup["g_stop"], f"{name}_stophalf")
                add_blend_rows({"holdout": name, "arm": arm, "seed": seed, "stop": "holdout_half", "setting": "normal",
                                "impute": "zero", "group": "half_b"},
                               probs(model, x, graph.edge_index), setup["half_b"])
            pd.DataFrame(rows).to_csv(f"{stem}_runs.csv", index=False)

df = pd.DataFrame(rows)
df.to_csv(f"{stem}_runs.csv", index=False)
keys = ["holdout", "arm", "stop", "setting", "impute", "group", "w_rule"]
summary = df.groupby(keys)[["acc", "macro_f1", "w"]].agg(["mean", "std"]).round(4)
summary.to_csv(f"{stem}_summary.csv")
print("written", len(df), "rows")
