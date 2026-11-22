"""TAPE reproduction: h_orig, h_expl and h_pred, each with MLP / GCN / GraphSAGE, plus TAPE's ensemble.

Usage: python scripts/run_tape_repro.py --config {tape,ours} [--lm-seed 0] [--max-epochs N]
Configs:
  tape  TAPE's settings at commit d9881f7 (run.sh, core/config.py, core/GNNs): 4 layers, hidden 128, BatchNorm,
        dropout 0, weight decay 0, at most 200 epochs, GPT's top 5 through a learned embedding, TAPE's early stopping
        (a val tie resets the patience). Compare with TAPE's Table 1.
        Still different from TAPE: 3 seeds instead of 4; no RevGAT; DeBERTa fine-tuned with batch 9 on one GPU, while
        run.sh exposes 4 GPUs (the Trainer then uses batch 36, with fewer steps); GNN seeds are nested in DeBERTa seeds.
  ours  The settings of our other arms (scripts/run_baselines.py): 2 layers, hidden 256, dropout 0.5,
        weight decay 5e-4, at most 500 epochs, GPT's top 3 as one-hot. Compare with A/B/G/H.
Features:
  orig  DeBERTa fine-tuned on the original text, CLS vector (768-d)
  expl  DeBERTa fine-tuned on TAPE's GPT explanations, CLS vector (768-d)
  pred  GPT's ranked categories: TAPE's own gpt_preds/ogbn-products.csv under tape (see
        results/shared/tape_gpt_preds_manifest.json), our src/gpt_parse.py under ours
Ensemble: the mean of the three models' logits, as in TAPE's core/GNNs/ensemble_trainer.py.
Settings: normal; cold (edges touching a test node removed); cold_strict (edges touching a val or test node
removed, so val nodes are as isolated as test nodes are under cold). MLP runs only under normal.

Test labels are not read: metrics cover train and val only. Weights go to outputs/tape_repro/ for the single
test evaluation at the end of the project. Writes results/3_gnn_text/tape_repro_{config}_lm{seed}_{runs,summary}.csv and _meta.json.
"""
import argparse
import hashlib
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
from src.data import load_products, load_tape_explanations  # noqa: E402
from src.gpt_parse import category_patterns, load_tape_gpt_preds, ranked_categories  # noqa: E402
from src.models import GCN, MLP, SAGE, TapeNet  # noqa: E402
from src.train import cold_start_edges, masked_metrics, set_seed, train, train_tape_stopping  # noqa: E402

THREADS = 4  # results are bit-identical only with the same thread count (see set_seed)
SEEDS = [0, 1, 2]
NUM_CLASSES = 47
KINDS = ["MLP", "GCN", "SAGE"]
CONFIGS = {
    "tape": {"hidden": 128, "weight_decay": 0.0, "max_epochs": 200, "top_k": 5},
    "ours": {"hidden": 256, "weight_decay": 5e-4, "max_epochs": 500, "top_k": 3},
}

parser = argparse.ArgumentParser()
parser.add_argument("--config", choices=CONFIGS, required=True)
parser.add_argument("--lm-seed", type=int, default=0)
parser.add_argument("--max-epochs", type=int, default=None)
args = parser.parse_args()
cfg = CONFIGS[args.config]
max_epochs = cfg["max_epochs"] if args.max_epochs is None else args.max_epochs
torch.set_num_threads(THREADS)

graph, _ = load_products()


def load_lm_vectors(text_variant):
    name = f"tape_lm_{text_variant}_deberta-base_seed{args.lm_seed}"
    path = REPO / "outputs" / "tape_lm" / f"{name}.emb.npy"
    meta = json.loads((REPO / "results" / "shared" / "embeddings" / f"{name}.json").read_text())
    if hashlib.sha256(path.read_bytes()).hexdigest() != meta["sha256"]:
        raise ValueError(f"{path.name} does not match its manifest in results/shared/embeddings/")
    emb = torch.from_numpy(np.load(path).astype(np.float32))  # stored as float16
    assert emb.shape == (graph.num_nodes, 768), emb.shape
    assert torch.isfinite(emb).all()
    return emb, meta


x_orig, orig_meta = load_lm_vectors("orig")
x_expl, expl_meta = load_lm_vectors("expl")

if args.config == "tape":
    preds_meta = json.loads((REPO / "results" / "shared" / "tape_gpt_preds_manifest.json").read_text())
    preds_path = REPO / preds_meta["file"]
    if hashlib.sha256(preds_path.read_bytes()).hexdigest() != preds_meta["sha256"]:
        raise ValueError(f"{preds_path} does not match results/shared/tape_gpt_preds_manifest.json")
    rank = load_tape_gpt_preds(preds_path, graph.num_nodes, cfg["top_k"])
    x_pred = rank + 1  # 0 = no answer, as in TAPE's load_gpt_preds
else:
    patterns = category_patterns()
    rank = torch.tensor([ranked_categories(t, patterns, cfg["top_k"])
                         for t in load_tape_explanations(graph.num_nodes)])
    x_pred = torch.cat([F.one_hot(rank[:, k].clamp(min=0), NUM_CLASSES) * (rank[:, k:k + 1] >= 0)
                        for k in range(cfg["top_k"])], dim=1).float()
answers_per_node = (rank >= 0).sum(dim=1).bincount(minlength=cfg["top_k"] + 1).tolist()
print("nodes with 0, 1, ... GPT answers:", answers_per_node, flush=True)
features = {"orig": x_orig, "expl": x_expl, "pred": x_pred}


def build_model(kind, feature):
    x = features[feature]
    if args.config == "ours":
        return {"MLP": MLP, "GCN": GCN, "SAGE": SAGE}[kind](x.size(1), cfg["hidden"], NUM_CLASSES)
    use_pred = feature == "pred"
    in_dim = cfg["hidden"] * cfg["top_k"] if use_pred else x.size(1)
    return TapeNet(kind, in_dim, cfg["hidden"], NUM_CLASSES, use_pred=use_pred)


edge_index = graph.edge_index
held_out = graph.val_mask | graph.test_mask
all_settings = {
    "normal": edge_index,
    "cold": cold_start_edges(graph),
    "cold_strict": edge_index[:, ~(held_out[edge_index[0]] | held_out[edge_index[1]])],
}
weights_dir = REPO / "outputs" / "tape_repro"
weights_dir.mkdir(parents=True, exist_ok=True)
stem = REPO / "results" / "3_gnn_text" / f"tape_repro_{args.config}_lm{args.lm_seed}"
rows = []


def add_rows(base, pred):
    for split in ["train", "val"]:
        mask = graph[f"{split}_mask"]
        rows.append({**base, "eval": split, "n": int(mask.sum()), **masked_metrics(graph.y, pred, mask)})


@torch.no_grad()
def logits_of(model, x, edges):
    model.eval()
    return model(x, edges)


for seed in SEEDS:
    for kind in KINDS:
        settings = {"normal": all_settings["normal"]} if kind == "MLP" else all_settings
        logits_by_setting = {setting: [] for setting in settings}
        for feature, x in features.items():
            model = build_model(kind, feature)
            if feature == "pred" and args.lm_seed != 0:
                # GPT's answers do not depend on the DeBERTa seed: reuse the model from --lm-seed 0.
                model.load_state_dict(torch.load(weights_dir / f"{args.config}_lm0_{feature}_{kind}_s{seed}.pt",
                                                 weights_only=True))
                run_info = {}
            else:
                set_seed(seed)
                model = build_model(kind, feature)
                t0 = time.time()
                fit = train_tape_stopping if args.config == "tape" else train
                model, hist = fit(model, x, graph, weight_decay=cfg["weight_decay"], max_epochs=max_epochs)
                best_val = max(h["val_acc"] for h in hist)
                last_best = max(h["epoch"] for h in hist if h["val_acc"] == best_val)
                first_best = min(h["epoch"] for h in hist if h["val_acc"] == best_val)
                run_info = {"epochs_run": len(hist), "best_epoch": last_best if args.config == "tape" else first_best,
                            "minutes": round((time.time() - t0) / 60, 2)}
                torch.save(model.state_dict(),
                           weights_dir / f"{args.config}_lm{args.lm_seed}_{feature}_{kind}_s{seed}.pt")
            for setting, edges in settings.items():
                logits = logits_of(model, x, edges)
                logits_by_setting[setting].append(logits)
                add_rows({"config": args.config, "lm_seed": args.lm_seed, "model": kind, "feature": feature,
                          "seed": seed, "setting": setting, **run_info, "threads": THREADS}, logits.argmax(dim=1))
            val = [r["acc"] for r in rows if r["model"] == kind and r["feature"] == feature and r["seed"] == seed
                   and r["setting"] == "normal" and r["eval"] == "val"][0]
            print(f"seed {seed} {kind} {feature}: {run_info or 'reused from lm seed 0'}, val {val:.4f}", flush=True)
        for setting, logits in logits_by_setting.items():
            add_rows({"config": args.config, "lm_seed": args.lm_seed, "model": kind, "feature": "ensemble",
                      "seed": seed, "setting": setting, "threads": THREADS},
                     (sum(logits) / len(logits)).argmax(dim=1))
        pd.DataFrame(rows).to_csv(f"{stem}_runs.csv", index=False)

df = pd.DataFrame(rows)
summary = df.groupby(["model", "feature", "setting", "eval"])[["acc", "macro_f1"]].agg(["mean", "std"]).round(4)
summary.to_csv(f"{stem}_summary.csv")
meta = {"config": args.config, **cfg, "max_epochs": max_epochs, "lm_seed": args.lm_seed, "seeds": SEEDS,
        "num_classes": NUM_CLASSES, "threads": THREADS, "lr": 0.01, "patience": 50,
        "dropout": 0.0 if args.config == "tape" else 0.5,
        "layers": 4 if args.config == "tape" else 2,
        "embeddings": [orig_meta["file"], expl_meta["file"]],
        "embeddings_sha256": [orig_meta["sha256"], expl_meta["sha256"]],
        "nodes_by_number_of_gpt_answers": answers_per_node,
        "ensemble": "mean of the logits of the orig, expl and pred models with the same seed",
        "std": "sample std (ddof=1) over GNN seeds; all seeds share one DeBERTa seed",
        "test_labels": "not read; metrics cover train and val only",
        "code": "scripts/run_tape_repro.py"}
Path(f"{stem}_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
with pd.option_context("display.width", 200, "display.max_rows", 200):
    print(summary.xs("val", level="eval"))
