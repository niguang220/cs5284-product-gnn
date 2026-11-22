"""Text-only Transformer trained from scratch (arm T), over three seeds, with the test-set slices.

Usage: python scripts/run_text_transformer.py
A small grid is tried on seed 0 and the configuration with the best validation accuracy is run on all seeds.
Writes results/2_text_only/text_transformer_{runs,summary,sweep}.csv and text_transformer_meta.json; per-seed logits and
pooled embeddings go to outputs/ (not in git) for error analysis and graph + text combinations.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from src.data import load_products  # noqa: E402
from src.models import TextTransformer  # noqa: E402
from src.text_clean import build_original_text  # noqa: E402
from src.train import masked_metrics, set_seed, test_subsets  # noqa: E402
from src.train_text import predict, tokenize, train  # noqa: E402

SEEDS = [0, 1, 2]
NUM_CLASSES = 47
MAX_LEN = 256
DIM = 256
# Grid tried on seed 0 only; selection uses validation accuracy.
GRID = [
    {"num_layers": 2, "lr": 5e-4, "dropout": 0.1},
    {"num_layers": 2, "lr": 1e-3, "dropout": 0.1},
    {"num_layers": 4, "lr": 5e-4, "dropout": 0.1},
    {"num_layers": 2, "lr": 5e-4, "dropout": 0.3},
]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

graph, text = load_products()
texts = [build_original_text(t, c) for t, c in zip(text.title, text.content)]
ids, mask, vocab_size = tokenize(texts, MAX_LEN)
print("truncated to", MAX_LEN, "tokens:", int(mask[:, -1].sum()), "of", len(ids), "products")
y = graph.y
train_idx, val_idx = graph.train_mask.nonzero().view(-1), graph.val_mask.nonzero().view(-1)
subsets = test_subsets(graph, text)


def run(cfg, seed):
    set_seed(seed)
    model = TextTransformer(vocab_size, NUM_CLASSES, dim=DIM, heads=4, num_layers=cfg["num_layers"],
                            max_len=MAX_LEN, dropout=cfg["dropout"])
    t0 = time.time()
    model, hist = train(model, ids, mask, y, train_idx, val_idx, device, lr=cfg["lr"], seed=seed)
    minutes = (time.time() - t0) / 60
    logits, emb = predict(model, ids, mask, device)
    return model, hist, minutes, logits, emb


sweep = []
for cfg in GRID:
    _, hist, minutes, _, _ = run(cfg, SEEDS[0])
    best = max(hist, key=lambda h: h["val_acc"])
    sweep.append({**cfg, "seed": SEEDS[0], "best_epoch": best["epoch"], "val_acc": best["val_acc"],
                  "epochs_run": len(hist), "minutes": round(minutes, 2)})
    print("sweep", sweep[-1], flush=True)
best_cfg = {k: v for k, v in max(sweep, key=lambda r: r["val_acc"]).items() if k in GRID[0]}
print("selected", best_cfg)

rows = []
out_dir = REPO / "outputs"
(out_dir / "predictions").mkdir(parents=True, exist_ok=True)
(out_dir / "embeddings").mkdir(parents=True, exist_ok=True)
for seed in SEEDS:
    model, hist, minutes, logits, emb = run(best_cfg, seed)
    pred = logits.argmax(1)
    np.save(out_dir / "predictions" / f"T_seed{seed}.npy", logits.numpy())
    np.save(out_dir / "embeddings" / f"text_transformer__seed{seed}.npy", emb.numpy())
    best_epoch = int(np.argmax([h["val_acc"] for h in hist]))
    base = {"arm": "T", "model": "TextTransformer", "features": "tokens", "seed": seed, "setting": "normal",
            "epochs_run": len(hist), "best_epoch": best_epoch, "minutes": round(minutes, 2), "device": device.type}
    for split in ["train", "val"]:
        m = graph[f"{split}_mask"]
        rows.append({**base, "eval": split, "n": int(m.sum()), **masked_metrics(y, pred, m)})
    for name, m in subsets.items():
        rows.append({**base, "eval": name, "n": int(m.sum()), **masked_metrics(y, pred, m)})
    val = masked_metrics(y, pred, graph.val_mask)["acc"]
    test = masked_metrics(y, pred, graph.test_mask)["acc"]
    print(f"seed {seed}: {len(hist)} epochs, best {best_epoch}, {minutes:.1f} min, val {val:.4f}, test {test:.4f}",
          flush=True)

res = REPO / "results" / "2_text_only"
df = pd.DataFrame(rows)
df.to_csv(res / "text_transformer_runs.csv", index=False)
pd.DataFrame(sweep).to_csv(res / "text_transformer_sweep.csv", index=False)
summary = df.groupby(["arm", "setting", "eval"])[["acc", "macro_f1"]].agg(["mean", "std"]).round(4)
summary.to_csv(res / "text_transformer_summary.csv")
meta = {"text": "build_original_text(title, content)", "tokenizer": "WordPiece vocabulary of all-MiniLM-L6-v2 "
        "(no pretrained weights)", "max_len": MAX_LEN, "dim": DIM, "heads": 4, "ffn": 4 * DIM,
        "pooling": "mean over non-padding tokens", "selected": best_cfg, "grid": GRID,
        "selection": "best validation accuracy on seed 0", "optimizer": "AdamW, weight_decay 0.01, 1 warm-up epoch, "
        "grad clip 1.0", "batch_size": 64, "max_epochs": 50, "patience": 5, "seeds": SEEDS, "device": device.type,
        "cold_start": "not applicable: the model never sees edges", "std": "sample std (ddof=1) over seeds",
        "code": "scripts/run_text_transformer.py"}
(res / "text_transformer_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
with pd.option_context("display.width", 200, "display.max_rows", 200):
    print(summary)
