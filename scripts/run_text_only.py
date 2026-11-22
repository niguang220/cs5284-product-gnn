"""Text-only MLP arms on frozen MiniLM features, over three seeds, with the test-set slices. No graph is used.

Usage: python scripts/run_text_only.py [max_epochs]
Arms, all the same 2-layer MLP with the settings of scripts/run_baselines.py:
  A      MiniLM embedding of the product text (384-d)
  K      A plus 384 random unit-norm dimensions: capacity control for G (same width as G, no extra information)
  G      MiniLM embeddings of the product text and of TAPE's GPT explanation (768-d)
  P_MLP  product-text embedding plus GPT's 1st, 2nd and 3rd ranked categories as one-hot (384 + 3 x 47)
Features, seeds and training settings are identical to run_baselines.py (A, K), run_tape.py (G) and run_gpt_pred.py
(P_MLP), so with 4 threads the weights are the same and are saved under the same names in outputs/weights/; the graph
arms of those scripts are not trained here.
Needs outputs/embeddings/{original,tape}__all-MiniLM-L6-v2.npy (each checked against its manifest) and TAPE's GPT
responses (outputs/tape_responses/ogbn-products.zip). Writes results/2_text_only/text_only_{runs,summary}.csv and text_only_meta.json.
"""
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
from src.gpt_parse import category_patterns, ranked_categories  # noqa: E402
from src.models import MLP  # noqa: E402
from src.train import evaluate, masked_metrics, set_seed, test_subsets, train  # noqa: E402

THREADS = 4  # results are bit-identical only with the same thread count (see set_seed)
SEEDS = [0, 1, 2]
NUM_CLASSES = 47
HIDDEN = 256
TOP_K = 3
EMB_NAMES = {"orig": "original__all-MiniLM-L6-v2", "expl": "tape__all-MiniLM-L6-v2"}

max_epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 500
torch.set_num_threads(THREADS)
WEIGHTS_DIR = REPO / "outputs" / "weights"  # not kept in git
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

graph, text = load_products()
N = graph.num_nodes


def load_embedding(name):
    path = REPO / "outputs" / "embeddings" / f"{name}.npy"
    meta = json.loads((REPO / "results" / "shared" / "embeddings" / f"{name}.json").read_text())
    if hashlib.sha256(path.read_bytes()).hexdigest() != meta["sha256"]:
        raise ValueError(f"{path.name} does not match its manifest in results/shared/embeddings/")
    emb = torch.from_numpy(np.load(path))
    assert emb.shape == (N, 384), emb.shape
    assert torch.isfinite(emb).all()
    return emb, meta


x_orig, orig_meta = load_embedding(EMB_NAMES["orig"])
x_expl, expl_meta = load_embedding(EMB_NAMES["expl"])

# GPT's ranked categories, parsed from TAPE's responses as in run_gpt_pred.py (-1 = no answer at that rank).
patterns = category_patterns()
rank = torch.tensor([ranked_categories(t, patterns, TOP_K) for t in load_tape_explanations(N)])
h_pred = torch.cat([F.one_hot(rank[:, k].clamp(min=0), NUM_CLASSES) * (rank[:, k:k + 1] >= 0)
                    for k in range(TOP_K)], dim=1).float()

subsets = test_subsets(graph, text)
rows = []
for seed in SEEDS:
    # K: the same random vectors as run_baselines.py (Generator seed 1000 + seed, rows scaled to unit norm).
    gen = torch.Generator().manual_seed(1000 + seed)
    rand = torch.randn(N, 384, generator=gen)
    rand = rand / rand.norm(dim=1, keepdim=True)
    arms = {
        "A": (x_orig, "orig"),
        "K": (torch.cat([x_orig, rand], dim=1), "orig+rand"),
        "G": (torch.cat([x_orig, x_expl], dim=1), "orig+tape"),
        "P_MLP": (torch.cat([x_orig, h_pred], dim=1), "orig+gpt_rank"),
    }
    for arm, (x, features) in arms.items():
        set_seed(seed)
        model = MLP(x.size(1), HIDDEN, NUM_CLASSES)
        t0 = time.time()
        model, hist = train(model, x, graph, max_epochs=max_epochs)
        minutes = (time.time() - t0) / 60
        torch.save(model.state_dict(), WEIGHTS_DIR / f"{arm}_s{seed}.pt")
        best_epoch = int(np.argmax([h["val_acc"] for h in hist]))
        split_metrics, pred = evaluate(model, x, graph)
        base = {"arm": arm, "model": "MLP", "features": features, "seed": seed, "setting": "normal",
                "epochs_run": len(hist), "best_epoch": best_epoch, "minutes": round(minutes, 2), "threads": THREADS}
        for split in ["train", "val"]:
            rows.append({**base, "eval": split, "n": int(graph[f"{split}_mask"].sum()),
                         "acc": split_metrics[f"{split}_acc"], "macro_f1": split_metrics[f"{split}_macro_f1"]})
        for name, mask in subsets.items():
            rows.append({**base, "eval": name, "n": int(mask.sum()), **masked_metrics(graph.y, pred, mask)})
        print(f"seed {seed} arm {arm}: {len(hist)} epochs, best {best_epoch}, {minutes:.1f} min, "
              f"val {split_metrics['val_acc']:.4f}, val macro-F1 {split_metrics['val_macro_f1']:.4f}", flush=True)

out_dir = REPO / "results" / "2_text_only"
df = pd.DataFrame(rows)
df.to_csv(out_dir / "text_only_runs.csv", index=False)
summary = df.groupby(["arm", "setting", "eval"])[["acc", "macro_f1"]].agg(["mean", "std"]).round(4)
summary.to_csv(out_dir / "text_only_summary.csv")
meta = {"arms": {"A": "MiniLM(product text), 384-d", "K": "A + 384 random unit-norm dims",
                 "G": "MiniLM(product text) ; MiniLM(GPT explanation), 768-d",
                 "P_MLP": f"MiniLM(product text) ; one-hot of GPT's top {TOP_K} categories"},
        "embeddings": [orig_meta["file"], expl_meta["file"]],
        "embeddings_sha256": [orig_meta["sha256"], expl_meta["sha256"]],
        "encoder": orig_meta["encoder"], "encoder_revision": orig_meta["encoder_revision"],
        "seeds": SEEDS, "num_classes": NUM_CLASSES, "hidden": HIDDEN, "threads": THREADS,
        "max_epochs": max_epochs, "lr": 0.01, "weight_decay": 5e-4, "dropout": 0.5, "patience": 50,
        "selection": "early stopping on validation accuracy (src/train.py)",
        "graph": "not used: every arm is an MLP on per-product features",
        "std": "sample std (ddof=1) over seeds", "code": "scripts/run_text_only.py"}
(out_dir / "text_only_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
with pd.option_context("display.width", 200, "display.max_rows", 200):
    print(summary.xs("val", level="eval"))
