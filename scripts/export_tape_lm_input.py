"""Write the input files for the TAPE LM reproduction notebook (notebooks/2_text_only/tape_lm_colab.ipynb).

Usage: python scripts/export_tape_lm_input.py
One JSON line per node with the two texts TAPE fine-tunes DeBERTa on, built exactly as TAPE does
(core/data_utils/load_products.py and load.py at commit d9881f7; no cleaning, missing values become "nan"):
  "orig": f"Product:{title}; Description: {content}\n"
  "expl": the GPT-3.5 response text
Labels are included for train and val nodes only (val picks the best checkpoint); test nodes get -1.
Written in parts under 10 MB to outputs/tape_lm/ (not in git), for upload to Colab.
"""
import gzip
import json
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from src.data import DATA_DIR, load_products, load_tape_explanations  # noqa: E402

OUT_DIR = REPO / "outputs" / "tape_lm"
PARTS = 4

graph, _ = load_products()
# Re-read the CSV without our loader so NaN stays NaN and formats as "nan", as in TAPE.
raw = pd.read_csv(DATA_DIR / "ogbn-products_subset.csv")
orig = [f"Product:{ti}; Description: {cont}\n" for ti, cont in zip(raw["title"], raw["content"])]
expl = load_tape_explanations(graph.num_nodes)

split = ["train" if t else "val" if v else "test"
         for t, v in zip(graph.train_mask.tolist(), graph.val_mask.tolist())]
y = graph.y.tolist()
lines = [json.dumps({"i": i, "split": split[i], "label": y[i] if split[i] != "test" else -1,
                     "orig": orig[i], "expl": expl[i]}, ensure_ascii=False) + "\n"
         for i in range(graph.num_nodes)]

OUT_DIR.mkdir(parents=True, exist_ok=True)
size = -(-len(lines) // PARTS)
for p in range(PARTS):
    path = OUT_DIR / f"tape_lm_input_part{p}.jsonl.gz"
    with gzip.open(path, "wt", encoding="utf-8") as f:
        f.writelines(lines[p * size:(p + 1) * size])
    print(f"{path.name}: {path.stat().st_size / 1e6:.1f} MB")
print(f"{len(lines)} nodes; labels for {split.count('train')} train + {split.count('val')} val")
