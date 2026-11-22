"""Zero-shot category answers from a local open LLM (Qwen3.5-9B via SGLang), as a reproducible stand-in for GPT-3.5.

Usage: python scripts/llm_zero_shot.py --shard I --num-shards K [--limit N]
Run in the sglang-qwen35 env, one process per GPU (set CUDA_VISIBLE_DEVICES). Each shard writes
outputs/llm/qwen35_9b_shard{I}.jsonl with {"node", "answer"} per node. No labels are read.
"""
import argparse
import json
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
MODEL = "Qwen/Qwen3.5-9B"
MAX_WORDS = 200  # description words kept; titles are short

ap = argparse.ArgumentParser()
ap.add_argument("--shard", type=int, default=0)
ap.add_argument("--num-shards", type=int, default=1)
ap.add_argument("--limit", type=int, default=None)
args = ap.parse_args()

import sglang as sgl  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

import sys  # noqa: E402
sys.path.insert(0, str(REPO))
from src.text_clean import clean_text  # noqa: E402

names = pd.read_csv(REPO / "data/TAPE/labelidx2productcategory.csv.gz")["product category"].tolist()
cats = "\n".join(f"- {n}" for n in dict.fromkeys(" ".join(n.split()) for n in names if n != "MISSING"))
text = pd.read_csv(REPO / "data/TAPE/ogbn-products_subset.csv")
nodes = list(range(args.shard, len(text), args.num_shards))[: args.limit]

tok = AutoTokenizer.from_pretrained(MODEL)


def prompt(i):
    title = clean_text(text.title[i]) or "(none)"
    desc = " ".join(clean_text(text.content[i]).split()[:MAX_WORDS]) or "(none)"
    msg = (f"Classify this Amazon product into exactly one of the categories below.\n\nCategories:\n{cats}\n\n"
           f"Product title: {title}\nProduct description: {desc}\n\n"
           "Answer with the category name only, copied exactly from the list.")
    return tok.apply_chat_template([{"role": "user", "content": msg}], tokenize=False, add_generation_prompt=True,
                                   enable_thinking=False)


if __name__ == "__main__":
    llm = sgl.Engine(model_path=MODEL, mem_fraction_static=0.55, log_level="error")
    out_dir = REPO / "outputs" / "llm"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"qwen35_9b_shard{args.shard}{'_test' if args.limit else ''}.jsonl"
    outs = llm.generate([prompt(i) for i in nodes], {"temperature": 0, "max_new_tokens": 16})
    with open(path, "w") as f:
        for i, o in zip(nodes, outs):
            f.write(json.dumps({"node": i, "answer": o["text"].strip()}) + "\n")
    llm.shutdown()
    print("wrote", path, len(nodes))
