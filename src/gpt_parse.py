"""Read GPT's ranked category answers out of TAPE's explanations.

TAPE's prompt lists every category name and asks GPT-3.5 for the most likely ones, so each response names
categories in its own words order. We take category names in order of first appearance.
"""
import csv
import re

import pandas as pd
import torch

from src.data import DATA_DIR


def category_patterns(data_dir=DATA_DIR):
    """(class id, regex) for every category name; whitespace inside a name may vary, case is ignored."""
    names = pd.read_csv(data_dir / "labelidx2productcategory.csv.gz")["product category"].tolist()
    return [(i, re.compile(re.escape(re.sub(r"\s+", " ", n)).replace(r"\ ", r"\s+"), re.I))
            for i, n in enumerate(names) if n != "MISSING"]


def ranked_categories(text, patterns, k=3):
    """Class ids in order of first appearance in text, padded with -1 to length k.

    At the same position the longest name wins, and a shorter name inside one already taken is skipped
    ("Baby" inside "Baby Products").
    """
    hits = []
    for i, p in patterns:
        for m in p.finditer(text):
            hits.append((m.start(), -(m.end() - m.start()), i, m.end()))
    hits.sort()
    out, covered_until = [], -1
    for start, _, i, end in hits:
        if start < covered_until:
            continue
        if i not in out:
            out.append(i)
        covered_until = end
        if len(out) == k:
            break
    return out + [-1] * (k - len(out))


def load_tape_gpt_preds(path, num_nodes, k=5):
    """TAPE's own parse (gpt_preds/ogbn-products.csv) as a (num_nodes, k) tensor of class ids, -1 = none.

    Each row lists category mentions in order, with repeats. TAPE fills in class 1 when GPT names no category.
    """
    with open(path, newline="") as f:
        rows = [[int(v) for v in row] for row in csv.reader(f)]
    assert len(rows) == num_nodes, (len(rows), num_nodes)
    out = torch.full((num_nodes, k), -1, dtype=torch.long)
    for i, row in enumerate(rows):
        out[i, :min(len(row), k)] = torch.tensor(row[:k], dtype=torch.long)
    return out
