"""Average temperature-scaled probabilities of several base models (seed i with seed i) into one tag.

Usage: python exp/main/make_ensemble.py NAME TAG [TAG ...]  -> outputs/probs/ENS-NAME_{normal,cold}_s{seed}.npy
Temperatures are fitted on val NLL per member; nothing else is fitted.
"""
import sys
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(REPO))
from src.data import load_products
from src.label_shift import fit_temperature, temper
g, _ = load_products(); y, va = g.y.numpy(), g.val_mask.numpy()
name, tags = sys.argv[1], sys.argv[2:]
for setting in ["normal", "cold"]:
    for s in range(3):
        ps = []
        for t in tags:
            f = REPO / f"outputs/probs/{t}_{setting}_s{s}.npy"
            f = f if f.exists() else REPO / f"outputs/probs/{t}_normal_s{s}.npy"
            p = np.load(f).astype(np.float64); ps.append(temper(p, fit_temperature(p, y, va)))
        np.save(REPO / f"outputs/probs/ENS-{name}_{setting}_s{s}.npy", np.mean(ps, 0).astype(np.float32))
print("saved ENS-" + name, tags)
