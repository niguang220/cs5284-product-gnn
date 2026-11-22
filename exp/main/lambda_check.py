"""Is the 'no tuned weight' fusion over- or under-confident? Scale the LLM log-likelihood by λ, chosen on val."""
import csv, sys
from pathlib import Path
import numpy as np, torch
REPO = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(REPO))
from src.data import load_products
from src.label_shift import K, confusion, fit_temperature, temper
from src.train import masked_metrics
g, _ = load_products(); y = g.y.numpy(); tr, va, te = g.train_mask.numpy(), g.val_mask.numpy(), g.test_mask.numpy()
gpt = np.array([int(r[0]) for r in csv.reader(open(REPO / "outputs/tape_gpt_preds/ogbn-products.csv"))])
qw = np.load(REPO / "outputs/llm/qwen35_9b.npy")
L = np.log(confusion(gpt, y, tr | va)[gpt % (K + 1)]) + np.log(confusion(qw, y, tr | va)[qw % (K + 1)])
yt = torch.from_numpy(y)
for tag in ["GT", "SAGE@ft-bge-base_s0"]:
    for lam in [0.5, 0.75, 1.0, 1.25, 1.5]:
        r = []
        for s in range(3):
            p = np.load(REPO / f"outputs/probs/{tag}_normal_s{s}.npy").astype(float); p = temper(p, fit_temperature(p, y, va))
            pred = torch.from_numpy((np.log(p) + lam * L).argmax(1))
            r.append([masked_metrics(yt, pred, torch.from_numpy(m))["acc"] for m in (va, te)])
        r = np.mean(r, 0); print(f"{tag:22s} lambda={lam:<5} val {r[0]:.4f} test {r[1]:.4f}")
