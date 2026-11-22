"""Does a graph variant's gain survive label-shift correction? Temperature on val, EM prior (Saerens 2002) estimated
from the model's own unlabelled test posteriors, then the posterior is reweighted (src/label_shift.py). No test labels.

Usage: python exp/graph/stack_check.py DIR/VARIANT ...
"""
import sys

import numpy as np

from common import REPO, hop_buckets, hops_to_train, load
from src.label_shift import adjust, em_prior, fit_temperature, prior, temper
from src.train import masked_metrics
import torch

graph, _ = load()
y = graph.y.numpy()
tr, va, te = graph.train_mask.numpy(), graph.val_mask.numpy(), graph.test_mask.numpy()
b = hop_buckets(graph, hops_to_train(graph))
pi_tr = prior(y, tr)
seen = pi_tr > 0  # EM is ill-posed for classes with no train node (ratio pi/pi_tr undefined): restrict to seen ones
for spec in sys.argv[1:]:
    res = {}
    for st in ["normal", "cold"]:
        rr = []
        for s in range(3):
            p = REPO / f"{spec}_{st}_s{s}.npy"
            if not p.exists():
                break
            prob = temper(np.load(p).astype(np.float64), fit_temperature(np.load(p).astype(np.float64), y, va))
            prob = prob[:, seen] / prob[:, seen].sum(1, keepdims=True)
            q = np.zeros((len(y), len(seen)))
            q[:, seen] = adjust(prob, pi_tr[seen], em_prior(prob[te], pi_tr[seen]))
            m = masked_metrics(graph.y, torch.from_numpy(q.argmax(1)), graph.test_mask)
            m.update({f"hop{k}": float((q.argmax(1)[v.numpy()] == y[v.numpy()]).mean()) for k, v in b.items()})
            rr.append(m)
        if rr:
            res[st] = {k: round(float(np.mean([r[k] for r in rr])), 4) for k in rr[0]}
    print(spec, "EM-adjusted", res, flush=True)
