"""Paired significance tests for two arms evaluated on the same nodes."""
import numpy as np
from scipy.stats import binomtest


def paired_bootstrap(y, pred_a, pred_b, n_boot=10000, seed=0):
    """Accuracy difference acc(a) - acc(b) and its 95% percentile interval.

    Nodes are resampled with replacement, the same resample for both arms, so the interval reflects
    which nodes happen to be in the test set, not training randomness (use several seeds for that).
    """
    y, pred_a, pred_b = np.asarray(y), np.asarray(pred_a), np.asarray(pred_b)
    diff = (pred_a == y).astype(float) - (pred_b == y).astype(float)
    rng = np.random.default_rng(seed)
    boots = np.array([diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(n_boot)])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"diff": float(diff.mean()), "ci_low": float(lo), "ci_high": float(hi), "n": len(diff)}


def mcnemar(y, pred_a, pred_b):
    """Exact McNemar test: among nodes where exactly one arm is right, is each arm right half the time?"""
    y, pred_a, pred_b = np.asarray(y), np.asarray(pred_a), np.asarray(pred_b)
    a_only = int(((pred_a == y) & (pred_b != y)).sum())
    b_only = int(((pred_a != y) & (pred_b == y)).sum())
    p = binomtest(a_only, a_only + b_only, 0.5).pvalue if a_only + b_only else 1.0
    return {"a_only": a_only, "b_only": b_only, "p_value": float(p)}
