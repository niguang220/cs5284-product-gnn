"""Holdouts of train nodes that look like the test set, found without any labels (adversarial validation).

A classifier learns to tell train from test nodes from the MiniLM embedding and the log degree. Its out-of-fold
P(test) on train nodes says which of them resemble test nodes. Labels are not used.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def _classifier():
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=3000))


def domain_scores(x, degree, train_mask, test_mask, folds=5, seed=0):
    """5-fold out-of-fold P(test) for train and test nodes (NaN elsewhere), and the cross-validated AUC."""
    features = np.hstack([x, np.log1p(degree)[:, None].astype(np.float32)])
    idx = np.where(train_mask | test_mask)[0]
    is_test = test_mask[idx].astype(int)
    oof = np.zeros(len(idx))
    for fit, hold in StratifiedKFold(folds, shuffle=True, random_state=seed).split(idx, is_test):
        model = _classifier().fit(features[idx[fit]], is_test[fit])
        oof[hold] = model.predict_proba(features[idx[hold]])[:, 1]
    p_test = np.full(len(x), np.nan)
    p_test[idx] = oof
    return p_test, roc_auc_score(is_test, oof)


def make_holdouts(p_test, train_mask, n=2000):
    """Three holdouts of n train nodes, sorted by node id.

    testlike: the n train nodes with the highest P(test), the extreme tail of the test-like side.
    iw: sampled with probability proportional to p / (1 - p), the density ratio test / train, so it approximates
        the test distribution of the features rather than its tail.
    random: uniform, as a control.
    """
    train_idx = np.where(train_mask)[0]
    order = train_idx[np.argsort(-p_test[train_idx])]
    ratio = p_test[train_idx] / (1 - p_test[train_idx])
    return {
        "testlike": np.sort(order[:n]),
        "random": np.sort(np.random.default_rng(0).choice(train_idx, n, replace=False)),
        "iw": np.sort(np.random.default_rng(1).choice(train_idx, n, replace=False, p=ratio / ratio.sum())),
    }
