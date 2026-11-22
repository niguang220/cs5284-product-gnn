"""Evaluation artifacts for the notebook's released-split GNN experiments.

No model training lives here. All filtered scores recompute false positives after
filtering true labels; retained categories' original F1 scores are never reused.
"""
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd


SUPPORT_GROUPS = ['0 (unseen)', '1-9', '10-99', '100-999', '1000+']


def support_groups(counts):
    counts = np.asarray(counts)
    return np.select([counts == 0, counts < 10, counts < 100, counts < 1000],
                     SUPPORT_GROUPS[:4], default=SUPPORT_GROUPS[4])


def wilson_interval(correct, total, confidence=0.95):
    """Marginal binomial recall intervals; undefined when support is zero."""
    correct, total = np.asarray(correct, float), np.asarray(total, float)
    z = NormalDist().inv_cdf((1 + confidence) / 2)
    denominator = np.where(total > 0, total, 1)
    p = correct / denominator
    scale = 1 + z * z / denominator
    centre = (p + z * z / (2 * denominator)) / scale
    radius = z * np.sqrt(p * (1 - p) / denominator + z * z / (4 * denominator**2)) / scale
    return (np.where(total > 0, np.maximum(0, centre - radius), np.nan),
            np.where(total > 0, np.minimum(1, centre + radius), np.nan))


def confusion(y, pred, classes):
    y, pred = np.asarray(y, int), np.asarray(pred, int)
    if len(y) != len(pred) or np.any(y < 0) or np.any(y >= classes) or np.any(pred < 0) or np.any(pred >= classes):
        raise ValueError('Invalid labels or prediction shape')
    return np.bincount(y * classes + pred, minlength=classes**2).reshape(classes, classes)


def summarise(cm):
    cm = np.asarray(cm)
    support, predicted, tp = cm.sum(1), cm.sum(0), cm.diagonal()
    precision = np.divide(tp, predicted, out=np.zeros(len(tp)), where=predicted > 0)
    recall = np.divide(tp, support, out=np.zeros(len(tp)), where=support > 0)
    f1 = np.divide(2 * precision * recall, precision + recall,
                   out=np.zeros(len(tp)), where=(precision + recall) > 0)
    present = support > 0
    return dict(accuracy=float(tp.sum() / support.sum()) if support.sum() else np.nan,
                macro_f1_present=float(f1[present].mean()) if present.any() else np.nan,
                macro_f1_47=float(f1.mean()),
                balanced_accuracy=float(recall[present].mean()) if present.any() else np.nan,
                products=int(support.sum())), precision, recall, f1


def category_table(cm, names, train_support):
    _, precision, recall, f1 = summarise(cm)
    support, tp = cm.sum(1), cm.diagonal()
    lower, upper = wilson_interval(tp, support)
    return pd.DataFrame(dict(category_id=np.arange(len(names)), category=names,
                             support=support, predicted_support=cm.sum(0), correct=tp,
                             precision=precision, recall=recall, f1=f1,
                             recall_ci95_lower=lower, recall_ci95_upper=upper,
                             train_support=train_support,
                             training_support_group=support_groups(train_support)))


def evaluate_predictions(y, pred, masks, names, architecture, condition, seed, best_epoch, input_dim):
    """Build reproducible metrics from the same predictions under multiple masks."""
    y, pred = np.asarray(y), np.asarray(pred)
    classes = len(names)
    train_support = np.bincount(y[masks['train']], minlength=classes)
    metadata = dict(architecture=architecture, condition=condition, seed=seed,
                    best_epoch=best_epoch, input_dim=input_dim)
    rows, group_rows, matrices, details = [], [], {}, {}
    for split, mask in masks.items():
        cm = confusion(y[mask], pred[mask], classes)
        matrices[split] = cm
        summary, _, _, _ = summarise(cm)
        rows.append(dict(**metadata, split=split, **summary))
        if split not in ('valid', 'test'):
            continue
        details[split] = category_table(cm, names, train_support)
        groups = {'full': mask, 'seen': mask & (train_support[y] > 0),
                  'unseen': mask & (train_support[y] == 0)}
        for label in SUPPORT_GROUPS:
            groups['train_support_' + label] = mask & (support_groups(train_support)[y] == label)
        for label, selected in groups.items():
            summary, _, _, _ = summarise(confusion(y[selected], pred[selected], classes))
            group_rows.append(dict(**metadata, split=split, group=label, **summary))
    comparison = details['valid'].merge(details['test'], on=['category_id', 'category', 'train_support',
                                                            'training_support_group'], suffixes=('_valid', '_test'))
    for metric in ('precision', 'recall', 'f1'):
        # A missing category is not evidence of a zero observed validation/test recall.
        comparison[f'{metric}_test_minus_valid'] = np.where(
            (comparison.support_valid > 0) & (comparison.support_test > 0),
            comparison[f'{metric}_test'] - comparison[f'{metric}_valid'], np.nan)
    # Hold true-category proportions fixed to uniform over categories observed in BOTH splits.
    common = (matrices['valid'].sum(1) > 0) & (matrices['test'].sum(1) > 0)
    standardised = []
    for split in ('valid', 'test'):
        cm = matrices[split].astype(float)
        row_support = cm.sum(1)
        transition = np.divide(cm, row_support[:, None], out=np.zeros_like(cm), where=row_support[:, None] > 0)
        weighted = transition * (common / common.sum())[:, None] if common.any() else np.zeros_like(cm)
        summary, _, _, _ = summarise(weighted)
        summary.pop('products')  # Weighted mass is not an evaluation sample count.
        standardised.append(dict(**metadata, split=split, common_categories=int(common.sum()),
                                 reference='uniform over categories with validation AND test support', **summary))
    prediction_rows = pd.DataFrame(dict(node_id=np.arange(len(y)), true_label=y, predicted_label=pred))
    prediction_rows['split'] = 'unassigned'
    for split, mask in masks.items():
        prediction_rows.loc[mask, 'split'] = split
    prediction_rows['train_support'] = train_support[y]
    prediction_rows = prediction_rows[prediction_rows.split.isin(['valid', 'test'])].copy()
    return dict(metrics=pd.DataFrame(rows), groups=pd.DataFrame(group_rows), details=details,
                matrices=matrices, comparison=comparison, standardised=pd.DataFrame(standardised),
                predictions=prediction_rows)


def save_evaluation(bundle, directory, prefix):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for key, suffix in [('metrics', 'metrics'), ('groups', 'group_metrics'),
                        ('comparison', 'validation_test_per_category'),
                        ('standardised', 'standardised_metrics'), ('predictions', 'predictions')]:
        bundle[key].to_csv(directory / f'{prefix}_{suffix}.csv', index=False)
    for split in ('valid', 'test'):
        bundle['details'][split].to_csv(directory / f'{prefix}_{split}_per_category.csv', index=False)
        cm = bundle['matrices'][split]
        pd.DataFrame(cm).rename_axis('true_category_id').to_csv(directory / f'{prefix}_{split}_confusion.csv')
        denominator = cm.sum(1)[:, None]
        normalised = np.divide(cm, denominator, out=np.full(cm.shape, np.nan), where=denominator > 0)
        pd.DataFrame(normalised).rename_axis('true_category_id').to_csv(
            directory / f'{prefix}_{split}_confusion_row_normalised.csv')


def seed_summary(frame, group_columns, metrics):
    """Seed variability is separate from evaluation-sample uncertainty."""
    return frame.groupby(group_columns, dropna=False)[metrics].agg(['mean', 'std', 'count'])


def plot_confusions(bundle, names, title):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(24, 11), sharey=True)
    image = None
    for ax, split in zip(axes, ('valid', 'test')):
        cm = bundle['matrices'][split]
        row_support = cm.sum(1)[:, None]
        normalised = np.divide(cm, row_support, out=np.full(cm.shape, np.nan), where=row_support > 0)
        image = ax.imshow(np.ma.masked_invalid(normalised), vmin=0, vmax=1, cmap='Blues')
        ax.set(title=split, xlabel='Predicted category ID', xticks=np.arange(len(names)))
        ax.tick_params(axis='x', labelsize=6, rotation=90)
    axes[0].set(yticks=np.arange(len(names)), yticklabels=[f'{i} - {name}' for i, name in enumerate(names)],
                ylabel='True category (blank rows: no support)')
    axes[0].tick_params(axis='y', labelsize=6)
    fig.suptitle(title)
    fig.colorbar(image, ax=axes, fraction=0.02, pad=0.02, label='Fraction of true category')
    return fig
