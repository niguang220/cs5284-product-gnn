"""Full-batch training and evaluation shared by all arms (MLP, GCN, GraphSAGE)."""
import copy
import hashlib
import random

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score


def set_seed(seed):
    """Seed Python, NumPy and PyTorch.

    Results are bit-identical only with the same seed AND the same CPU thread count
    (torch.get_num_threads()): float sums are reordered across threads. Record the thread count with results.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def masked_metrics(y, pred, mask):
    """Accuracy and macro-F1 over the nodes in mask.

    Macro-F1 averages over the classes present in the true labels of these nodes, so predicting a class
    that does not occur in the split counts as an error for the true class but does not add a zero-F1 class.
    """
    y_true, y_pred = y[mask].cpu().numpy(), pred[mask].cpu().numpy()
    labels = np.unique(y_true)
    return {
        "acc": float((y_true == y_pred).mean()),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
    }


def cold_start_edges(graph):
    """edge_index without any edge that touches a test node (cold-start test at inference)."""
    test = graph.test_mask
    keep = ~(test[graph.edge_index[0]] | test[graph.edge_index[1]])
    return graph.edge_index[:, keep]


def test_subsets(graph, text):
    """Masks of the test-set slices we report, keyed by name.

    test_seen: class occurs in train (uses the train label set only).
    test_dedup: text does not duplicate a training product (the text fingerprint from the EDA notebook:
    lowercased title + content, whitespace collapsed; blank text never counts as a duplicate).
    test_degX: node degree in the full graph.
    """
    test = graph.test_mask
    seen = test & torch.isin(graph.y, graph.y[graph.train_mask].unique())

    norm = text.title.fillna("").astype(str).str.strip() + " " + text.content.fillna("").astype(str).str.strip()
    norm = norm.str.lower().str.replace(r"\s+", " ", regex=True).str.strip()
    fp = norm.map(lambda s: hashlib.sha256(s.encode("utf-8")).hexdigest())
    train_fps = set(fp[graph.train_mask.numpy() & norm.ne("").values])
    dedup = test & ~torch.from_numpy((fp.isin(train_fps) & norm.ne("")).values)

    deg = torch.bincount(graph.edge_index[0], minlength=graph.num_nodes)
    return {
        "test": test,
        "test_seen": seen,
        "test_dedup": dedup,
        "test_deg0": test & (deg == 0),
        "test_deg1": test & (deg == 1),
        "test_deg2": test & (deg == 2),
        "test_deg3-5": test & (deg >= 3) & (deg <= 5),
        "test_deg6+": test & (deg >= 6),
    }


@torch.no_grad()
def evaluate(model, x, graph, edge_index=None):
    """Accuracy and macro-F1 on each split. edge_index overrides graph.edge_index (cold-start test)."""
    model.eval()
    edge_index = graph.edge_index if edge_index is None else edge_index
    pred = model(x, edge_index).argmax(dim=1)
    metrics = {}
    for split in ["train", "val", "test"]:
        for name, value in masked_metrics(graph.y, pred, graph[f"{split}_mask"]).items():
            metrics[f"{split}_{name}"] = value
    return metrics, pred


def train(model, x, graph, lr=0.01, weight_decay=5e-4, max_epochs=500, patience=50, loss_offset=None):
    """Train on train nodes; keep the weights with the best validation accuracy.

    Stops when validation accuracy has not improved for `patience` epochs. Test labels are never used here.
    loss_offset: added to the logits in the loss only (logit adjustment, Menon et al. 2021: log of the train prior),
    so the model's own logits estimate a class-balanced posterior.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_val, best_state, best_epoch, history = -1.0, None, 0, []
    for epoch in range(max_epochs):
        model.train()
        optimizer.zero_grad()
        out = model(x, graph.edge_index)
        assert out.size(1) > int(graph.y.max()), "num_classes is smaller than the largest label id"
        logits = out if loss_offset is None else out + loss_offset
        loss = F.cross_entropy(logits[graph.train_mask], graph.y[graph.train_mask])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            pred = model(x, graph.edge_index).argmax(dim=1)
        val_acc = float((pred[graph.val_mask] == graph.y[graph.val_mask]).float().mean())
        history.append({"epoch": epoch, "loss": float(loss), "val_acc": val_acc})

        if val_acc > best_val:
            best_val, best_state, best_epoch = val_acc, copy.deepcopy(model.state_dict()), epoch
        elif epoch - best_epoch >= patience:
            break

    model.load_state_dict(best_state)
    return model, history


def train_tape_stopping(model, x, graph, lr=0.01, weight_decay=0.0, max_epochs=200, patience=50):
    """train(), but with the early stopping of TAPE's core/GNNs/gnn_utils.py (commit d9881f7).

    A val tie counts as an improvement: it keeps the later weights and resets the patience. Val has only 1,572 nodes,
    so ties are common and these runs go on longer than under train(). Used to reproduce TAPE's numbers only.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_val, best_state, wait, history = -1.0, None, 0, []
    for epoch in range(max_epochs):
        model.train()
        optimizer.zero_grad()
        out = model(x, graph.edge_index)
        assert out.size(1) > int(graph.y.max()), "num_classes is smaller than the largest label id"
        loss = F.cross_entropy(out[graph.train_mask], graph.y[graph.train_mask])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            pred = model(x, graph.edge_index).argmax(dim=1)
        val_acc = float((pred[graph.val_mask] == graph.y[graph.val_mask]).float().mean())
        history.append({"epoch": epoch, "loss": float(loss), "val_acc": val_acc})

        if val_acc >= best_val:
            best_val, best_state, wait = val_acc, copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1
            if wait >= patience:
                break

    model.load_state_dict(best_state)
    return model, history
