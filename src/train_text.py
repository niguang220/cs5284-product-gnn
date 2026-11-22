"""Tokenisation and mini-batch training for the text Transformer (TextTransformer in src/models.py)."""
import copy

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from src.encode import MODEL_NAME, MODEL_REVISION


def tokenize(texts, max_len=256):
    """Token ids [N, max_len] (int64, right-padded) and mask [N, max_len] (True on real tokens), plus vocab size.

    Only MiniLM's WordPiece vocabulary is reused, so token boundaries match the sentence encoder; no pretrained
    weights are loaded. max_len 256 matches the sentence encoder's truncation.
    """
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, revision=MODEL_REVISION)
    enc = tok(list(texts), max_length=max_len, truncation=True, padding="max_length", return_tensors="pt")
    assert tok.pad_token_id == 0
    return enc["input_ids"], enc["attention_mask"].bool(), tok.vocab_size


def _batches(idx, batch_size, shuffle, generator=None):
    if shuffle:
        idx = idx[torch.randperm(len(idx), generator=generator)]
    for i in range(0, len(idx), batch_size):
        yield idx[i : i + batch_size]


def _trim(ids, mask):
    # Drop columns that are padding in every row of the batch.
    length = int(mask.sum(1).max())
    return ids[:, :length], mask[:, :length]


@torch.no_grad()
def predict(model, ids, mask, device, batch_size=512):
    """Logits [N, C] and pooled embeddings [N, dim] for every node, on CPU."""
    model.eval()
    logits, embs = [], []
    for b in _batches(torch.arange(len(ids)), batch_size, shuffle=False):
        i, m = _trim(ids[b].to(device), mask[b].to(device))
        e = model.embed(i, m)
        logits.append(model.head(e).cpu())
        embs.append(e.cpu())
    return torch.cat(logits), torch.cat(embs)


def train(model, ids, mask, y, train_idx, val_idx, device, lr=5e-4, weight_decay=0.01, batch_size=64,
          max_epochs=50, patience=5, warmup_epochs=1, seed=0):
    """Train on train_idx; keep the weights with the best validation accuracy (test labels are never used).

    AdamW with linear warm-up over warmup_epochs, then constant lr. Stops when validation accuracy has not
    improved for `patience` epochs.
    """
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    steps_per_epoch = -(-len(train_idx) // batch_size)
    warmup = max(1, warmup_epochs * steps_per_epoch)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: min(1.0, (s + 1) / warmup))
    gen = torch.Generator().manual_seed(seed)
    y_val = y[val_idx]
    best_val, best_state, best_epoch, history = -1.0, None, 0, []
    for epoch in range(max_epochs):
        model.train()
        total = 0.0
        for b in _batches(train_idx, batch_size, shuffle=True, generator=gen):
            i, m = _trim(ids[b].to(device), mask[b].to(device))
            loss = F.cross_entropy(model(i, m), y[b].to(device))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total += float(loss) * len(b)

        logits, _ = predict(model, ids[val_idx], mask[val_idx], device)
        val_acc = float((logits.argmax(1) == y_val).float().mean())
        history.append({"epoch": epoch, "loss": total / len(train_idx), "val_acc": val_acc})

        if val_acc > best_val:
            best_val, best_state, best_epoch = val_acc, copy.deepcopy(model.state_dict()), epoch
        elif epoch - best_epoch >= patience:
            break

    model.load_state_dict(best_state)
    return model, history
