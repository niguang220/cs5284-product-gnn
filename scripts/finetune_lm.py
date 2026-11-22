"""Fine-tune a small encoder end-to-end on the train nodes (TAPE's LM step), then embed every node.

Usage: python scripts/finetune_lm.py [--model Alibaba-NLP/gte-large-en-v1.5] [--device cuda:0] [--seed 0]
Defaults are the settings of the reported runs (gte-large, 2 epochs, lr 2e-5, 320 tokens, gradient checkpointing;
about 20 min and 12GB on an A100).
Classifier = linear head on the [CLS] vector. Early stopping / checkpoint choice on val accuracy only.
Writes outputs/embeddings/ft-<short>_s<seed>.npy ([CLS] of the best checkpoint, L2-normalised) and
results/shared/embeddings/ft-<short>_s<seed>_finetune.json (arguments and val curve). With --save-ckpt, also the best
checkpoint in outputs/ckpt/ft-<short>_s<seed>/ (encoder in bf16 via save_pretrained, tokenizer, head.pt).
"""
import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from src.data import load_products  # noqa: E402
from src.text_clean import build_original_text  # noqa: E402
from src.train import set_seed  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Alibaba-NLP/gte-large-en-v1.5")
ap.add_argument("--tag", default="")
ap.add_argument("--no-grad-ckpt", dest="grad_ckpt", action="store_false")
ap.add_argument("--device", default="cuda:0")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--max-len", type=int, default=320)
ap.add_argument("--epochs", type=int, default=2)
ap.add_argument("--batch", type=int, default=32)
ap.add_argument("--lr", type=float, default=2e-5)
ap.add_argument("--evals-per-epoch", type=int, default=2)
ap.add_argument("--save-ckpt", action="store_true")
args = ap.parse_args()
set_seed(args.seed)
torch.set_num_threads(16)
dev = torch.device(args.device)

graph, text = load_products()
texts = [build_original_text(t, c) for t, c in zip(text["title"], text["content"])]
tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
enc = tok(texts, truncation=True, max_length=args.max_len)
ids = enc["input_ids"]
y = graph.y.numpy()
train_idx = graph.train_mask.nonzero().view(-1).numpy()
val_idx = graph.val_mask.nonzero().view(-1).numpy()

lm = AutoModel.from_pretrained(args.model, trust_remote_code=True).to(dev)
if args.grad_ckpt:  # keeps a large encoder under the 12GB budget
    lm.gradient_checkpointing_enable()
head = torch.nn.Linear(lm.config.hidden_size, 47).to(dev)
params = list(lm.parameters()) + list(head.parameters())
opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
steps_per_epoch = (len(train_idx) + args.batch - 1) // args.batch
sched = get_linear_schedule_with_warmup(opt, int(0.06 * steps_per_epoch * args.epochs), steps_per_epoch * args.epochs)


def batch_tensors(idx):
    b = tok.pad({"input_ids": [ids[i] for i in idx]}, return_tensors="pt")
    return b["input_ids"].to(dev), b["attention_mask"].to(dev)


@torch.no_grad()
def embed(idx, bs=256):
    lm.eval()
    order = sorted(idx, key=lambda i: len(ids[i]))  # length-sorted batches waste less padding
    out = {}
    for s in range(0, len(order), bs):
        chunk = order[s:s + bs]
        inp, att = batch_tensors(chunk)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            h = lm(input_ids=inp, attention_mask=att).last_hidden_state[:, 0].float()
        for i, v in zip(chunk, h.cpu()):
            out[i] = v
    return torch.stack([out[i] for i in idx])


def val_acc():
    h = embed(list(val_idx)).to(dev)
    return float((head(h).argmax(1).cpu().numpy() == y[val_idx]).mean())


log, best, best_state, step = [], -1.0, None, 0
eval_every = steps_per_epoch // args.evals_per_epoch
t0 = time.time()
for ep in range(args.epochs):
    perm = np.random.permutation(train_idx)
    for s in range(0, len(perm), args.batch):
        lm.train()
        idx = perm[s:s + args.batch]
        inp, att = batch_tensors(idx)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            h = lm(input_ids=inp, attention_mask=att).last_hidden_state[:, 0]
            loss = F.cross_entropy(head(h).float(), torch.from_numpy(y[idx]).to(dev))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        step += 1
        if step % eval_every == 0:
            va = val_acc()
            log.append({"step": step, "epoch": step / steps_per_epoch, "loss": float(loss), "val_acc": va,
                        "min": round((time.time() - t0) / 60, 1)})
            print(log[-1], flush=True)
            if va > best:
                best = va
                best_state = (copy.deepcopy(lm.state_dict()), copy.deepcopy(head.state_dict()))

lm.load_state_dict(best_state[0])
head.load_state_dict(best_state[1])
emb = F.normalize(embed(list(range(len(texts)))), dim=1).numpy().astype(np.float32)
short = args.model.split("/")[-1].replace("-en-v1.5", "")
name = f"ft-{short}{args.tag}_s{args.seed}"
np.save(REPO / "outputs" / "embeddings" / f"{name}.npy", emb)
(REPO / "results" / "shared" / "embeddings" / f"{name}_finetune.json").write_text(json.dumps(
    {"args": vars(args), "best_val_acc_lm_head": best, "log": log,
     "peak_gpu_gb": round(torch.cuda.max_memory_allocated(dev) / 1e9, 2)}, indent=1))
if args.save_ckpt:
    ckpt = REPO / "outputs" / "ckpt" / name
    lm.to(torch.bfloat16).save_pretrained(ckpt)  # bf16 halves the size; the embeddings above use bf16 autocast
    tok.save_pretrained(ckpt)
    torch.save(head.state_dict(), ckpt / "head.pt")
print(name, "best val", best, flush=True)
