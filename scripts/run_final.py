"""Final post-hoc pipeline on saved base-model probabilities: LLM fusion + PPR local prior + belief propagation over
co-purchase, text-kNN and sibling edges.

Usage: python scripts/run_final.py TAG [TAG ...]
  TAG names outputs/probs/{TAG}_{normal|cold}_s{seed}.npy (3 GNN seeds), e.g. NAGphormer@ft-gte-large-r_s0.
Per TAG and setting (normal / cold start):
  1. temperature on val NLL; fuse with the LLM answers through their confusion matrices (weight 1, Bayes' rule);
  2. add the PPR local prior log π_i/π_tr with λ = 1 (the full Saerens correction; val prefers weaker corrections
     because val has the train label mix, see docs/method.md);
  3. add β · co-purchase BP messages, γ · text-kNN messages and g2 · degree-normalised sibling messages.
Chosen on val accuracy (mean over the GNN seeds): the LLM combination, the PPR setting (a, iterations), β, γ, g2.
Test labels are not used for any choice made here. Cold start uses no co-purchase edge touching a test node.
Writes results/4_fusion/final_{runs,summary}.csv, results/4_fusion/final_meta.json and the selected posteriors to
outputs/final/{TAG}_{setting}_s{seed}.npy.
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from src.data import load_products  # noqa: E402
from src.improve import nearest_in_pool  # noqa: E402
from src.label_shift import (K, compatibility, confusion, fit_temperature, neighbour_log_messages,  # noqa: E402
                             ppr_log_prior, sibling_log_messages)
from src.train import cold_start_edges, masked_metrics  # noqa: E402

PPR = [(0.15, 1), (0.15, 3), (0.5, 1), (0.5, 3)]
BETAS, GAMMAS, G2S = [0.0, 0.25, 0.5, 1.0], [0.0, 0.25, 0.5, 1.0], [0.0, 0.5, 1.0]
LAM, KNN = 1.0, 5

graph, _ = load_products()
y, N = graph.y.numpy(), graph.num_nodes
tr, va, te = graph.train_mask.numpy(), graph.val_mask.numpy(), graph.test_mask.numpy()
yt = torch.from_numpy(y)

llm = {"gpt": np.array([int(r[0]) for r in csv.reader(open(REPO / "outputs/tape_gpt_preds/ogbn-products.csv"))])}
for name, path in [("qwen", REPO / "outputs/llm/qwen35_9b.npy")] + [
        (p.stem, p) for p in sorted((REPO / "outputs/llm_answers").glob("*.npy"))]:
    if path.exists():
        llm[name] = np.load(path)
C = {k: confusion(a, y, tr | va) for k, a in llm.items()}
# GPT alone, or GPT plus one other LLM: several answer sets from the same LLM are not independent observations.
combos = [("gpt",)] + [("gpt", k) for k in llm if k != "gpt"]


def evidence(combo):
    """What a node tells its neighbours about its own label: its train label, else its LLM answers' likelihood."""
    e = np.ones((N, K))
    for k in combo:
        e *= C[k][llm[k] % (K + 1)]
    e /= e.sum(1, keepdims=True)
    e[tr] = np.eye(K)[y[tr]]  # val labels are never used as evidence
    return e


def adjacency(ei):
    A = sp.coo_matrix((np.ones(ei.shape[1]), (ei[1], ei[0])), shape=(N, N)).tocsr()
    A = A + A.T
    A.data[:] = 1
    A.setdiag(0)
    A.eliminate_zeros()
    return A


edges = {"normal": graph.edge_index.numpy(), "cold": cold_start_edges(graph).numpy()}
x = torch.nn.functional.normalize(torch.from_numpy(
    np.load(REPO / "outputs/embeddings/original__all-MiniLM-L6-v2.npy")).float(), dim=1)
knn = nearest_in_pool(torch.arange(N), x, torch.arange(N), KNN).numpy()  # text-kNN edges j -> i, kept in cold start
H, H_knn = compatibility(edges["normal"], y, tr), compatibility(knn, y, tr)
ppr_graph = {s: ((adjacency(e) + adjacency(knn)) > 0).astype(np.float64).tocsr() for s, e in edges.items()}
EV = {c: evidence(c) for c in combos}
LLMLOG = {c: sum(np.log(C[k][llm[k] % (K + 1)]) for k in c) for c in combos}
MSG = {(s, c): neighbour_log_messages(edges[s], N, EV[c], H) for s in edges for c in combos}
KMSG = {c: neighbour_log_messages(knn, N, EV[c], H_knn) for c in combos}
SMSG = {(s, c): sibling_log_messages(edges[s], N, EV[c], y, tr) for s in edges for c in combos}
LAB = np.unique(y[te])


def log_softmax(z):
    z = z - z.max(1, keepdims=True)
    return z - np.log(np.exp(z).sum(1, keepdims=True))


def quick(pred):
    """Val / test accuracy and test macro-F1 (over the classes present in test, as masked_metrics)."""
    pt, yt_ = pred[te], y[te]
    tp = np.bincount(yt_[pt == yt_], minlength=K)[LAB]
    f1 = (2 * tp / np.maximum(np.bincount(pt, minlength=K)[LAB] + np.bincount(yt_, minlength=K)[LAB], 1)).mean()
    return {"val_acc": (pred[va] == y[va]).mean(), "test_acc": (pt == yt_).mean(), "test_f1": f1}


def posterior(p, setting, combo, ppr, beta, gamma, g2):
    """The pipeline's log posterior for one configuration."""
    lp = log_softmax(np.log(np.clip(p, 1e-12, 1)) / fit_temperature(p, y, va))
    fused = log_softmax(lp + LLMLOG[combo])
    z = fused + LAM * ppr_log_prior(fused, ppr_graph[setting], y, tr, *ppr)
    return log_softmax(z + beta * MSG[(setting, combo)] + gamma * KMSG[combo] + g2 * SMSG[(setting, combo)])


if __name__ == "__main__":
    tags = sys.argv[1:]
    out_dir = REPO / "outputs/final"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, sel = [], []
    for tag in tags:
        for setting in ["normal", "cold"]:
            probs = [np.load(REPO / f"outputs/probs/{tag}_{setting}_s{seed}.npy").astype(np.float64) for seed in range(3)]
            for seed, p in enumerate(probs):
                lp = log_softmax(np.log(np.clip(p, 1e-12, 1)) / fit_temperature(p, y, va))
                for combo in combos:
                    fused = log_softmax(lp + LLMLOG[combo])
                    for ppr in PPR:
                        z0 = fused + LAM * ppr_log_prior(fused, ppr_graph[setting], y, tr, *ppr)
                        for beta in BETAS:
                            for gamma in GAMMAS:
                                z1 = z0 + beta * MSG[(setting, combo)] + gamma * KMSG[combo]
                                for g2 in G2S:
                                    pred = (z1 + g2 * SMSG[(setting, combo)]).argmax(1)
                                    rows.append({"model": tag, "setting": setting, "seed": seed, "llms": "+".join(combo),
                                                 "a": ppr[0], "iters": ppr[1], "beta": beta, "gamma": gamma, "g2": g2,
                                                 **quick(pred)})
            df = pd.DataFrame([r for r in rows if r["model"] == tag and r["setting"] == setting])
            key = ["llms", "a", "iters", "beta", "gamma", "g2"]
            best = df.groupby(key)["val_acc"].mean().idxmax()
            cfg = dict(zip(key, best))
            combo = tuple(cfg["llms"].split("+"))
            for seed, p in enumerate(probs):
                z = posterior(p, setting, combo, (cfg["a"], cfg["iters"]), cfg["beta"], cfg["gamma"], cfg["g2"])
                np.save(out_dir / f"{tag}_{setting}_s{seed}.npy", np.exp(z).astype(np.float32))
                pred = torch.from_numpy(z.argmax(1))
                v, t = masked_metrics(yt, pred, torch.from_numpy(va)), masked_metrics(yt, pred, torch.from_numpy(te))
                sel.append({"model": tag, "setting": setting, "seed": seed, **cfg, "val_acc": v["acc"],
                            "test_acc": t["acc"], "test_f1": t["macro_f1"]})
            print(pd.DataFrame(sel[-3:]).round(4).to_string(index=False), flush=True)
    stem = "final" if len(tags) > 1 else f"final_{tags[0]}"
    pd.DataFrame(rows).to_csv(REPO / f"results/4_fusion/{stem}_runs.csv", index=False)
    s = pd.DataFrame(sel)
    s.to_csv(REPO / f"results/4_fusion/{stem}_selected.csv", index=False)
    summ = s.groupby(["model", "setting"])[["val_acc", "test_acc", "test_f1"]].agg(["mean", "std"]).round(4)
    summ.to_csv(REPO / f"results/4_fusion/{stem}_summary.csv")
    with pd.option_context("display.width", 200):
        print(summ)
    (REPO / f"results/4_fusion/{stem}_meta.json").write_text(json.dumps(
        {"llms": list(llm), "lambda": LAM, "ppr": PPR, "betas": BETAS, "gammas": GAMMAS, "g2s": G2S, "knn": KNN,
         "selection": "val accuracy, mean over GNN seeds, per model and setting",
         "knn_emb": "original__all-MiniLM-L6-v2"}, indent=1))
