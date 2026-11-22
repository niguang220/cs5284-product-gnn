"""Train one g-arch variant: small grid on val (GNN seed 0), then GNN seeds 0, 1, 2 with the selected config.

Usage: python exp/g-arch/run.py VARIANT [--emb ft-gte-large_s0] [--device cuda:0] [--grid-only-first]
Saves outputs/probs/garch-{VARIANT}@{emb}_{normal,cold}_s{seed}.npy (input to scripts/run_fusion.py) and appends
per-seed metrics (val / test / cold / hop buckets / train-rare recall) to exp/g-arch/results.jsonl.
Selection uses val accuracy only; test labels are read only for the logged metrics.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))
from src.data import load_products  # noqa: E402
from src.models import GCN, MLP, SAGE  # noqa: E402
from src.train import cold_start_edges, masked_metrics, set_seed, train  # noqa: E402
import encodings_ga as enc  # noqa: E402
from models import APPNPNet, GAT, GCNII, GPRGNN, GTPE, JKNet, NAGphormer, NAGphormerKNN, SGFormerGT  # noqa: E402

C, H = 47, 256
RARE = {12: "Electronics", 18: "Office", 21: "Tools", 16: "Automotive"}
GT_BASE = {"lr": 1e-3, "dropout": 0.5}


def g(**axes):
    """Cartesian grid from keyword lists."""
    out = [{}]
    for k, vs in axes.items():
        out = [{**o, k: v} for o in out for v in vs]
    return out


# variant: (builder(cfg, in_dim, pe_dim), grid, encodings)
V = {
    "MLP": (lambda c, d, p: MLP(d, H, C, c["dropout"]), g(lr=[1e-3, 1e-2], dropout=[0.2, 0.5]), []),
    "GCN": (lambda c, d, p: GCN(d, H, C, c["dropout"]), g(lr=[1e-3, 1e-2], dropout=[0.2, 0.5]), []),
    "SAGE": (lambda c, d, p: SAGE(d, H, C, c["dropout"]), g(lr=[1e-3, 1e-2], dropout=[0.2, 0.5]), []),
    "GAT": (lambda c, d, p: GAT(d, H, C, c["dropout"]), g(lr=[1e-3, 1e-2], dropout=[0.2, 0.5]), []),
    "GATv2": (lambda c, d, p: GAT(d, H, C, c["dropout"], v2=True), g(lr=[1e-3, 1e-2], dropout=[0.2, 0.5]), []),
    "GT": (lambda c, d, p: GTPE(d, H, C, c["dropout"]), g(lr=[1e-3, 5e-3], dropout=[0.2, 0.5]), []),
    "APPNP": (lambda c, d, p: APPNPNet(d, H, C, c["dropout"], K=10, alpha=c["alpha"]),
              g(lr=[1e-3, 1e-2], alpha=[0.1, 0.3], dropout=[0.5]), []),
    "GPRGNN": (lambda c, d, p: GPRGNN(d, H, C, c["dropout"], K=10, alpha=c["alpha"], dprate=c["dprate"]),
               g(lr=[1e-3, 1e-2], alpha=[0.1, 0.5], dropout=[0.5], dprate=[0.5]), []),
    "GCNII": (lambda c, d, p: GCNII(d, H, C, c["dropout"], layers=c["layers"]),
              g(lr=[1e-3, 5e-3], layers=[4, 16], dropout=[0.5]), []),
    "JKNet": (lambda c, d, p: JKNet(d, H, C, c["dropout"], layers=c["layers"], mode=c["mode"]),
              g(lr=[1e-3, 1e-2], layers=[3], mode=["cat", "max"], dropout=[0.5]), []),
    "GT-DropEdge": (lambda c, d, p: GTPE(d, H, C, GT_BASE["dropout"], edge_drop=c["edge_drop"]),
                    g(lr=[GT_BASE["lr"]], edge_drop=[0.2, 0.5]), []),
    "GT-LapPE": (lambda c, d, p: GTPE(d, H, C, GT_BASE["dropout"], pe_dim=p, sign_flip=True),
                 g(lr=[GT_BASE["lr"]], dim=[8, 16]), ["lap"]),
    "GT-RWPE": (lambda c, d, p: GTPE(d, H, C, GT_BASE["dropout"], pe_dim=p),
                g(lr=[GT_BASE["lr"]], dim=[8, 16]), ["rw"]),
    "GT-Deg": (lambda c, d, p: GTPE(d, H, C, GT_BASE["dropout"], pe_dim=p), g(lr=[GT_BASE["lr"]]), ["deg"]),
    "GT-Dist": (lambda c, d, p: GTPE(d, H, C, GT_BASE["dropout"], pe_dim=p), g(lr=[GT_BASE["lr"]]), ["dist"]),
    "NAGphormer": (lambda c, d, p: NAGphormer(d, H, C, c["dropout"], K=c["K"]),
                   g(lr=[1e-3], K=[3, 6], dropout=[0.2, 0.5]), []),
    "NAGphormer-x": (lambda c, d, p: NAGphormer(d, H, C, c["dropout"], K=c["K"], layers=c["layers"]),
                     g(lr=[1e-3], K=[10], layers=[1, 2], dropout=[0.5]), []),
    # As in the NAGphormer paper: Laplacian PE concatenated to X before hop2token (no sign flips there).
    "NAGphormer-LapPE": (lambda c, d, p: NAGphormer(d, H, C, c["dropout"], K=c["K"]),
                         g(lr=[1e-3], K=[6], dropout=[0.5], dim=[8, 16]), ["lap"]),
    "NAGphormer-kNN": (lambda c, d, p: NAGphormerKNN(d, H, C, KNN["ei"], c["dropout"], K=c["K"], K2=c["K2"]),
                       g(lr=[1e-3], K=[6], K2=[2, 4], dropout=[0.5]), []),
    "NAGphormer-noTF": (lambda c, d, p: NAGphormer(d, H, C, c["dropout"], K=c["K"], layers=0),
                        g(lr=[1e-3], K=[6], dropout=[0.5]), []),
    # Cold-start variants, selected on the val-cold proxy (--select valcold).
    "NAGphormer-iso": (lambda c, d, p: NAGphormer(d, H, C, 0.5, K=6, iso=c["iso"]), g(lr=[1e-3], iso=[0.0, 0.2, 0.5]), []),
    "NAGphormer-kNN-iso": (lambda c, d, p: NAGphormerKNN(d, H, C, KNN["ei"], 0.5, K=6, K2=2, iso=c["iso"]),
                           g(lr=[1e-3], iso=[0.0, 0.2, 0.5]), []),
    "SGFormer-GT": (lambda c, d, p: SGFormerGT(d, H, C, GT_BASE["dropout"], graph_weight=c["w"]),
                    g(lr=[GT_BASE["lr"]], w=[0.5, 0.8]), []),
    "GCNII-b": (lambda c, d, p: GCNII(d, H, C, c["dropout"], layers=c["layers"], alpha=c["alpha"]),
                g(lr=[1e-2], layers=[8], alpha=[0.1, 0.5], dropout=[0.2]), []),
    "GT-RWPE+Deg+Dist": (lambda c, d, p: GTPE(d, H, C, GT_BASE["dropout"], pe_dim=p),
                         g(lr=[GT_BASE["lr"]], dim=[16]), ["rw", "deg", "dist"]),
}


KNN = {}


def knn_graph(n, k=5):
    """Symmetrised k-nearest-neighbour graph by cosine on the label-free MiniLM embeddings (as in run_fusion.py)."""
    from src.improve import nearest_in_pool
    x = torch.nn.functional.normalize(torch.from_numpy(
        np.load(REPO / "outputs/embeddings/original__all-MiniLM-L6-v2.npy")).float(), dim=1)
    e = nearest_in_pool(torch.arange(n), x, torch.arange(n), k)
    return torch.cat([e, e.flip(0)], 1).unique(dim=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("variant")
    ap.add_argument("--emb", default="ft-gte-large_s0")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--select", default="val", choices=["val", "valcold"],
                    help="valcold: val acc with every edge touching a val or test node removed (cold-start proxy)")
    ap.add_argument("--cfg", default=None, help="json config: skip the grid (used to confirm on other encoders)")
    args = ap.parse_args()
    torch.set_num_threads(16)
    dev = torch.device(args.device)
    build, grid, encs = V[args.variant]
    if args.cfg:
        grid = [json.loads(args.cfg)]

    graph, _ = load_products()
    n = graph.num_nodes
    x0 = torch.from_numpy(np.load(REPO / "outputs" / "embeddings" / f"{args.emb}.npy")).float()
    ei = {"normal": graph.edge_index, "cold": cold_start_edges(graph)}
    vt = graph.val_mask | graph.test_mask
    ei["valcold"] = graph.edge_index[:, ~(vt[graph.edge_index[0]] | vt[graph.edge_index[1]])]
    _, dist = enc.train_distance(graph.edge_index, n, graph.train_mask.numpy())
    t = graph.test_mask.numpy()
    hop = {"1": t & (dist == 1), "2": t & (dist == 2), "3": t & (dist == 3), "4+": t & (dist >= 4)}
    y = graph.y
    if "kNN" in args.variant:
        KNN["ei"] = knn_graph(n).to(dev)
    graph = graph.to(dev)
    ei = {k: v.to(dev) for k, v in ei.items()}

    def feats(cfg, setting):
        if not encs:
            return x0.to(dev), 0
        parts = [enc.get(e, ei[setting].cpu(), n, setting, graph.train_mask.cpu(), cfg.get("dim", 16), dev)
                 for e in encs]
        pe = torch.cat(parts, 1)
        return torch.cat([x0, pe], 1).to(dev), pe.size(1)

    def fit(cfg, seed):
        set_seed(seed)
        xn, pd_ = feats(cfg, "normal")
        model = build(cfg, xn.size(1), pd_).to(dev)
        t0 = time.time()
        model, hist = train(model, xn, graph, lr=cfg["lr"])
        model.eval()
        probs = {}
        with torch.no_grad():
            for s in ["normal", "cold", "valcold"]:
                if s == "valcold" and encs:  # encodings are only cached for normal / cold
                    continue
                probs[s] = model(feats(cfg, s)[0], ei[s]).softmax(1).cpu()
        torch.cuda.empty_cache()
        return probs, max(h["val_acc"] for h in hist), len(hist), (time.time() - t0) / 60

    sweep = []
    for cfg in grid:
        probs, va, ep, mins = fit(cfg, 0)
        vc = float((probs["valcold"].argmax(1) == y)[graph.val_mask.cpu()].float().mean()) if "valcold" in probs else None
        sweep.append({"cfg": cfg, "val_acc": va, "valcold_acc": vc, "epochs": ep, "minutes": round(mins, 2),
                      "probs": probs})
        print(args.variant, "grid", cfg, f"val {va:.4f}", f"test {float((probs['normal'].argmax(1) == y)[graph.test_mask.cpu()].float().mean()):.4f}",
              f"{ep} ep {mins:.1f} min", flush=True)
    best = max(sweep, key=lambda r: r["val_acc"] if args.select == "val" else r["valcold_acc"])
    cfg = best["cfg"]
    tag = f"garch-{args.variant}@{args.emb}"
    rows = []
    for seed in [0, 1, 2]:
        if seed == 0:
            probs, va, ep, mins = best["probs"], best["val_acc"], best["epochs"], best["minutes"]
        else:
            probs, va, ep, mins = fit(cfg, seed)
        r = {"variant": args.variant, "emb": args.emb, "seed": seed, "cfg": cfg, "epochs": ep, "minutes": mins}
        if "valcold" in probs:
            r["valcold_acc"] = float((probs["valcold"].argmax(1) == y)[graph.val_mask.cpu()].float().mean())
        for s, p in probs.items():
            if s == "valcold":
                continue
            np.save(REPO / "outputs" / "probs" / f"{tag}_{s}_s{seed}.npy", p.numpy().astype(np.float32))
            pred = p.argmax(1)
            pre = "" if s == "normal" else "cold_"
            if s == "normal":
                r["val_acc"] = masked_metrics(y, pred, graph.val_mask.cpu())["acc"]
            m = masked_metrics(y, pred, graph.test_mask.cpu())
            r[pre + "test_acc"], r[pre + "test_f1"] = m["acc"], m["macro_f1"]
            for k, msk in hop.items():
                r[f"{pre}hop{k}"] = float((pred[msk] == y[msk]).float().mean())
            for c, name in RARE.items():
                msk = graph.test_mask.cpu() & (y == c)
                r[f"{pre}rec_{name}"] = float((pred[msk] == c).float().mean())
        rows.append(r)
        print(tag, seed, {k: round(v, 4) for k, v in r.items() if isinstance(v, float)}, flush=True)
    with open(HERE / "results.jsonl", "a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    meta = {"variant": args.variant, "emb": args.emb, "selected": cfg, "selection": f"{args.select} acc, GNN seed 0",
            "grid": [{k: v for k, v in s.items() if k != "probs"} for s in sweep], "encodings": encs,
            "hidden": H, "weight_decay": 5e-4, "max_epochs": 500, "patience": 50, "threads": 16}
    (REPO / "outputs" / "probs" / f"{tag}.json").write_text(json.dumps(meta, indent=1))
    with open(HERE / "grid.jsonl", "a") as f:
        f.write(json.dumps(meta) + "\n")


if __name__ == "__main__":
    main()
