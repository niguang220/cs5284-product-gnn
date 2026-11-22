"""Graph variants on MiniLM features, 3 seeds, early stopping on val accuracy (test labels never used).

  --model SAGE | GT | SIGN       SIGN: precomputed [X, AX, ..., A^K X] (and label hops) -> per-hop linear -> MLP
  --label-trick                  UniMP / ogbn-products "label reuse": each epoch half the train nodes give their
                                 one-hot label as input, loss on the other half; at inference all train labels are input
  --knn K --knn-mode all|low     add symmetric text-kNN edges (cosine on MiniLM) to every node / to nodes of degree <= 1
  --adj sym|ego                  SIGN operator: D^-1/2 (A+I) D^-1/2, or ego/neighbour separation (D^-1 A, no self loop,
                                 H2GCN-style) so a node's own text is never averaged with heterophilous neighbours
Cold start: co-purchase edges touching test nodes are removed at inference; text-kNN edges are kept (text is known).
"""
import argparse
import copy
import json

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.utils import add_self_loops, degree, remove_self_loops, to_undirected

from common import C, OUT, cold_start_edges, hop_buckets, hops_to_train, load, log_rows, metrics, set_seed
from src.models import GraphTransformer

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="SAGE")
ap.add_argument("--label-trick", action="store_true")
ap.add_argument("--knn", type=int, default=0)
ap.add_argument("--knn-mode", default="all")
ap.add_argument("--hops", type=int, default=3)
ap.add_argument("--adj", default="sym")
ap.add_argument("--hidden", type=int, default=256)
ap.add_argument("--layers", type=int, default=2)
ap.add_argument("--isolate", type=float, default=0.0,
                help="cold-start augmentation: each epoch drop all co-purchase edges of this fraction of nodes")
ap.add_argument("--dropout", type=float, default=0.5)
ap.add_argument("--lr", type=float, default=None)
ap.add_argument("--wd", type=float, default=5e-4)
ap.add_argument("--epochs", type=int, default=500)
ap.add_argument("--patience", type=int, default=50)
ap.add_argument("--device", default="cuda:4")
ap.add_argument("--seeds", default="0,1,2")
ap.add_argument("--name", required=True)
ap.add_argument("--save", action="store_true")
args = ap.parse_args()
torch.set_num_threads(16)
dev = torch.device(args.device)

graph, x = load()
N = graph.num_nodes
dist = hops_to_train(graph)
buckets = hop_buckets(graph, dist)
ei_normal, ei_cold = graph.edge_index, cold_start_edges(graph)
# Val proxy for cold start: also drop edges touching val nodes, then score val (no test labels involved).
vt = graph.val_mask | graph.test_mask
ei_vcold = graph.edge_index[:, ~(vt[graph.edge_index[0]] | vt[graph.edge_index[1]])]


def knn_edges(k, mode, base_ei):
    """Symmetric edges from each selected node to its k most similar nodes (cosine on MiniLM)."""
    xn = F.normalize(x, dim=1).to(dev)
    deg = degree(base_ei[0], N)
    nodes = torch.arange(N) if mode == "all" else (deg <= 1).nonzero().flatten()
    out = []
    for s in range(0, len(nodes), 2048):
        q = nodes[s:s + 2048].to(dev)
        sim = xn[q] @ xn.T
        sim[torch.arange(len(q), device=dev), q] = -2
        top = sim.topk(k, dim=1).indices
        out.append(torch.stack([q.repeat_interleave(k), top.flatten()]).cpu())
    return torch.cat(out, 1)


ei_cp, ei_knn = graph.edge_index, torch.empty(2, 0, dtype=torch.long)
if args.knn:
    ei_knn = knn_edges(args.knn, args.knn_mode, ei_normal)
    # Low-degree selection is by co-purchase degree of the setting at hand (cold: test nodes have degree 0).
    ei_normal = to_undirected(torch.cat([ei_normal, ei_knn], 1), num_nodes=N)
    ei_cold = to_undirected(torch.cat([ei_cold, knn_edges(args.knn, args.knn_mode, ei_cold)], 1), num_nodes=N)
    ei_vcold = to_undirected(torch.cat([ei_vcold, knn_edges(args.knn, args.knn_mode, ei_vcold)], 1), num_nodes=N)

y = graph.y.to(dev)
train_idx = graph.train_mask.nonzero().flatten().to(dev)
val_mask, x = graph.val_mask.to(dev), x.to(dev)
ei_normal, ei_cold, ei_vcold = ei_normal.to(dev), ei_cold.to(dev), ei_vcold.to(dev)
ei_cp, ei_knn = ei_cp.to(dev), ei_knn.to(dev)
ei_train = None  # set per epoch when --isolate
LT = C if args.label_trick else 0


def operator(ei):
    if args.adj == "sym":
        ei2, _ = add_self_loops(remove_self_loops(ei)[0], num_nodes=N)
        d = degree(ei2[0], N).pow(-0.5)
        w = d[ei2[0]] * d[ei2[1]]
    else:
        ei2 = remove_self_loops(ei)[0]
        w = degree(ei2[1], N).clamp(min=1).pow(-1)[ei2[1]]
    return torch.sparse_coo_tensor(torch.stack([ei2[1], ei2[0]]), w, (N, N)).coalesce()


def hops(A, h, K):
    out = [h]
    for _ in range(K):
        out.append(torch.sparse.mm(A, out[-1]))
    return out


class SIGN(torch.nn.Module):
    def __init__(self, in_dim, n_feat_hops, n_label_hops, hidden, dropout):
        super().__init__()
        self.proj = torch.nn.ModuleList(torch.nn.Linear(in_dim, hidden) for _ in range(n_feat_hops))
        self.lproj = torch.nn.ModuleList(torch.nn.Linear(C, hidden) for _ in range(n_label_hops))
        self.out = torch.nn.Sequential(torch.nn.PReLU(), torch.nn.Dropout(dropout),
                                       torch.nn.Linear(hidden * (n_feat_hops + n_label_hops), hidden),
                                       torch.nn.ReLU(), torch.nn.Dropout(dropout), torch.nn.Linear(hidden, C))
        self.dropout = dropout

    def forward(self, feats, lfeats):
        hs = [p(F.dropout(f, self.dropout, self.training)) for p, f in zip(self.proj, feats)]
        hs += [p(f) for p, f in zip(self.lproj, lfeats)]
        return self.out(torch.cat(hs, 1))


class SAGE2(torch.nn.Module):
    def __init__(self, in_dim, hidden, dropout):
        super().__init__()
        self.c1, self.c2, self.dropout = SAGEConv(in_dim, hidden), SAGEConv(hidden, C), dropout
        for c in (self.c1, self.c2):
            glorot(c.lin_l.weight); zeros(c.lin_l.bias); glorot(c.lin_r.weight)

    def forward(self, h, ei):
        h = F.dropout(h, self.dropout, self.training)
        h = F.dropout(self.c1(h, ei).relu(), self.dropout, self.training)
        return self.c2(h, ei)


ops = {}
if args.model == "SIGN":
    for st, ei in [("normal", ei_normal), ("cold", ei_cold), ("vcold", ei_vcold)]:
        A = operator(ei)
        ops[st] = (A, hops(A, x, args.hops))


def label_input(mask_idx):
    yin = torch.zeros(N, C, device=dev)
    yin[mask_idx] = F.one_hot(y[mask_idx], C).float()
    return yin


def forward(model, setting, yin):
    ei = {"normal": ei_normal, "cold": ei_cold, "vcold": ei_vcold, "train": ei_train}[setting]
    if args.model == "SIGN":
        A, feats = ops[setting]
        lfeats = hops(A, yin, args.hops)[1:] if LT else []  # hop 0 would be the node's own (masked) label
        return model(feats, lfeats)
    h = torch.cat([x, yin], 1) if LT else x
    return model(h, ei)


def build():
    if args.model == "SIGN":
        return SIGN(x.size(1), args.hops + 1, args.hops if LT else 0, args.hidden, args.dropout), args.lr or 1e-3
    if args.model == "GT":
        return GraphTransformer(x.size(1) + LT, args.hidden, C, dropout=args.dropout, num_layers=args.layers, heads=8), args.lr or 1e-3
    return SAGE2(x.size(1) + LT, args.hidden, args.dropout), args.lr or 1e-2


rows, all_train = [], label_input(train_idx) if LT else None
for seed in [int(s) for s in args.seeds.split(",")]:
    set_seed(seed)
    model, lr = build()
    model = model.to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=args.wd)
    best, best_state, best_ep = -1, None, 0
    for ep in range(args.epochs):
        model.train()
        opt.zero_grad()
        setting = "normal"
        if args.isolate > 0:
            iso = torch.rand(N, device=dev) < args.isolate
            keep = ~(iso[ei_cp[0]] | iso[ei_cp[1]])
            ei_train = to_undirected(torch.cat([ei_cp[:, keep], ei_knn], 1), num_nodes=N)
            setting = "train"
        if LT:
            perm = train_idx[torch.randperm(len(train_idx), device=dev)]
            inp, tgt = perm[: len(perm) // 2], perm[len(perm) // 2:]
            out = forward(model, setting, label_input(inp))
        else:
            tgt, out = train_idx, forward(model, setting, None)
        F.cross_entropy(out[tgt], y[tgt]).backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            va = float((forward(model, "normal", all_train).argmax(1)[val_mask] == y[val_mask]).float().mean())
        if va > best:
            best, best_state, best_ep = va, copy.deepcopy(model.state_dict()), ep
        elif ep - best_ep >= args.patience:
            break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pn = forward(model, "normal", all_train).softmax(1).cpu()
        pc = forward(model, "cold", all_train).softmax(1).cpu()
        pvp = forward(model, "vcold", all_train).softmax(1).cpu()
        pv = pvp.argmax(1)
    m = metrics(graph, pn, pc)
    m["val_cold_acc"] = float((pv[graph.val_mask] == graph.y[graph.val_mask]).float().mean())
    m.update({f"hop{k}": float((pn.argmax(1)[b] == graph.y[b]).float().mean()) for k, b in buckets.items()})
    m.update({f"cold_hop{k}": float((pc.argmax(1)[b] == graph.y[b]).float().mean()) for k, b in buckets.items()})
    rows.append({"variant": args.name, "seed": seed, "best_epoch": best_ep, **m, "args": vars(args)})
    print(args.name, seed, {k: round(v, 4) for k, v in m.items()}, flush=True)
    if args.save:
        np.save(OUT / f"{args.name}_normal_s{seed}.npy", pn.numpy().astype(np.float32))
        np.save(OUT / f"{args.name}_cold_s{seed}.npy", pc.numpy().astype(np.float32))
        np.save(OUT / f"{args.name}_vcold_s{seed}.npy", pvp.numpy().astype(np.float32))
log_rows("run", rows)
keys = [k for k in rows[0] if k not in ("variant", "seed", "args")]
print("MEAN", args.name, {k: f"{np.mean([r[k] for r in rows]):.4f}±{np.std([r[k] for r in rows]):.4f}" for k in keys}, flush=True)
