"""Save the current best stack's posterior (ft-gte-large + GT, fuse GPT, BP β=0.25; mean over 3 GNN seeds).

Writes outputs/stack/best_{normal,cold}.npy ([N, 47] float32). Every choice was made on val (results/3_gnn_text/fusion_GT@ft-gte-large_s0_*).
"""
import csv, sys
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(REPO))
from src.data import load_products
from src.label_shift import K, compatibility, confusion, fit_temperature, fuse, neighbour_log_messages, temper
from src.train import cold_start_edges
g, _ = load_products(); y, N = g.y.numpy(), g.num_nodes
tr, va, te = g.train_mask.numpy(), g.val_mask.numpy(), g.test_mask.numpy()
gpt = np.array([int(r[0]) for r in csv.reader(open(REPO / "outputs/tape_gpt_preds/ogbn-products.csv"))])
C = confusion(gpt, y, tr | va)
e = C[gpt % (K + 1)]; e = e / e.sum(1, keepdims=True); e[tr] = np.eye(K)[y[tr]]
H = compatibility(g.edge_index.numpy(), y, tr)
out = REPO / "outputs/stack"; out.mkdir(exist_ok=True)
for setting, ei in [("normal", g.edge_index.numpy()), ("cold", cold_start_edges(g).numpy())]:
    M = neighbour_log_messages(ei, N, e, H)
    qs = []
    for s in range(3):
        p = np.load(REPO / f"outputs/probs/GT@ft-gte-large_s0_{setting}_s{s}.npy").astype(float)
        z = np.log(fuse(temper(p, fit_temperature(p, y, va)), [gpt], [C])) + 0.25 * M
        q = np.exp(z - z.max(1, keepdims=True)); qs.append(q / q.sum(1, keepdims=True))
    q = np.mean(qs, 0).astype(np.float32); np.save(out / f"best_{setting}.npy", q)
    pr = q.argmax(1); print(setting, "val", (pr[va] == y[va]).mean().round(4), "test", (pr[te] == y[te]).mean().round(4),
          "test share with max prob < 0.9:", (q[te].max(1) < 0.9).mean().round(3), "acc there", (pr[te & (q.max(1) < 0.9)] == y[te & (q.max(1) < 0.9)]).mean().round(3))
