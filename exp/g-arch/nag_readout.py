"""Which hops does NAGphormer use? Retrain the selected NAGphormer (ft-gte-large_s0, GNN seed 0) and average the
per-node readout weights alpha_k over hop tokens 1..K, by node group (train / val / test by hop distance to the
nearest train node), on the full graph. Writes exp/g-arch/nag_readout.csv."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))
from src.data import load_products  # noqa: E402
from src.train import set_seed, train  # noqa: E402
import encodings_ga as enc  # noqa: E402
from models import NAGphormer  # noqa: E402

torch.set_num_threads(16)
dev = torch.device(sys.argv[1] if len(sys.argv) > 1 else "cuda:0")
graph, _ = load_products()
_, dist = enc.train_distance(graph.edge_index, graph.num_nodes, graph.train_mask.numpy())
x = torch.from_numpy(np.load(REPO / "outputs/embeddings/ft-gte-large_s0.npy")).float().to(dev)
graph = graph.to(dev)
set_seed(0)
model, _ = train(NAGphormer(x.size(1), 256, 47, 0.5, K=6).to(dev), x, graph, lr=1e-3)
model.eval()
with torch.no_grad():
    t = model.tokens(x, graph.edge_index)
    h = model.norm(model.enc(model.inp(t)))
    node, hops = h[:, :1], h[:, 1:]
    a = torch.softmax(model.read(torch.cat([node.expand_as(hops), hops], -1)), dim=1).squeeze(-1).cpu().numpy()
    # How much the node's own token counts relative to the hop mixture (z = node + sum_k a_k hop_k).
    ratio = (node.squeeze(1).norm(dim=1) / (a[..., None] * hops.cpu().numpy()).sum(1).__abs__().sum(1).clip(1e-9)
             if False else None)
te, tr, va = (m.cpu().numpy() for m in (graph.test_mask, graph.train_mask, graph.val_mask))
groups = {"train": tr, "val": va, "test hop1": te & (dist == 1), "test hop2": te & (dist == 2),
          "test hop3": te & (dist == 3), "test hop4+": te & (dist >= 4)}
rows = []
for g, m in groups.items():
    w = a[m].mean(0)
    rows.append({"group": g, "n": int(m.sum()), **{f"a{k + 1}": round(float(v), 3) for k, v in enumerate(w)},
                 "mean_hop": round(float((a[m] * np.arange(1, a.shape[1] + 1)).sum(1).mean()), 3)})
df = pd.DataFrame(rows)
df.to_csv(HERE / "nag_readout.csv", index=False)
print(df.to_string(index=False))
