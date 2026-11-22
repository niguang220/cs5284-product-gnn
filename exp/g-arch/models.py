"""Model zoo for the g-arch comparison. Every model takes (x, edge_index) so it fits src/train.py: train.

All use hidden 256 and the same input dropout as src/models.py; the GT here extends src/models.py's
GraphTransformer with optional positional / structural encodings (extra columns of x) and DropEdge.
"""
import torch
import torch.nn.functional as F
from torch_geometric.nn import (APPNP, GATConv, GATv2Conv, GCN2Conv, JumpingKnowledge, Linear, SAGEConv,
                                TransformerConv)
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.utils import add_self_loops, dropout_edge, remove_self_loops


class GAT(torch.nn.Module):
    """GAT (Velickovic et al. 2018) or GATv2 (Brody et al. 2022: dynamic attention, a^T LeakyReLU(W[h_i||h_j]))."""

    def __init__(self, in_dim, hidden, C, dropout=0.5, heads=8, v2=False, attn_dropout=0.2):
        super().__init__()
        conv = GATv2Conv if v2 else GATConv
        self.c1 = conv(in_dim, hidden // heads, heads=heads, dropout=attn_dropout)
        self.c2 = conv(hidden, C, heads=1, dropout=attn_dropout)
        self.dropout = dropout

    def forward(self, x, ei):
        x = F.dropout(x, self.dropout, self.training)
        h = F.elu(self.c1(x, ei))
        h = F.dropout(h, self.dropout, self.training)
        return self.c2(h, ei)


class MLPHead(torch.nn.Module):
    def __init__(self, in_dim, hidden, C, dropout):
        super().__init__()
        self.l1, self.l2, self.dropout = Linear(in_dim, hidden), Linear(hidden, C), dropout

    def forward(self, x):
        x = F.dropout(x, self.dropout, self.training)
        h = F.dropout(self.l1(x).relu(), self.dropout, self.training)
        return self.l2(h)


class APPNPNet(torch.nn.Module):
    """APPNP (Gasteiger et al. 2019): predict with an MLP, then propagate with personalised PageRank (K steps,
    teleport alpha). Decoupling lets the label signal travel K hops without K layers of parameters."""

    def __init__(self, in_dim, hidden, C, dropout=0.5, K=10, alpha=0.1):
        super().__init__()
        self.mlp, self.prop = MLPHead(in_dim, hidden, C, dropout), APPNP(K, alpha, cached=False)

    def forward(self, x, ei):
        return self.prop(self.mlp(x), ei)


class GPRGNN(torch.nn.Module):
    """GPR-GNN (Chien et al. 2021): MLP, then sum_k gamma_k A_hat^k Z with learnable, possibly negative hop weights
    gamma (initialised to PPR weights), which can adapt to heterophily and to how far the label signal should go."""

    def __init__(self, in_dim, hidden, C, dropout=0.5, K=10, alpha=0.1, dprate=0.5):
        super().__init__()
        self.mlp, self.K, self.dprate = MLPHead(in_dim, hidden, C, dropout), K, dprate
        g = alpha * (1 - alpha) ** torch.arange(K + 1, dtype=torch.float)
        g[-1] = (1 - alpha) ** K
        self.gamma = torch.nn.Parameter(g)

    def forward(self, x, ei):
        z = self.mlp(x)
        z = F.dropout(z, self.dprate, self.training)
        ei, w = gcn_norm(ei, None, x.size(0), add_self_loops=True)
        A = torch.sparse_coo_tensor(ei, w, (x.size(0), x.size(0))).coalesce()
        out = self.gamma[0] * z
        for k in range(1, self.K + 1):
            z = torch.sparse.mm(A, z)
            out = out + self.gamma[k] * z
        return out


class GCNII(torch.nn.Module):
    """GCNII (Chen et al. 2020): deep GCN with initial residual (alpha) and identity mapping (beta = log(theta/l+1))."""

    def __init__(self, in_dim, hidden, C, dropout=0.5, layers=8, alpha=0.1, theta=0.5):
        super().__init__()
        self.inp, self.out = Linear(in_dim, hidden), Linear(hidden, C)
        self.convs = torch.nn.ModuleList(GCN2Conv(hidden, alpha, theta, layer=i + 1, cached=False)
                                         for i in range(layers))
        self.dropout = dropout

    def forward(self, x, ei):
        x = F.dropout(x, self.dropout, self.training)
        h = h0 = self.inp(x).relu()
        for conv in self.convs:
            h = F.dropout(h, self.dropout, self.training)
            h = conv(h, h0, ei).relu()
        return self.out(F.dropout(h, self.dropout, self.training))


class JKNet(torch.nn.Module):
    """Jumping Knowledge (Xu et al. 2018) over SAGE layers: the classifier sees every layer's output, so each node
    picks its own neighbourhood range (mode cat or max)."""

    def __init__(self, in_dim, hidden, C, dropout=0.5, layers=3, mode="cat"):
        super().__init__()
        dims = [in_dim] + [hidden] * layers
        self.convs = torch.nn.ModuleList(SAGEConv(a, b) for a, b in zip(dims[:-1], dims[1:]))
        self.jk = JumpingKnowledge(mode, hidden, layers) if mode == "lstm" else JumpingKnowledge(mode)
        self.out = Linear(hidden * layers if mode == "cat" else hidden, C)
        self.dropout = dropout

    def forward(self, x, ei):
        h, hs = x, []
        for conv in self.convs:
            h = F.dropout(h, self.dropout, self.training)
            h = conv(h, ei).relu()
            hs.append(h)
        return self.out(F.dropout(self.jk(hs), self.dropout, self.training))


class GTPE(torch.nn.Module):
    """src/models.py GraphTransformer plus encodings and DropEdge.

    The last pe_dim columns of x are encodings, projected and added to the input embedding (as in Dwivedi &
    Bresson 2021 for Laplacian PE and Dwivedi et al. 2022 for RWPE). sign_flip: random sign flips per eigenvector
    in training (Laplacian PE only). edge_drop: DropEdge (Rong et al. 2020) on undirected edges in training.
    """

    def __init__(self, in_dim, hidden, C, dropout=0.5, num_layers=2, heads=8, pe_dim=0, sign_flip=False,
                 edge_drop=0.0):
        super().__init__()
        self.pe_dim, self.sign_flip, self.edge_drop, self.dropout = pe_dim, sign_flip, edge_drop, dropout
        self.input = Linear(in_dim - pe_dim, hidden, weight_initializer="glorot", bias_initializer="zeros")
        self.pe_proj = Linear(pe_dim, hidden) if pe_dim else None
        self.attn = torch.nn.ModuleList(TransformerConv(hidden, hidden // heads, heads=heads, root_weight=False)
                                        for _ in range(num_layers))
        self.attn_out = torch.nn.ModuleList(Linear(hidden, hidden) for _ in range(num_layers))
        self.ffn = torch.nn.ModuleList(torch.nn.Sequential(Linear(hidden, 2 * hidden), torch.nn.ReLU(),
                                                           Linear(2 * hidden, hidden)) for _ in range(num_layers))
        self.norm1 = torch.nn.ModuleList(torch.nn.BatchNorm1d(hidden) for _ in range(num_layers))
        self.norm2 = torch.nn.ModuleList(torch.nn.BatchNorm1d(hidden) for _ in range(num_layers))
        self.head = Linear(hidden, C)

    def forward(self, x, ei):
        split = x.size(1) - self.pe_dim
        x, pe = x[:, :split], x[:, split:]
        x = F.dropout(x, self.dropout, self.training)
        h = self.input(x)
        if self.pe_dim:
            if self.sign_flip and self.training:
                pe = pe * (torch.randint(0, 2, (1, self.pe_dim), device=pe.device) * 2 - 1)
            h = h + self.pe_proj(pe)
        ei = remove_self_loops(ei)[0]
        if self.training and self.edge_drop > 0:
            ei = dropout_edge(ei, p=self.edge_drop, force_undirected=True)[0]
        ei, _ = add_self_loops(ei, num_nodes=x.size(0))
        for attn, out, ffn, n1, n2 in zip(self.attn, self.attn_out, self.ffn, self.norm1, self.norm2):
            h = n1(h + F.dropout(out(attn(h, ei)), self.dropout, self.training))
            h = n2(h + F.dropout(ffn(h), self.dropout, self.training))
        return self.head(h)


def _norm_adj(ei, n):
    ei, w = gcn_norm(ei, None, n, add_self_loops=True)
    return torch.sparse_coo_tensor(ei, w, (n, n)).coalesce()


class NAGphormer(torch.nn.Module):
    """NAGphormer (Chen et al. 2023): each node is a sequence of K+1 hop tokens [X, ÂX, ..., Â^K X]; a Transformer
    runs over the tokens of one node and an attention-based readout weights the hops per node. Range is decoupled
    from depth, and each node chooses how much to trust far hops (useful when neighbours disagree)."""

    def __init__(self, in_dim, hidden, C, dropout=0.5, K=4, layers=1, heads=8, iso=0.0):
        super().__init__()
        self.K, self.dropout, self.iso = K, dropout, iso
        self.inp = Linear(in_dim, hidden)
        layer = torch.nn.TransformerEncoderLayer(hidden, heads, 2 * hidden, dropout, batch_first=True,
                                                 norm_first=True)
        # layers = 0: ablation without the Transformer (tokens only projected, then the same readout).
        self.enc = (torch.nn.TransformerEncoder(layer, layers, enable_nested_tensor=False) if layers
                    else torch.nn.Identity())
        self.norm = torch.nn.LayerNorm(hidden)
        self.read = Linear(2 * hidden, 1)
        self.out = torch.nn.Sequential(Linear(hidden, hidden // 2), torch.nn.ReLU(), Linear(hidden // 2, C))
        self._cache = None

    def tokens(self, x, ei):
        key = (ei.data_ptr(), ei.size(1), x.data_ptr())
        if self._cache is None or self._cache[0] != key:
            with torch.no_grad():
                A, h, toks = _norm_adj(ei, x.size(0)), x, [x]
                for _ in range(self.K):
                    h = torch.sparse.mm(A, h)
                    toks.append(h)
            self._cache = (key, torch.stack(toks, 1))
        return self._cache[1]

    def forward(self, x, ei):
        t = self.tokens(x, ei)
        if self.training and self.iso > 0:
            # Cold-start augmentation: for a random share of nodes, replace the co-purchase hop tokens by what an
            # isolated node has (its own features), so the model learns to rely on the ego / kNN tokens then.
            m = torch.rand(t.size(0), device=t.device) < self.iso
            t = t.clone()
            t[m, 1:self.K + 1] = t[m, :1]
        t = F.dropout(t, self.dropout, self.training)
        h = self.norm(self.enc(self.inp(t)))  # [N, K+1, hidden]
        node, hops = h[:, :1], h[:, 1:]
        a = torch.softmax(self.read(torch.cat([node.expand_as(hops), hops], -1)), dim=1)
        z = node.squeeze(1) + (a * hops).sum(1)
        return self.out(F.dropout(z, self.dropout, self.training))


class GlobalLinearAttention(torch.nn.Module):
    """SGFormer's one-layer all-pair attention with linear cost (Wu et al. 2023): Frobenius-normalised Q, K;
    out = (V + Q(K^T V)/N) / (1 + Q(K^T 1)/N), averaged over heads."""

    def __init__(self, dim, heads=1):
        super().__init__()
        self.q, self.k, self.v = (Linear(dim, dim * heads) for _ in range(3))
        self.heads, self.dim = heads, dim

    def forward(self, x):
        n = x.size(0)
        q = self.q(x).view(n, self.heads, self.dim)
        k = self.k(x).view(n, self.heads, self.dim)
        v = self.v(x).view(n, self.heads, self.dim)
        q, k = q / q.norm(p=2), k / k.norm(p=2)
        kv = torch.einsum("lhm,lhd->hmd", k, v)
        num = torch.einsum("nhm,hmd->nhd", q, kv) + n * v
        den = torch.einsum("nhm,hm->nh", q, k.sum(0)).unsqueeze(-1) + n
        return (num / den).mean(1)


class SGFormerGT(torch.nn.Module):
    """SGFormer-style model (Wu et al. 2023): a global linear-attention branch over ALL nodes in parallel with a
    local graph branch (here our 2-layer GT instead of SGFormer's GCN, since GT is the strongest local model);
    out = head(w * local + (1 - w) * global). The global branch lets a node far from the train region attend to
    similar products anywhere (labelled ones included), with no edges, so it also works in cold start."""

    def __init__(self, in_dim, hidden, C, dropout=0.5, graph_weight=0.8):
        super().__init__()
        self.local = GTPE(in_dim, hidden, hidden, dropout)
        self.inp = Linear(in_dim, hidden)
        self.ln1, self.ln2 = torch.nn.LayerNorm(hidden), torch.nn.LayerNorm(hidden)
        self.attn = GlobalLinearAttention(hidden)
        self.head = Linear(hidden, C)
        self.w, self.dropout = graph_weight, dropout

    def forward(self, x, ei):
        g = F.dropout(x, self.dropout, self.training)
        g = F.dropout(self.ln1(self.inp(g)).relu(), self.dropout, self.training)
        g = F.dropout(self.ln2((self.attn(g) + g) / 2).relu(), self.dropout, self.training)
        return self.head(self.w * self.local(x, ei) + (1 - self.w) * g)


class NAGphormerKNN(NAGphormer):
    """NAGphormer with two token families: K hops over the co-purchase graph and K2 hops over a fixed text-kNN graph
    (label-free MiniLM cosine kNN, symmetrised). The readout picks per node which relation and range to trust; in
    cold start the co-purchase tokens of a test node collapse to its own features while its kNN tokens stay."""

    def __init__(self, in_dim, hidden, C, knn_ei, dropout=0.5, K=6, K2=2, layers=1, heads=8, iso=0.0):
        super().__init__(in_dim, hidden, C, dropout, K, layers, heads, iso)
        self.knn_ei, self.K2 = knn_ei, K2

    def tokens(self, x, ei):
        key = (ei.data_ptr(), ei.size(1), x.data_ptr())
        if self._cache is None or self._cache[0] != key:
            with torch.no_grad():
                toks = [x]
                for e, k in [(ei, self.K), (self.knn_ei, self.K2)]:
                    A, h = _norm_adj(e, x.size(0)), x
                    for _ in range(k):
                        h = torch.sparse.mm(A, h)
                        toks.append(h)
            self._cache = (key, torch.stack(toks, 1))
        return self._cache[1]
