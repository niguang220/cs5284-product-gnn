"""Two-layer MLP, GCN and GraphSAGE with the same width and dropout, so arms differ only in aggregation.

Also a Graph Transformer and NAGphormer on the same node features, and a text Transformer trained from scratch on
token ids.
"""
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, Linear, SAGEConv, TransformerConv
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.utils import add_self_loops, remove_self_loops


class MLP(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, num_classes, dropout=0.5):
        super().__init__()
        # Same initialisation scheme as GCNConv (Glorot weights, zero bias), so A vs B differ only in
        # aggregation. The drawn values differ even with the same seed; comparisons average over seeds.
        self.lin1 = Linear(in_dim, hidden_dim, weight_initializer="glorot", bias_initializer="zeros")
        self.lin2 = Linear(hidden_dim, num_classes, weight_initializer="glorot", bias_initializer="zeros")
        self.dropout = dropout

    def forward(self, x, edge_index=None):
        # edge_index is ignored; it is accepted so MLP and GCN share one training loop.
        x = F.dropout(x, self.dropout, self.training)
        h = self.lin1(x).relu()
        h = F.dropout(h, self.dropout, self.training)
        return self.lin2(h)


class GCN(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, num_classes, dropout=0.5):
        super().__init__()
        # cached=False: the normalised adjacency is recomputed on every call, so the cold-start test
        # can pass an edge_index with test edges removed at inference time.
        self.conv1 = GCNConv(in_dim, hidden_dim, cached=False)
        self.conv2 = GCNConv(hidden_dim, num_classes, cached=False)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = F.dropout(x, self.dropout, self.training)
        h = self.conv1(x, edge_index).relu()
        h = F.dropout(h, self.dropout, self.training)
        return self.conv2(h, edge_index)


class TapeNet(torch.nn.Module):
    """TAPE's MLP, GCN and GraphSAGE (XiaoxinHe/TAPE at d9881f7, core/GNNs/*/model.py).

    Each layer but the last is followed by BatchNorm, ReLU and dropout; layers use PyG's default initialisation.
    use_pred: the input is GPT's ranked classes as ids (0 = no answer, else class id + 1). Each rank goes through
    a learned embedding and the embeddings are concatenated, so in_dim must be hidden_dim * number of ranks.
    Unlike TAPE, GCNConv does not cache the adjacency, so the cold-start test can pass different edges.
    """

    def __init__(self, kind, in_dim, hidden_dim, num_classes, num_layers=4, dropout=0.0, use_pred=False):
        super().__init__()
        layer = {"MLP": torch.nn.Linear, "SAGE": SAGEConv, "GCN": lambda a, b: GCNConv(a, b, cached=False)}[kind]
        dims = [in_dim] + [hidden_dim] * (num_layers - 1) + [num_classes]
        self.encoder = torch.nn.Embedding(num_classes + 1, hidden_dim) if use_pred else None
        self.layers = torch.nn.ModuleList(layer(a, b) for a, b in zip(dims[:-1], dims[1:]))
        self.bns = torch.nn.ModuleList(torch.nn.BatchNorm1d(hidden_dim) for _ in range(num_layers - 1))
        self.uses_edges = kind != "MLP"
        self.dropout = dropout

    def _apply_layer(self, layer, x, edge_index):
        return layer(x, edge_index) if self.uses_edges else layer(x)

    def forward(self, x, edge_index=None):
        if self.encoder is not None:
            x = self.encoder(x).flatten(start_dim=1)
        for layer, bn in zip(self.layers[:-1], self.bns):
            x = bn(self._apply_layer(layer, x, edge_index)).relu()
            x = F.dropout(x, self.dropout, self.training)
        return self._apply_layer(self.layers[-1], x, edge_index)


class SAGE(torch.nn.Module):
    """GraphSAGE with mean aggregation: h_i' = W_self h_i + W_neigh * mean of h_j over neighbours j.

    The separate self weight is why it is the cold-start comparison for GCN.
    num_layers defaults to 2 to preserve existing experiments and checkpoint keys.
    """

    def __init__(self, in_dim, hidden_dim, num_classes, dropout=0.5, num_layers=2):
        super().__init__()
        if not isinstance(num_layers, int) or num_layers < 2:
            raise ValueError("num_layers must be an integer of at least 2")
        self.conv1 = SAGEConv(in_dim, hidden_dim, aggr="mean")
        self.hidden_convs = torch.nn.ModuleList(
            SAGEConv(hidden_dim, hidden_dim, aggr="mean") for _ in range(num_layers - 2)
        )
        self.conv2 = SAGEConv(hidden_dim, num_classes, aggr="mean")
        self.num_layers = num_layers
        self.dropout = dropout
        for conv in (self.conv1, *self.hidden_convs, self.conv2):  # Glorot weights, zero bias
            glorot(conv.lin_l.weight)
            zeros(conv.lin_l.bias)
            glorot(conv.lin_r.weight)

    def forward(self, x, edge_index):
        x = F.dropout(x, self.dropout, self.training)
        h = self.conv1(x, edge_index).relu()
        for conv in self.hidden_convs:
            h = F.dropout(h, self.dropout, self.training)
            h = conv(h, edge_index).relu()
        h = F.dropout(h, self.dropout, self.training)
        return self.conv2(h, edge_index)


class GraphTransformer(torch.nn.Module):
    """Graph Transformer (Dwivedi & Bresson, 2021): multi-head attention over each node's neighbours and itself.

    Each layer: h <- BN(h + O(attention)), then h <- BN(h + FFN(h)). The last pe_dim columns of x are Laplacian
    eigenvectors (src/positional.py), projected and added to the input embedding. An eigenvector's sign is
    arbitrary, so signs are flipped at random during training.
    """

    def __init__(self, in_dim, hidden_dim, num_classes, dropout=0.5, num_layers=2, heads=8, pe_dim=0):
        super().__init__()
        self.pe_dim = pe_dim
        self.dropout = dropout
        self.input = Linear(in_dim - pe_dim, hidden_dim, weight_initializer="glorot", bias_initializer="zeros")
        self.pe_proj = Linear(pe_dim, hidden_dim) if pe_dim else None
        # root_weight=False: the node attends to itself through the self-loop instead of a separate skip weight.
        self.attn = torch.nn.ModuleList(
            TransformerConv(hidden_dim, hidden_dim // heads, heads=heads, root_weight=False) for _ in range(num_layers)
        )
        self.attn_out = torch.nn.ModuleList(Linear(hidden_dim, hidden_dim) for _ in range(num_layers))
        self.ffn = torch.nn.ModuleList(
            torch.nn.Sequential(Linear(hidden_dim, 2 * hidden_dim), torch.nn.ReLU(), Linear(2 * hidden_dim, hidden_dim))
            for _ in range(num_layers)
        )
        self.norm1 = torch.nn.ModuleList(torch.nn.BatchNorm1d(hidden_dim) for _ in range(num_layers))
        self.norm2 = torch.nn.ModuleList(torch.nn.BatchNorm1d(hidden_dim) for _ in range(num_layers))
        self.head = Linear(hidden_dim, num_classes)

    def forward(self, x, edge_index):
        split = x.size(1) - self.pe_dim
        x, pe = x[:, :split], x[:, split:]
        x = F.dropout(x, self.dropout, self.training)
        h = self.input(x)
        if self.pe_dim:
            if self.training:
                pe = pe * (torch.randint(0, 2, (1, self.pe_dim), device=pe.device) * 2 - 1)
            h = h + self.pe_proj(pe)
        # Self-loops keep isolated nodes (e.g. cold-start test nodes) from attending over an empty set.
        edge_index, _ = add_self_loops(remove_self_loops(edge_index)[0], num_nodes=x.size(0))
        for attn, out, ffn, norm1, norm2 in zip(self.attn, self.attn_out, self.ffn, self.norm1, self.norm2):
            a = F.dropout(out(attn(h, edge_index)), self.dropout, self.training)
            h = norm1(h + a)
            h = norm2(h + F.dropout(ffn(h), self.dropout, self.training))
        return self.head(h)


class NAGphormer(torch.nn.Module):
    """NAGphormer (Chen et al., 2023): each node is a sequence of K+1 hop tokens [X, ÂX, ..., Â^K X] (Â the
    GCN-normalised adjacency with self-loops). A Transformer runs over the tokens of one node and an attention
    readout weights the hops per node, so the range is decoupled from depth. Selected in exp/g-arch (K = 6).
    """

    def __init__(self, in_dim, hidden_dim, num_classes, dropout=0.5, K=6, layers=1, heads=8):
        super().__init__()
        self.K, self.dropout = K, dropout
        self.inp = Linear(in_dim, hidden_dim)
        layer = torch.nn.TransformerEncoderLayer(hidden_dim, heads, 2 * hidden_dim, dropout, batch_first=True,
                                                 norm_first=True)
        self.enc = torch.nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = torch.nn.LayerNorm(hidden_dim)
        self.read = Linear(2 * hidden_dim, 1)
        self.out = torch.nn.Sequential(Linear(hidden_dim, hidden_dim // 2), torch.nn.ReLU(),
                                       Linear(hidden_dim // 2, num_classes))
        self._cache = None

    def tokens(self, x, edge_index):
        # The hop tokens are fixed features, so they are computed once per (x, graph) and cached.
        key = (edge_index.data_ptr(), edge_index.size(1), x.data_ptr())
        if self._cache is None or self._cache[0] != key:
            with torch.no_grad():
                ei, w = gcn_norm(edge_index, None, x.size(0), add_self_loops=True)
                A = torch.sparse_coo_tensor(ei, w, (x.size(0), x.size(0))).coalesce()
                h, toks = x, [x]
                for _ in range(self.K):
                    h = torch.sparse.mm(A, h)
                    toks.append(h)
            self._cache = (key, torch.stack(toks, 1))
        return self._cache[1]

    def forward(self, x, edge_index):
        t = F.dropout(self.tokens(x, edge_index), self.dropout, self.training)
        h = self.norm(self.enc(self.inp(t)))  # [N, K+1, hidden]
        node, hops = h[:, :1], h[:, 1:]
        a = torch.softmax(self.read(torch.cat([node.expand_as(hops), hops], -1)), dim=1)
        z = node.squeeze(1) + (a * hops).sum(1)
        return self.out(F.dropout(z, self.dropout, self.training))


class TextTransformer(torch.nn.Module):
    """Transformer encoder trained from scratch on WordPiece token ids (no pretrained weights).

    Token + learned position embeddings, pre-norm encoder layers, mean pooling over non-padding tokens,
    linear classifier. embed() returns the pooled vector, for use as node features in graph models.
    """

    def __init__(self, vocab_size, num_classes, dim=256, heads=4, num_layers=2, max_len=256, dropout=0.1, pad_id=0):
        super().__init__()
        self.tok = torch.nn.Embedding(vocab_size, dim, padding_idx=pad_id)
        self.pos = torch.nn.Embedding(max_len, dim)
        layer = torch.nn.TransformerEncoderLayer(
            dim, heads, 4 * dim, dropout, activation="gelu", batch_first=True, norm_first=True
        )
        self.encoder = torch.nn.TransformerEncoder(layer, num_layers, enable_nested_tensor=False)
        self.norm = torch.nn.LayerNorm(dim)
        self.head = torch.nn.Linear(dim, num_classes)
        self.dropout = dropout

    def embed(self, ids, mask):
        """ids, mask: [batch, length]; mask is True on real tokens."""
        pos = torch.arange(ids.size(1), device=ids.device)
        h = F.dropout(self.tok(ids) + self.pos(pos), self.dropout, self.training)
        h = self.norm(self.encoder(h, src_key_padding_mask=~mask))
        m = mask.unsqueeze(-1).to(h.dtype)
        return (h * m).sum(1) / m.sum(1).clamp(min=1)

    def forward(self, ids, mask):
        return self.head(F.dropout(self.embed(ids, mask), self.dropout, self.training))
