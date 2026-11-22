"""Improvements applied at inference without retraining.

Neighbour imputation; the GPT blend (a standard ensemble); reliability fusion with P(g | y) (Kerrigan et al. 2021) and a
variant conditioned on neighbour agreement; GPT-agreement edge weights for a trained GCN; one step of belief propagation.
None of these is new; scripts/run_final_eval.py gives the fixed parameters.
"""
import torch
import torch.nn.functional as F


def drop_edges_touching(edge_index, mask):
    """edge_index without any edge with a node in mask at either end."""
    return edge_index[:, ~(mask[edge_index[0]] | mask[edge_index[1]])]


def nearest_in_pool(nodes, x_norm, pool, k, batch=4096):
    """Directed edges (source -> node) from the k most similar pool nodes to each node, by cosine similarity.

    x_norm must have unit-norm rows. A node never gets itself as a neighbour.
    """
    out = []
    for start in range(0, len(nodes), batch):
        query = nodes[start:start + batch]
        sim = x_norm[query] @ x_norm[pool].T
        sim[pool[None, :] == query[:, None]] = -2
        top = sim.topk(k, dim=1).indices
        out.append(torch.stack([pool[top].flatten(), query.repeat_interleave(k)]))
    return torch.cat(out, dim=1) if out else torch.empty(2, 0, dtype=torch.long)


def impute_edges(edge_index, how, x_norm, pool, num_nodes, k=5):
    """Give every node with no incoming edge a neighbour set, leaving all other nodes as they are.

    GraphSAGE: h_i' = W_s h_i + W_n m_i with m_i the mean over neighbours. With no neighbours PyG sets m_i = 0,
    a value no training node ever produced (no train node has degree 0). The neighbour set becomes:
      zero: nothing, m_i stays 0 (the default)
      self: {i}, so m_i = h_i
      knn:  the k most similar pool nodes
      both: {i} plus the k most similar pool nodes
    """
    isolated = (torch.bincount(edge_index[1], minlength=num_nodes) == 0).nonzero().flatten()
    extra = []
    if how in ("self", "both"):
        extra.append(torch.stack([isolated, isolated]))
    if how in ("knn", "both"):
        extra.append(nearest_in_pool(isolated, x_norm, pool, k))
    return torch.cat([edge_index] + extra, dim=1) if extra else edge_index


def gpt_one_hot(gpt_top1, num_classes):
    """One-hot of GPT's first answer; a node with no answer (-1) gets all zeros."""
    return F.one_hot(gpt_top1.clamp(min=0), num_classes).float() * (gpt_top1 >= 0).unsqueeze(1)


def blend_gpt(prob, gpt_hot, w):
    """(1 - w) * softmax(model) + w * one-hot(GPT). A node with no GPT answer keeps the model's probabilities."""
    has_answer = gpt_hot.sum(dim=1, keepdim=True) > 0
    return torch.where(has_answer, (1 - w) * prob + w * gpt_hot, prob)


def split_in_two(nodes, num_nodes, seed=0):
    """Two boolean masks that split nodes at random into equal halves."""
    perm = nodes[torch.randperm(len(nodes), generator=torch.Generator().manual_seed(seed))]
    half_a, half_b = torch.zeros(num_nodes, dtype=torch.bool), torch.zeros(num_nodes, dtype=torch.bool)
    half_a[perm[:len(perm) // 2]] = True
    half_b[perm[len(perm) // 2:]] = True
    return half_a, half_b


def answer_keys(gpt_top1, num_classes):
    """GPT's first answer as a table row: the class id, or num_classes for a node with no answer."""
    return torch.where(gpt_top1 >= 0, gpt_top1, torch.full_like(gpt_top1, num_classes))


def answer_given_class(y, gpt_top1, labelled, num_classes, smooth=0.5):
    """P(g | y) from the labelled nodes: (num_classes + 1, num_classes), each column sums to 1.

    Row j < num_classes is "GPT's first answer is j"; the last row is "no answer". Every cell gets smooth added.
    """
    keys = answer_keys(gpt_top1, num_classes)
    counts = torch.zeros(num_classes + 1, num_classes)
    counts.index_put_((keys[labelled], y[labelled]), torch.ones(int(labelled.sum())), accumulate=True)
    return (counts + smooth) / (counts + smooth).sum(0, keepdim=True)


def reliability_fusion(prob, gpt_top1, table, gamma):
    """log p_i(j) + gamma * log P(g_i | y = j), Kerrigan et al. 2021. Returns scores; argmax gives the prediction."""
    keys = answer_keys(gpt_top1, prob.size(1))
    return prob.clamp(min=1e-12).log() + gamma * table.log()[keys]


def agreement_buckets(edge_index, gpt_top1, num_nodes):
    """Per node, the share a_i of its neighbours whose first GPT answer equals its own (both must have an answer).

    Bucket 0: a_i = 0 (also nodes with no neighbours); 1: 0 < a_i < 1; 2: a_i = 1.
    """
    src, dst = edge_index
    has = gpt_top1 >= 0
    same = (gpt_top1[src] == gpt_top1[dst]) & has[src] & has[dst]
    degree = torch.bincount(dst, minlength=num_nodes).float().clamp(min=1)
    share = torch.zeros(num_nodes).index_add_(0, dst, same.float()) / degree
    return torch.where(share == 0, 0, torch.where(share == 1, 2, 1))


def answer_given_class_and_bucket(y, gpt_top1, labelled, bucket, base, kappa=5.0):
    """P(g | y, b) for the three agreement buckets, each shrunk toward base = P(g | y) with weight kappa.

    Returns (3, num_classes + 1, num_classes). With n labelled nodes of class j in bucket b, a cell is
    (count + kappa * base) / (n + kappa).
    """
    num_classes = base.size(1)
    keys = answer_keys(gpt_top1, num_classes)
    counts = torch.zeros(3, num_classes + 1, num_classes)
    counts.index_put_((bucket[labelled], keys[labelled], y[labelled]), torch.ones(int(labelled.sum())),
                      accumulate=True)
    return (counts + kappa * base.unsqueeze(0)) / (counts.sum(1, keepdim=True) + kappa)


def neighbour_aware_fusion(prob, gpt_top1, bucket, tables, gamma):
    """log p_i(j) + gamma * log P(g_i | y = j, b_i): reliability fusion that trusts GPT more when neighbours agree."""
    keys = answer_keys(gpt_top1, prob.size(1))
    return prob.clamp(min=1e-12).log() + gamma * tables[bucket, keys].log()


def agreement_edge_weights(edge_index, gpt_top1, rho):
    """Edge weight 1 when both ends have the same first GPT answer, rho otherwise (including a missing answer)."""
    src, dst = edge_index
    has = gpt_top1 >= 0
    same = (gpt_top1[src] == gpt_top1[dst]) & has[src] & has[dst]
    return torch.where(same, torch.ones(edge_index.size(1)), torch.full((edge_index.size(1),), float(rho)))


@torch.no_grad()
def gcn_logits_weighted(model, x, edge_index, edge_weight):
    """Logits of a trained src.models.GCN with weighted edges at inference. GCNConv adds self loops with weight 1."""
    model.eval()
    return model.conv2(model.conv1(x, edge_index, edge_weight).relu(), edge_index, edge_weight)


def neighbour_tables(y, gpt_top1, labelled, edge_index, num_classes, smooth=0.5):
    """Message tables for one step of belief propagation, from edges whose centre node is labelled.

    edge_label[k, j]  = P(y_u = k | y_i = j) over edges between two labelled nodes
    edge_answer[k, j] = P(g_u = k | y_i = j) over edges from an unlabelled neighbour with an answer
    Each column sums to 1 after adding smooth to every cell.
    """
    src, dst = edge_index
    centre = labelled[dst]
    both = centre & labelled[src]
    edge_label = torch.zeros(num_classes, num_classes)
    edge_label.index_put_((y[src[both]], y[dst[both]]), torch.ones(int(both.sum())), accumulate=True)
    answer = centre & ~labelled[src] & (gpt_top1[src] >= 0)
    edge_answer = torch.zeros(num_classes, num_classes)
    edge_answer.index_put_((gpt_top1[src[answer]], y[dst[answer]]), torch.ones(int(answer.sum())), accumulate=True)

    def norm(t):
        return (t + smooth) / (t + smooth).sum(0, keepdim=True)
    return norm(edge_label), norm(edge_answer)


def neighbour_evidence(y, gpt_top1, labelled, edge_index, edge_label, edge_answer):
    """Sum over neighbours u of log M(u | j): a labelled neighbour's label, else an unlabelled neighbour's GPT answer."""
    src, dst = edge_index
    total = torch.zeros(len(y), edge_label.size(1))
    from_answer = ~labelled[src] & (gpt_top1[src] >= 0)
    total.index_add_(0, dst[from_answer], edge_answer.log()[gpt_top1[src[from_answer]]])
    from_label = labelled[src]
    total.index_add_(0, dst[from_label], edge_label.log()[y[src[from_label]]])
    return total


def crossfit_blend(prob, gpt_hot, y, half_a, half_b, grid):
    """Predictions where each half gets the w chosen on the other half, so no node helps choose its own w.

    Returns the predictions, with the model's own on all other nodes, and the two chosen values of w.
    """
    pred, chosen = prob.argmax(1), []
    for pick, apply in [(half_a, half_b), (half_b, half_a)]:
        accs = [float((blend_gpt(prob[pick], gpt_hot[pick], w).argmax(1) == y[pick]).float().mean()) for w in grid]
        w = float(grid[max(range(len(grid)), key=accs.__getitem__)])
        pred[apply] = blend_gpt(prob[apply], gpt_hot[apply], w).argmax(1)
        chosen.append(w)
    return pred, chosen
