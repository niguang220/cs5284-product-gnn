"""Label-shift correction and Bayesian fusion with an LLM's zero-shot answers.

Under label shift p(x | y) is the same in train and test but the class prior moves. A model trained on train
nodes estimates p_tr(y | x); the test posterior is p_te(y | x) ∝ p_tr(y | x) · π_te(y) / π_tr(y).
π_te is unknown, but an LLM's answers g on the unlabelled test nodes give it: E[onehot(g)] = C π_te, with
C[g, y] = P(g | y) estimated on labelled nodes (black-box shift estimation, Lipton et al. 2018). The LLM never saw
the train prior, so its confusion matrix is the same on train and test.
The same C turns the LLM's answer into a likelihood, so the fused posterior needs no tuned weight:
    p(y | x, g) ∝ p_tr(y | x) · π_te(y) / π_tr(y) · P(g | y)
(the model and the LLM are assumed conditionally independent given y).
"""
import numpy as np
from scipy.optimize import minimize_scalar, nnls

K = 47


def prior(y, mask, floor=None):
    """Class frequencies on mask; classes below `floor` are raised to it (then renormalised)."""
    p = np.bincount(y[mask], minlength=K) / mask.sum()
    if floor:
        p = np.maximum(p, floor)
        p /= p.sum()
    return p


def confusion(answer, y, mask, alpha=1.0):
    """C[g, y] = P(LLM answers g | true class y) on labelled nodes, Laplace-smoothed; columns sum to 1.

    answer is a class id, or -1 for no parsable answer, which gets its own last row. Classes with no labelled node
    get a uniform column: their answers carry no information about them.
    """
    C = np.full((K + 1, K), alpha)
    np.add.at(C, (answer[mask], y[mask]), 1)  # answer -1 indexes the last row
    return C / C.sum(0, keepdims=True)


def bbse_prior(answer, C, mask, seen):
    """π_te from the LLM's answers on mask: non-negative least squares of C π = answer frequencies, seen classes only."""
    mu = np.bincount(answer[mask] % (K + 1), minlength=K + 1) / mask.sum()  # -1 -> last row
    w, _ = nnls(C[:, seen], mu)
    pi = np.zeros(K)
    pi[seen] = w
    return pi / pi.sum()


def naive_prior(answer, mask, alpha=1.0):
    """π_te as the LLM's answer frequencies (biased by the LLM's own confusions)."""
    a = answer[mask]
    return (np.bincount(a[a >= 0], minlength=K) + alpha) / ((a >= 0).sum() + K * alpha)


def em_prior(prob, pi_tr, iters=500):
    """Saerens et al. (2002) EM estimate of π_te from the model's own test posteriors."""
    pi = pi_tr.copy()
    for _ in range(iters):
        q = prob * (pi / pi_tr)
        q /= q.sum(1, keepdims=True)
        new = q.mean(0)
        if np.abs(new - pi).max() < 1e-8:
            break
        pi = new
    return pi


def fit_temperature(prob, y, mask):
    """Temperature T minimising validation NLL of softmax(log p / T)."""
    logp = np.log(np.clip(prob[mask], 1e-12, 1))

    def nll(t):
        z = logp / t
        z -= z.max(1, keepdims=True)
        return -(z[np.arange(len(z)), y[mask]] - np.log(np.exp(z).sum(1))).mean()

    return minimize_scalar(nll, bounds=(0.05, 20), method="bounded").x


def temper(prob, t):
    z = np.log(np.clip(prob, 1e-12, 1)) / t
    z -= z.max(1, keepdims=True)
    p = np.exp(z)
    return p / p.sum(1, keepdims=True)


def adjust(prob, pi_tr, pi_te):
    """Posterior under the new prior."""
    q = prob * (pi_te / pi_tr)
    return q / q.sum(1, keepdims=True)


def fuse(prob, answers, Cs):
    """Multiply in the likelihood P(g | y) of each LLM's answer."""
    q = prob.copy()
    for a, C in zip(answers, Cs):
        q *= C[a % (K + 1)]
        q /= q.sum(1, keepdims=True)
    return q


def smooth(prob, adj_norm, y, train_mask, alpha=0.5, iters=10):
    """Label smoothing over the graph (the "smooth" step of Correct & Smooth): train nodes clamped to their labels."""
    z0 = prob.copy()
    z0[train_mask] = np.eye(K)[y[train_mask]]
    z = z0.copy()
    for _ in range(iters):
        z = (1 - alpha) * z0 + alpha * (adj_norm @ z)
        z[train_mask] = z0[train_mask]
    return z


def compatibility(edge_index, y, mask, strength=5.0, homophily=0.8):
    """H[a, b] = P(neighbour has label b | node has label a), counted on edges between nodes in mask.

    Each row gets `strength` pseudo-edges toward homophily·I + (1 - homophily)·uniform, so rare classes fall back
    to "neighbours share my label".
    """
    both = mask[edge_index[0]] & mask[edge_index[1]]
    H = np.zeros((K, K))
    np.add.at(H, (y[edge_index[0][both]], y[edge_index[1][both]]), 1)
    H += strength * (homophily * np.eye(K) + (1 - homophily) / K)
    return H / H.sum(1, keepdims=True)


def neighbour_log_messages(edge_index, num_nodes, evidence, H):
    """Σ over neighbours j of log m_j(y), m_j(y) = Σ_y' H[y, y'] evidence_j(y'): one round of belief propagation."""
    import scipy.sparse as sp
    A = sp.coo_matrix((np.ones(edge_index.shape[1]), (edge_index[1], edge_index[0])), shape=(num_nodes, num_nodes))
    A = A.tocsr()
    A.data[:] = 1  # duplicate edges count once
    return A @ np.log(np.clip(evidence @ H.T, 1e-9, None))


def ppr_log_prior(fused_log, adj, y, train_mask, a=0.15, iters=1, T=10):
    """Node-specific prior from the graph: log π_i / π_tr, where π_i is the personalised-PageRank diffusion of the
    neighbours' fused beliefs (train nodes one-hot), mixed 0.9 / 0.1 with the train prior.

    The label mix differs by graph region, so a region-level prior corrects the label shift locally; with iters > 1
    the beliefs are re-fused with the new prior and diffused again. adj is a scipy sparse adjacency; selected in
    exp/g-localprior. Added to the fused log posterior with weight λ = 1 (the full Saerens correction).
    """
    import scipy.sparse as sp
    pi_tr = np.maximum(np.bincount(y[train_mask], minlength=K) / train_mask.sum(), 1e-4)
    pi_tr /= pi_tr.sum()
    P = sp.diags(1 / np.maximum(np.asarray(adj.sum(1)).ravel(), 1)) @ adj
    q = np.exp(fused_log - fused_log.max(1, keepdims=True))
    q /= q.sum(1, keepdims=True)
    for _ in range(iters):
        q[train_mask] = np.eye(K)[y[train_mask]]
        s, h = np.zeros_like(q), q
        for t in range(1, T + 1):
            h = P @ h
            s += a * (1 - a) ** (t - 1) * h
        pi = 0.9 * s / np.maximum(s.sum(1, keepdims=True), 1e-12) + 0.1 * pi_tr
        z = fused_log + np.log(pi / pi_tr)
        q = np.exp(z - z.max(1, keepdims=True))
        q /= q.sum(1, keepdims=True)
    return np.log(pi / pi_tr)


def sibling_log_messages(edge_index, num_nodes, evidence, y, train_mask):
    """BP messages over "sibling" pairs (two products bought with the same product, i.e. 2-hop pairs that are not
    direct neighbours), averaged over a node's siblings so high-degree hubs do not make it over-confident.

    Compatibility is counted on train-train sibling pairs. Selected in exp/g-restore and the g-valshift exploration (docs/research_log.md).
    """
    import scipy.sparse as sp
    A = sp.csr_matrix((np.ones(edge_index.shape[1]), (edge_index[0], edge_index[1])), shape=(num_nodes, num_nodes))
    A.data[:] = 1
    S = (A @ A).tocsr()
    S.setdiag(0)
    S = S - S.multiply(A)
    S.eliminate_zeros()
    S.data[:] = 1
    H = compatibility(np.stack(S.nonzero()), y, train_mask)
    L = np.log(np.clip(evidence @ H.T, 1e-9, None))
    return (S @ L) / np.maximum(np.asarray(S.sum(1)).ravel(), 1)[:, None]
