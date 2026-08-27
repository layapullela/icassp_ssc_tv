"""SSC-TV: anisotropic TV on C with L1 on E, P = DC, and Q = CD^T.

See README.md for the shared ADMM formulation and how this file differs
from the other SSC-TV variants.
"""

import warnings
import numpy as np

from dp_contiguous_partition import cluster_from_C_ordered

warnings.filterwarnings('ignore', message='.*matmul.*', category=RuntimeWarning)


# ── Helpers ───────────────────────────────────────────────────────────────────

def soft_threshold(x, tau):
    return np.sign(x) * np.maximum(np.abs(x) - tau, 0.0)


def block_soft_threshold_cols(M, tau):
    """Proximal operator of tau * ||·||_{2,1} (sum of column L2-norms).

    Each column m_j is shrunk toward zero by the group lasso rule:
        prox(m_j) = max(0, 1 - tau / ||m_j||_2) * m_j
    """
    col_norms = np.linalg.norm(M, axis=0, keepdims=True)          # (1, N)
    scale = np.maximum(1.0 - tau / np.maximum(col_norms, 1e-12), 0.0)
    return scale * M


def finite_diff_matrix(N):
    """First-order finite-difference operator D ∈ ℝ^{(N-1)×N}."""
    D = np.zeros((N - 1, N))
    idx = np.arange(N - 1)
    D[idx, idx]     = -1.0
    D[idx, idx + 1] =  1.0
    return D


# ── ADMM solver ──────────────────────────────────────────────────────────────

def ssc_admm_nuc_tv(
    Y,
    lambda_e=1.0,
    lambda_z=0.1,
    gamma=0.1,
    mu=1.0,
    sigma=1.0,
    max_iter=50,
    tol=1e-4,
):
    """
    Sparse Subspace Clustering with anisotropic Total-Variation regularisation.

    Parameters
    ----------
    Y        : ndarray (n, N)   data matrix (columns = data points)
    lambda_e : float            weight on ||E||_1
    lambda_z : float            weight on reconstruction loss
    gamma    : float            TV regularisation weight  γ(||DC||_1 + ||CD^T||_1)
    mu       : float            ADMM penalty for the X = C_off constraint
    sigma    : float            ADMM penalty for the TV auxiliary constraints
    max_iter : int
    tol      : float            convergence tolerance (max primal Frobenius residual)

    Returns
    -------
    X, C, E : ndarrays
    """
    n, N = Y.shape

    # ── Precompute static quantities ──────────────────────────────────────────
    D = finite_diff_matrix(N)                 # (N-1, N)
    K = D.T @ D                               # (N, N), symmetric PSD
    eigs, V = np.linalg.eigh(K)               # eigs ascending, V orthogonal

    # Sylvester denominator: denom[i,j] = μ + σ(λ_i + λ_j)
    denom = mu + sigma * (eigs[:, None] + eigs[None, :])          # (N, N)
    A_inv = np.linalg.inv(lambda_z * (Y.T @ Y) + mu * np.eye(N))  # for X-update

    # ── Initialise primal and dual variables ──────────────────────────────────
    X = np.zeros((N, N))
    C = np.zeros((N, N))
    E = np.zeros((n, N))
    P = np.zeros((N - 1, N))     # DC   auxiliary
    Q = np.zeros((N, N - 1))     # CD^T auxiliary

    Lambda = np.zeros((N, N))    # dual for X = C_off
    Pi_P   = np.zeros((N - 1, N))   # dual for DC = P
    Pi_Q   = np.zeros((N, N - 1))   # dual for CD^T = Q

    for it in range(max_iter):
        X_prev = X

        # 1. X-update
        C_off = C - np.diag(np.diag(C))
        X = A_inv @ (lambda_z * (Y.T @ (Y - E)) + mu * C_off - Lambda)

        # 2. C-update (Sylvester equation via eigendecomposition of K)
        P_tilde = P - Pi_P / sigma
        Q_tilde = Q - Pi_Q / sigma
        RHS_C   = mu * (X + Lambda / mu) + sigma * (D.T @ P_tilde + Q_tilde @ D)
        C = V @ ((V.T @ RHS_C @ V) / denom) @ V.T
        np.fill_diagonal(C, 0.0)

        # 3-4. P- and Q-updates (soft threshold on row/col differences)
        DC  = D @ C
        CDt = C @ D.T
        P = soft_threshold(DC  + Pi_P / sigma, gamma / sigma)
        Q = soft_threshold(CDt + Pi_Q / sigma, gamma / sigma)

        # 5. E-update
        E = soft_threshold(Y - Y @ X, lambda_e / lambda_z)

        # 6. Dual updates
        C_off = C - np.diag(np.diag(C))
        Lambda += mu    * (X   - C_off)
        Pi_P   += sigma * (DC  - P)
        Pi_Q   += sigma * (CDt - Q)

        # Convergence check
        primal_res = max(
            np.linalg.norm(X   - C_off, 'fro'),
            np.linalg.norm(DC  - P,     'fro'),
            np.linalg.norm(CDt - Q,     'fro'),
        )
        dual_res = mu * np.linalg.norm(X - X_prev, 'fro')
        if primal_res < tol and dual_res < tol:
            break

        mu_max, gamma_0 = 10.0, 1.1
        gamma_step = gamma_0 if max(primal_res, dual_res) < tol else 1.0
        mu = min(mu_max, gamma_step * mu)
        sigma = min(mu_max, gamma_step * sigma)

    return X, C, E


def ssc_admm_nuc_tv_e21(
    Y,
    lambda_e=1.0,
    lambda_z=0.1,
    gamma=0.1,
    mu=1.0,
    sigma=1.0,
    max_iter=50,
    tol=1e-4,
):
    """
    SSC-ADMM with TV on C and L2,1 norm on E (column-group sparsity).

    Objective
    ---------
        min   λ_e ||E||_{2,1}  +  (λ_z/2) ||Y − YX − E||_F^2
              +  γ ( ||DC||_1 + ||CD^T||_1 )
        s.t.  X = C_off,   DC = P,   CD^T = Q,   diag(C) = 0

    The only difference from ``ssc_admm_nuc_tv`` is the E-update, which uses
    the column-wise block soft-threshold (group lasso proximal operator) instead
    of the element-wise soft-threshold:

        E_j = max(0, 1 − (λ_e/λ_z) / ||r_j||_2) · r_j,   r = Y − YX

    This encourages entire columns of E to be zero, modelling sample-level
    (rather than entry-level) corruption.

    Parameters
    ----------
    Y        : ndarray (n, N)   data matrix (columns = data points)
    lambda_e : float            weight on ||E||_{2,1}
    lambda_z : float            weight on reconstruction loss
    gamma    : float            TV regularisation weight  γ(||DC||_1 + ||CD^T||_1)
    mu       : float            ADMM penalty for the X = C_off constraint
    sigma    : float            ADMM penalty for the TV auxiliary constraints
    max_iter : int
    tol      : float            convergence tolerance (max primal Frobenius residual)

    Returns
    -------
    X, C, E : ndarrays
    """
    n, N = Y.shape

    # ── Precompute static quantities ──────────────────────────────────────────
    D = finite_diff_matrix(N)
    K = D.T @ D
    eigs, V = np.linalg.eigh(K)

    denom = mu + sigma * (eigs[:, None] + eigs[None, :])
    A_inv = np.linalg.inv(lambda_z * (Y.T @ Y) + mu * np.eye(N))

    # ── Initialise primal and dual variables ──────────────────────────────────
    X = np.zeros((N, N))
    C = np.zeros((N, N))
    E = np.zeros((n, N))
    P = np.zeros((N - 1, N))
    Q = np.zeros((N, N - 1))

    Lambda = np.zeros((N, N))
    Pi_P   = np.zeros((N - 1, N))
    Pi_Q   = np.zeros((N, N - 1))

    for it in range(max_iter):
        X_prev = X

        # 1. X-update
        C_off = C - np.diag(np.diag(C))
        X = A_inv @ (lambda_z * (Y.T @ (Y - E)) + mu * C_off - Lambda)

        # 2. C-update (Sylvester equation via eigendecomposition of K)
        P_tilde = P - Pi_P / sigma
        Q_tilde = Q - Pi_Q / sigma
        RHS_C   = mu * (X + Lambda / mu) + sigma * (D.T @ P_tilde + Q_tilde @ D)
        C = V @ ((V.T @ RHS_C @ V) / denom) @ V.T
        np.fill_diagonal(C, 0.0)

        # 3-4. P- and Q-updates
        DC  = D @ C
        CDt = C @ D.T
        P = soft_threshold(DC  + Pi_P / sigma, gamma / sigma)
        Q = soft_threshold(CDt + Pi_Q / sigma, gamma / sigma)

        # 5. E-update — column-wise block soft-threshold (L2,1 proximal step)
        E = block_soft_threshold_cols(Y - Y @ X, lambda_e / lambda_z)

        # 6. Dual updates
        C_off = C - np.diag(np.diag(C))
        Lambda += mu    * (X   - C_off)
        Pi_P   += sigma * (DC  - P)
        Pi_Q   += sigma * (CDt - Q)

        # Convergence check
        primal_res = max(
            np.linalg.norm(X   - C_off, 'fro'),
            np.linalg.norm(DC  - P,     'fro'),
            np.linalg.norm(CDt - Q,     'fro'),
        )
        dual_res = mu * np.linalg.norm(X - X_prev, 'fro')
        if primal_res < tol and dual_res < tol:
            break

    return X, C, E


# ── Clustering ────────────────────────────────────────────────────────────────

def estimate_k_eigengap(W, k_max=None, min_k=2):
    """Eigengap heuristic on the symmetrically-normalised affinity D^{-½}WD^{-½}.
 
    The naive version of this heuristic (search over ``k = 1..k_max``) almost
    always returns ``k = 1``: the top eigenvalue of ``D^{-½}WD^{-½}`` is
    *exactly* 1 whenever the graph is connected (it's the trivial eigenvector
    tied to connectivity, not cluster structure — equivalently, the
    normalised Laplacian always has a 0 eigenvalue). Real affinity matrices
    are almost always fully connected, so the gap right after that first
    eigenvalue is typically the largest gap in the whole spectrum by a wide
    margin, regardless of how many true clusters exist. That forces
    ``argmax(gaps) + 1 == 1`` essentially by construction, not because the
    data has one cluster.
 
    The fix: exclude that trivial gap by restricting the search to
    ``k >= min_k`` (default 2). Set ``min_k=1`` to recover the naive
    behaviour.
    """
    W = np.asarray(W, dtype=float)
    n = W.shape[0]
    if k_max is None:
        k_max = max(min_k, n // 20)
    d_inv_sqrt = 1.0 / np.sqrt(np.maximum(W.sum(axis=1), 1e-12))
    W_norm = d_inv_sqrt[:, None] * W * d_inv_sqrt[None, :]
    eigvals = np.linalg.eigvalsh(W_norm)[::-1]         # descending
    gaps = eigvals[:-1] - eigvals[1:]                  # gaps[i] = λ_{i+1} - λ_{i+2}
    if gaps.size == 0:
        return min_k
    candidate_gaps = gaps[min_k - 1:k_max]             # only k in [min_k, k_max]
    if candidate_gaps.size == 0:
        return min_k
    k = int(np.argmax(candidate_gaps)) + min_k
    return int(np.clip(k, min_k, k_max))
 
 
def _ncut_cost(W, labels):
    """Normalized-cut cost sum_c cut(c, ~c) / assoc(c, V) for a labeling."""
    cost = 0.0
    for c in np.unique(labels):
        mask = labels == c
        if mask.all():
            continue
        cut = W[np.ix_(mask, ~mask)].sum()
        assoc = W[mask, :].sum()
        cost += cut / max(assoc, 1e-12)
    return cost
 
 
def estimate_k_ncut(C, k_max=None, min_k=2, min_size=1, penalty=0.0):
    """Pick k from the *actual* DP NCut objective instead of a spectral proxy.
 
    ``cluster_from_C_ordered`` already returns the exact NCut-minimizing
    contiguous partition for a given ``k``. Rather than inferring k from
    eigenvalues of a generic affinity matrix (which can disagree with what
    the DP is actually optimizing, especially after TV smoothing sharpens or
    blurs the block structure), sweep k directly and pick the point on the
    true cost curve where returns diminish.
 
    ``penalty > 0`` instead does a simple linear complexity penalty
    (cost(k) + penalty * k), e.g. AIC/BIC-style, which is often more
    predictable/tunable than knee detection.
 
    Returns (k, labels) so you don't have to recompute the partition.
    """
    W = np.abs(C) + np.abs(C.T)
    N = C.shape[0]
    if k_max is None:
        k_max = max(min_k, N // 20)
 
    labelings = {}
    costs = {}
    for k in range(min_k, k_max + 1):
        labels = cluster_from_C_ordered(C, k, min_size=min_size)
        labelings[k] = labels
        costs[k] = _ncut_cost(W, labels)
 
    ks = np.array(sorted(costs))
    vals = np.array([costs[k] for k in ks])
 
    if penalty > 0:
        best_k = int(ks[np.argmin(vals + penalty * ks)])
    else:
        # Knee point: farthest point (in normalised coords) from the chord
        # joining the first and last (k, cost) pair on the monotone curve.
        x = (ks - ks.min()) / max(ks.max() - ks.min(), 1e-12)
        y = (vals - vals.min()) / max(vals.max() - vals.min(), 1e-12)
        x0, y0, x1, y1 = x[0], y[0], x[-1], y[-1]
        num = np.abs((y1 - y0) * x - (x1 - x0) * y + x1 * y0 - y1 * x0)
        den = np.hypot(y1 - y0, x1 - x0) + 1e-12
        best_k = int(ks[np.argmax(num / den)])
 
    return best_k, labelings[best_k]
 
 
def cluster_from_C(C, k=None, k_max=None, min_size=1, method='eigengap', min_k=2, penalty=0.0):
    """Contiguous DP Normalized Cut on W = |C| + |C|^T.
 
    Restricts to ``k`` contiguous segments along the given index order and
    returns the exact NCut minimizer in that class (see
    ``dp_contiguous_partition.py``).  Appropriate when clusters are
    contiguous in the data order (SBM blocks, chain-ordered proteins).
 
    If ``k`` is None it is estimated automatically:
 
    - ``method='eigengap'`` (default): eigengap heuristic on the
      symmetrically-normalised affinity, searching ``k`` in
      ``[min_k, k_max]`` (see ``estimate_k_eigengap`` for why ``min_k``
      defaults to 2 rather than 1).
    - ``method='ncut'``: sweeps ``k`` and picks it directly from the true
      DP NCut cost curve (see ``estimate_k_ncut``) — more robust when the
      spectral proxy disagrees with the actual contiguous-partition
      objective, at the cost of solving the DP once per candidate k.
 
    ``k_max`` defaults to N // 20 in both cases.
    """
    if k is not None:
        return cluster_from_C_ordered(C, k, min_size=min_size)
 
    if method == 'ncut':
        _, labels = estimate_k_ncut(C, k_max=k_max, min_k=min_k,
                                     min_size=min_size, penalty=penalty)
        return labels
    elif method == 'eigengap':
        W = np.abs(C) + np.abs(C.T)
        k = estimate_k_eigengap(W, k_max=k_max, min_k=min_k)
        return cluster_from_C_ordered(C, k, min_size=min_size)
    else:
        raise ValueError(f"unknown method {method!r}, expected 'eigengap' or 'ncut'")


def estimate_k_from_data(Y, k_max=None, min_k=2, method='eigengap',
                         min_size=1, penalty=0.0):
    """Infer k from an observation matrix (TKSS path).

    Square ``Y`` (SBM adjacency / contact maps) is treated as an affinity via
    ``W = |Y| + |Y|^T``.  Rectangular data uses the column Gram ``Y.T @ Y``.
    ``method`` / ``min_k`` match ``cluster_from_C`` (eigengap or ncut knee).
    """
    Y = np.asarray(Y, dtype=float)
    if Y.ndim != 2:
        raise ValueError("Y must be a 2-D array")
    if Y.shape[0] == Y.shape[1]:
        C = Y
        W = np.abs(Y) + np.abs(Y.T)
    else:
        W = Y.T @ Y
        C = W
    if method == 'eigengap':
        return estimate_k_eigengap(W, k_max=k_max, min_k=min_k)
    if method == 'ncut':
        k, _ = estimate_k_ncut(
            C, k_max=k_max, min_k=min_k, min_size=min_size, penalty=penalty,
        )
        return k
    raise ValueError(f"unknown method {method!r}, expected 'eigengap' or 'ncut'")


# ── Synthetic sanity check ──────────────────────────────────────────────────

if __name__ == '__main__':
    import time
    from sklearn.metrics import adjusted_rand_score

    cluster_sizes = [20, 25, 15, 20]
    rng = np.random.default_rng(42)
    labels = np.repeat(np.arange(len(cluster_sizes)), cluster_sizes)
    same = labels[:, None] == labels[None, :]
    probs = np.where(same, 0.75, 0.05)
    N = sum(cluster_sizes)
    upper = np.triu(rng.random((N, N)) < probs, k=0).astype(float)
    Y = upper + upper.T - np.diag(np.diag(upper))

    print(f"Y: {Y.shape},  clusters: {cluster_sizes}\n")
    t0 = time.perf_counter()
    X, C, E = ssc_admm_nuc_tv(Y, lambda_e=1.0, lambda_z=0.1, gamma=0.1)
    pred = cluster_from_C(X, k=len(cluster_sizes))
    print(f"ARI = {adjusted_rand_score(labels, pred):.4f}   "
          f"time = {time.perf_counter() - t0:.2f}s")
