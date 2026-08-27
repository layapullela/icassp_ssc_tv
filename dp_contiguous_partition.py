"""Contiguous k-segment partitioning of an ordered affinity matrix via DP.

Drop-in alternative to free (unconstrained) spectral clustering for
problems where the ground-truth clusters are known to be contiguous blocks
along the existing index order (e.g. the SBM benchmark cases in
benchmark_sbm.py, where cluster membership is a contiguous range).

Rather than embedding C's derived affinity and running k-means /
discretize on the eigenvectors (which has no notion of "contiguous"),
this searches exactly over the restricted hypothesis class of contiguous
partitions and picks the one that minimizes the k-way normalized cut --
solvable exactly by dynamic programming because NCut decomposes additively
over segments (see derivation in the docstring below).
"""

import numpy as np


def dp_contiguous_ncut_partition(C, k, min_size=1, symmetrize=True, criterion='ncut'):
    """
    Partition N points (indexed 0..N-1, assumed already in the order in
    which contiguous clusters are expected to appear) into k contiguous
    segments optimising an additive per-segment score over the affinity W.

    Two criteria are supported via the ``criterion`` argument:

    ``'ncut'`` (normalized cut, default)
    -------------------------------------
        NCut = sum_s [ 1 - within(S_s) / vol(S_s) ]
             = k - sum_s within(S_s) / vol(S_s)

        Minimizing NCut ⟺ maximizing sum_s within(S_s) / vol(S_s) — additive
        over segments, solvable by DP in O(k N²).

    ``'dcsbm'`` (degree-corrected SBM profile log-likelihood)
    -----------------------------------------------------------
        For each segment S with within-block edge mass m_s = within(S) and
        block volume κ_s = vol(S) = Σ_{i∈S} deg_i (full-graph degrees),
        the per-segment DC-SBM contribution is:

            L(S) = m_s · log(m_s) − 2 · κ_s · log(κ_s)

        (convention: 0·log(0) = 0). This is the within-block term of the
        Karrer–Newman (2011) profile log-likelihood for the degree-corrected
        SBM, with between-block terms dropped to preserve additivity over
        segments. Maximising Σ_s L(S_s) selects blocks where m_s is large
        relative to κ_s², removing NCut's balance bias and adapting naturally
        to heterogeneous-degree graphs.

    Common fields
    -------------
        within(S) = sum_{i,j in S} W_ij
        vol(S)    = sum_{i in S} deg_i        (deg_i = full-graph row sum)

    Parameters
    ----------
    C : ndarray (N, N)
        Self-expressiveness coefficient matrix (or any square affinity-like
        matrix already in index order).
    k : int
        Number of contiguous segments (clusters).
    min_size : int, default 1
        Minimum allowed segment length; set >1 to forbid degenerate
        near-empty clusters.
    symmetrize : bool, default True
        If True, use W = |C| + |C|^T as the affinity (matches the W used
        elsewhere for OSC / SSC-TV spectral clustering in this benchmark).
        If False, C is used directly and assumed already symmetric and
        nonnegative.
    criterion : {'ncut', 'dcsbm'}, default 'ncut'
        Scoring criterion for the DP (see above).

    Returns
    -------
    labels : ndarray (N,), int
        Cluster label 0..k-1 for each point, in index order.
    boundaries : list of int, length k+1
        Segment s covers indices boundaries[s] : boundaries[s+1].
    score_value : float
        Achieved DP-optimal score.  For ``criterion='ncut'`` this equals the
        NCut value k − Σ_s within(s)/vol(s); for ``criterion='dcsbm'`` it is
        the total DC-SBM log-likelihood Σ_s L(S_s).
    """
    C = np.asarray(C, dtype=float)
    if C.ndim != 2 or C.shape[0] != C.shape[1]:
        raise ValueError("C must be a square (N, N) matrix.")
    N = C.shape[0]
    k = int(k)
    if k < 1:
        raise ValueError("k must be >= 1.")
    if k > N:
        raise ValueError(f"k ({k}) cannot exceed N ({N}).")
    if min_size < 1:
        raise ValueError("min_size must be >= 1.")
    if k * min_size > N:
        raise ValueError(
            f"k * min_size ({k * min_size}) exceeds N ({N}); "
            "relax min_size or reduce k."
        )

    W = np.abs(C) + np.abs(C).T if symmetrize else np.array(C, copy=True)
    np.fill_diagonal(W, 0.0)  # no self-loops in within/degree accounting

    # 2D prefix sums for O(1) within(a, b) queries; S[i, j] = sum W[0:i, 0:j]
    S = np.zeros((N + 1, N + 1))
    S[1:, 1:] = np.cumsum(np.cumsum(W, axis=0), axis=1)
    deg = W.sum(axis=1)  # full-graph degree of each node
    deg_cum = np.concatenate([[0.0], np.cumsum(deg)])

    def within(a, b):
        return S[b, b] - S[a, b] - S[b, a] + S[a, a]

    def vol(a, b):
        return deg_cum[b] - deg_cum[a]

    # Per-segment gain for the chosen criterion.
    # NCut:   gain(a,b) = within(a,b) / vol(a,b)       ∈ [0, 1]
    # DC-SBM: gain(a,b) = m·log(m) − 2·κ·log(κ)       (unbounded, can be < 0)
    #   Both are additive over segments; DP maximises the total.
    NEG_INF = -np.inf
    gain = np.full((N + 1, N + 1), NEG_INF)
    if criterion == 'ncut':
        for a in range(N):
            for b in range(a + min_size, N + 1):
                v = vol(a, b)
                gain[a, b] = within(a, b) / v if v > 0 else 0.0
    elif criterion == 'dcsbm':
        for a in range(N):
            for b in range(a + min_size, N + 1):
                m_s = within(a, b)
                k_s = vol(a, b)
                if k_s <= 0.0:
                    gain[a, b] = 0.0
                elif m_s <= 0.0:
                    gain[a, b] = -2.0 * k_s * np.log(k_s)
                else:
                    gain[a, b] = m_s * np.log(m_s) - 2.0 * k_s * np.log(k_s)
    else:
        raise ValueError(f"criterion must be 'ncut' or 'dcsbm', got {criterion!r}")

    # dp[s, b] = best total gain using exactly s segments covering [0, b)
    dp = np.full((k + 1, N + 1), NEG_INF)
    dp[0, 0] = 0.0
    back = np.full((k + 1, N + 1), -1, dtype=int)

    for s in range(1, k + 1):
        lo = s * min_size
        hi = N - (k - s) * min_size
        for b in range(lo, hi + 1):
            a_lo = (s - 1) * min_size
            a_hi = b - min_size
            best_val, best_a = NEG_INF, -1
            for a in range(a_lo, a_hi + 1):
                prev = dp[s - 1, a]
                if prev == NEG_INF:
                    continue
                val = prev + gain[a, b]
                if val > best_val:
                    best_val, best_a = val, a
            dp[s, b] = best_val
            back[s, b] = best_a

    if dp[k, N] == NEG_INF:
        raise RuntimeError(
            "No feasible contiguous partition found; check k and min_size "
            "against N."
        )

    # Backtrack to recover boundaries.
    boundaries = [N]
    b, s = N, k
    while s > 0:
        a = back[s, b]
        boundaries.append(a)
        b, s = a, s - 1
    boundaries.reverse()

    labels = np.empty(N, dtype=int)
    for seg_idx in range(k):
        labels[boundaries[seg_idx]:boundaries[seg_idx + 1]] = seg_idx

    if criterion == 'ncut':
        score_value = float(k - dp[k, N])  # NCut = k − Σ gain
    else:
        score_value = float(dp[k, N])       # total DC-SBM log-likelihood
    return labels, boundaries, score_value


def cluster_from_C_ordered(C, k, min_size=1, symmetrize=True, criterion='ncut'):
    """Drop-in replacement for ``cluster_from_C(coeff, k)`` that restricts
    to contiguous partitions. Only appropriate when the input order is
    known to align with true cluster membership (e.g. the SBM benchmark
    cases, which generate contiguous blocks by construction).

    ``criterion`` selects the per-segment scoring function passed to
    ``dp_contiguous_ncut_partition``: ``'ncut'`` (default) uses the
    normalised-cut gain; ``'dcsbm'`` uses the degree-corrected SBM profile
    log-likelihood.
    """
    labels, _, _ = dp_contiguous_ncut_partition(
        C, k, min_size=min_size, symmetrize=symmetrize, criterion=criterion,
    )
    return labels


if __name__ == "__main__":
    # Minimal sanity check on a synthetic 3-block affinity matrix.
    rng = np.random.default_rng(0)
    sizes = [50, 60, 90]
    labels_true = np.repeat(np.arange(3), sizes)
    N = len(labels_true)
    same = labels_true[:, None] == labels_true[None, :]
    C_syn = np.where(same, rng.uniform(0.4, 1.0, (N, N)),
                      rng.uniform(0.0, 0.05, (N, N)))
    np.fill_diagonal(C_syn, 0.0)

    labels, boundaries, ncut = dp_contiguous_ncut_partition(C_syn, k=3)
    from sklearn.metrics import adjusted_rand_score
    print("boundaries:", boundaries)
    print("NCut:", ncut)
    print("ARI vs ground truth:", adjusted_rand_score(labels_true, labels))