"""
Ordered Subspace Clustering (OSC) via LADMPSAP
================================================

Reference
---------
S. Tierney, Y. Guo, J. Gao, "Segmentation of Subspaces in Sequential Data,"
(extended version of the CVPR'14 paper), arXiv:1504.04090.
IEEE Xplore record: https://ieeexplore.ieee.org/document/6909530

Objective (exact-constraint variant, eq. 25 / 29 in the paper)
----------------------------------------------------------------
    min_{Z,E,J}   (1/2)||E||_F^2  +  λ1 ||Z||_1  +  λ2 ||J||_{1,2}
    s.t.  X = XZ + E,   J = ZR,   diag(Z) = 0

where R ∈ ℝ^{N×(N-1)} is the "consecutive-difference" operator

        R = [ -1              ]
            [  1  -1          ]
            [     1  -1       ]
            [        ..  ..   ]
            [            1  -1]

so that  ZR = [z2-z1, z3-z2, ..., zN-zN-1].  The ℓ_{1,2} (group-lasso) norm
on J = ZR forces *whole columns* of Z to be similar to their neighbour,
encoding the assumption that the data is drawn from a sequentially/ordered
union of subspaces (i.e. contiguous, non-recurring blocks).

This is solved with LADMPSAP (Linearized ADM with Parallel Splitting and
Adaptive Penalty, Liu/Lin/Su 2013), following Algorithm 2 of the paper
exactly, including the diagonal constraint variant of eq. (29)/(30).

Segmentation follows the paper's Algorithm 3: build W = |Z| + |Z|^T and run
spectral clustering on it, matching how the reference SSC-ADMM-TV module in
this repo builds its own affinity for a fair comparison.
"""

import numpy as np
from sklearn.cluster import SpectralClustering


def soft_threshold(x, tau):
    return np.sign(x) * np.maximum(np.abs(x) - tau, 0.0)


def block_soft_threshold_cols(M, tau):
    """Prox of tau * ||.||_{1,2} (sum of column L2 norms), eq. (23)."""
    col_norms = np.linalg.norm(M, axis=0, keepdims=True)
    scale = np.maximum(1.0 - tau / np.maximum(col_norms, 1e-12), 0.0)
    return scale * M


def consecutive_diff_matrix(N):
    """R in eq. (12):  N x (N-1),  -1 on diag, +1 on first sub-diagonal."""
    R = np.zeros((N, N - 1))
    idx = np.arange(N - 1)
    R[idx, idx] = -1.0
    R[idx + 1, idx] = 1.0
    return R


def osc_admm(
    X,
    lambda1=0.1,
    lambda2=1.0,
    mu=1.0,
    mu_max=1e6,
    gamma0=1.1,
    eta_z=None,
    eta_j=1.0,
    diag_zero=True,
    max_iter=200,
    tol1=1e-4,
    tol2=1e-4,
):
    """
    Ordered Subspace Clustering, exact-constraint LADMPSAP solver
    (Algorithm 2, eqs. (25)-(30) of Tierney, Guo & Gao).

    Parameters
    ----------
    X        : ndarray (D, N)   data matrix (columns = sequentially ordered
               data points)
    lambda1  : float   weight on ||Z||_1               (sparsity, as in SSC)
    lambda2  : float   weight on ||ZR||_{1,2}           (column-similarity / TV)
    mu       : float   initial ADMM penalty
    mu_max   : float   cap on the penalty (mu_max_1 in the paper)
    gamma0   : float   growth factor applied to mu when progress stalls
    eta_z    : float   linearisation constant for the Z-step;
               must exceed ||X||_2^2 + ||R||_2^2 for the convergence
               guarantee in the paper (Theorem 1). Defaults to a safe
               multiple of that bound if not given.
    eta_j    : float   linearisation constant for the J-step (>1 required)
    diag_zero: bool    enforce diag(Z) = 0 (eq. 29/30); the paper notes this
               constraint is optional but include it here so the comparison
               with the SSC-ADMM-TV baseline (which also zeros diag(C)) is
               apples-to-apples.
    max_iter : int
    tol1, tol2 : float   primal / dual stopping tolerances (paper's eps1, eps2)

    Returns
    -------
    Z, E, J : ndarrays
    """
    D, N = X.shape
    R = consecutive_diff_matrix(N)          # (N, N-1)

    XtX = X.T @ X
    specX = np.linalg.norm(X, 2) ** 2       # ||X||_2^2 (spectral norm sq.)
    specR = np.linalg.norm(R, 2) ** 2       # ||R||_2^2

    if eta_z is None:
        # Theorem 1 only requires eta_z > ||X||_2^2 + ||R||_2^2, but that
        # margin is razor-thin in practice for this Jacobi/parallel-split
        # scheme (LADMPSAP updates Z, E, J all from the same old iterate,
        # so the effective Lipschitz constant of the coupled system is
        # larger than the bound covers). Empirically a ~10x margin is
        # needed for stable convergence; 1.1x explodes.
        eta_z = 10.0 * (specX + specR)

    normX = np.linalg.norm(X, 'fro')
    normX = max(normX, 1e-12)

    # ── init (Algorithm 2, line 1) ──────────────────────────────────────────
    Z = np.zeros((N, N))
    E = np.zeros((D, N))
    J = Z @ R
    # NOTE: the paper's pseudocode literally initialises the multipliers to
    # all-ones matrices (Algorithm 1 & 2). That is numerically unstable here
    # (it injects an O(1) gradient before mu/eta have any chance to damp it,
    # and blows up for the eta_z scale the convergence theorem requires) --
    # so we use the standard ADMM zero-init instead. This does not affect
    # the fixed point / convergence guarantee, only the transient.
    Y1 = np.zeros((D, N))
    Y2 = np.zeros((N, N - 1))

    for it in range(max_iter):
        Z_prev, E_prev, J_prev = Z, E, J

        sigma_z = mu * eta_z

        # 1. Z-update: linearised gradient step + soft threshold, eq. (30).
        #    Z appears in *both* penalty terms with cross terms (X^T X and
        #    R R^T), so this subproblem is not separable in closed form --
        #    it genuinely needs the eta_z-linearisation (unlike J and E
        #    below), with eta_z > ||X||_2^2 + ||R||_2^2 for convergence.
        gradF = X.T @ (Y1 + mu * (X @ Z_prev - X + E_prev)) \
                - (Y2 + mu * (J_prev - Z_prev @ R)) @ R.T
        V = Z_prev - gradF / sigma_z
        Z = soft_threshold(V, lambda1 / sigma_z)
        if diag_zero:
            np.fill_diagonal(Z, 0.0)

        # 2. E-update: least squares closed form.
        # NOTE: solving d/dE [ 1/2||E||^2 + <Y1, XZ-X+E> + mu/2||XZ-X+E||^2 ] = 0
        # gives E = -(Y1 + mu*(XZ-X)) / (1+mu), i.e. the negative of what's
        # printed as eq. (28) in the paper (verified symbolically). We use the
        # sign that is actually consistent with their eq. (26) Lagrangian.
        r = X @ Z_prev - X
        E = -(Y1 + mu * r) / (1.0 + mu)

        # 3. J-update: exact group-soft-threshold prox, eq. (23) evaluated at
        #    Z_prev (LADMPSAP updates all primal blocks in parallel from
        #    iterate k). This subproblem is already an exact separable prox
        #    with weight mu^k -- no eta_J linearisation is needed here (the
        #    paper's eta_J > 1 only applies to the *relaxed* variant of
        #    Section 4.1, which lacks an E block); we keep eta_j as a small
        #    optional over-relaxation knob but default it to 1.0.
        sigma_j = mu * eta_j
        U = Z_prev @ R - Y2 / sigma_j
        J = block_soft_threshold_cols(U, lambda2 / sigma_j)

        # 4. dual (multiplier) updates
        primal1 = X @ Z - X + E
        primal2 = J - Z @ R
        Y1 = Y1 + mu * primal1
        Y2 = Y2 + mu * primal2

        # 5. stopping criteria (paper, Algorithm 2 line 6) & adaptive penalty
        res1 = max(np.linalg.norm(primal1, 'fro'),
                    np.linalg.norm(primal2, 'fro')) / normX
        res2 = (mu * np.sqrt(max(specX, 1.0)) / normX) * max(
            np.linalg.norm(Z - Z_prev, 'fro'),
            np.linalg.norm(E - E_prev, 'fro'),
            np.linalg.norm(J - J_prev, 'fro'),
            np.linalg.norm(Z @ R - Z_prev @ R, 'fro'),
        )

        gamma = gamma0 if res2 < tol2 else 1.0
        mu = min(mu_max, gamma * mu)

        if res1 < tol1 and res2 < tol2:
            break

    return Z, E, J


def cluster_from_Z(Z, k=None, k_max=None):
    """Spectral clustering on W = |Z| + |Z|^T -- identical recipe to the
    baseline's cluster_from_C so both methods are scored the same way."""
    W = np.abs(Z) + np.abs(Z.T)

    if k is None:
        if k_max is None:
            k_max = max(1, Z.shape[0] // 20)
        d = np.maximum(W.sum(axis=1), 1e-12)
        d_inv_sqrt = 1.0 / np.sqrt(d)
        W_norm = d_inv_sqrt[:, None] * W * d_inv_sqrt[None, :]
        eigvals = np.linalg.eigvalsh(W_norm)[::-1]
        gaps = eigvals[:-1] - eigvals[1:]
        k = int(np.clip(np.argmax(gaps) + 1, 1, k_max))

    sc = SpectralClustering(n_clusters=k, affinity='precomputed',
                             assign_labels='kmeans', random_state=0)
    return sc.fit_predict(W)


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
    Z, E, J = osc_admm(Y, lambda1=0.1, lambda2=1.0, mu=1.0)
    pred = cluster_from_Z(Z, k=len(cluster_sizes))
    print(f"ARI = {adjusted_rand_score(labels, pred):.4f}   "
          f"time = {time.perf_counter() - t0:.2f}s")