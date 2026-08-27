import numpy as np

from ssc_tv import estimate_k_from_data


def compute_subspace_distance_sq(X, U):
    """Computes the squared distance from each point in X to subspace U.

    Parameters:
    -----------
    X : ndarray of shape (D, N)
        Data matrix containing N points of dimension D.
    U : ndarray of shape (D, d)
        Orthonormal basis for the subspace.

    Returns:
    --------
    distances : ndarray of shape (N,)
        Squared orthogonal distance of each point to subspace U.
    """
    if U.size == 0 or U.shape[1] == 0:
        return np.sum(X**2, axis=0)
    proj_norms_sq = np.sum((U.T @ X) ** 2, axis=0)
    total_norms_sq = np.sum(X**2, axis=0)
    return np.maximum(0.0, total_norms_sq - proj_norms_sq)


def _neighbor_kernel(s, weight, set_middle_to_zero=True):
    """Odd-length kernel: ``weight`` on each of the 2s neighbors, 0 at center.

    ``np.convolve(..., mode='same')`` with this kernel matches the original
    edge handling (missing neighbors outside ``[0, N)`` contribute 0).
    """
    if s <= 0:
        return np.array([0.0])
    kernel = np.full(2 * s + 1, float(weight))

    if set_middle_to_zero:
        kernel[s] = 0.0 

    return kernel


def tkss(X, K, d, lam=1.0, s=1, max_iter=50, random_state=None):
    """Temporal K-Subspaces (TKSS) Algorithm.

    Parameters:
    -----------
    X : ndarray of shape (D, N)
        Data matrix with D features and N sequential time steps.
    K : int
        Number of linear subspaces (clusters).
    d : int
        Dimension of each subspace (d < D).
    lam : float
        Sequential weighting parameter lambda.
    s : int
        Number of temporal neighbors on each side.
    max_iter : int
        Maximum number of outer loop iterations.
    random_state : int or None
        Seed for reproducible initialization.

    Returns:
    --------
    U : list of ndarray
        List of K matrices of shape (D, d) representing subspace bases.
    labels : ndarray of shape (N,)
        Cluster assignments (0 to K-1) for each sequential point.
    """
    rng = np.random.RandomState(random_state)
    D, N = X.shape
    K = int(K)
    d = int(max(1, min(int(d), max(D - 1, 1))))
    s = int(max(0, s))
    #breakpoint()

    #labels = rng.randint(0, K, size=N)  # random init
    # Contiguous equal-length blocks (sequential)
    cuts = np.linspace(0, N, K + 1, dtype=int)
    labels = np.repeat(np.arange(K), np.diff(cuts))

    U = [np.zeros((D, d)) for _ in range(K)] # init U to zero vectors
    w_kernel = _neighbor_kernel(s, lam if s > 0 else 0.0, set_middle_to_zero=False)
    seq_kernel = _neighbor_kernel(s, 1.0, set_middle_to_zero=False)

    for _iteration in range(max_iter):
        labels_prev = labels.copy()
        #breakpoint()

        # -------------------------------------------------------------
        # Subspace Learning Step: Update U_k for each cluster
        # -------------------------------------------------------------
        for k in range(K):
            member = (labels == k).astype(np.float64)
            w_k = member + np.convolve(member, w_kernel, mode="same") # diagonal
            if np.sum(w_k) > 0:
                X_weighted = X * np.sqrt(np.maximum(w_k, 0.0)) # multiply by the weight of the diagonal
                U_mat, _, _ = np.linalg.svd(X_weighted, full_matrices=False)
                n_keep = min(d, U_mat.shape[1])
                if n_keep <= 0:
                    U[k], _ = np.linalg.qr(rng.randn(D, d))
                else:
                    U[k] = U_mat[:, :n_keep]
            else:
                U[k], _ = np.linalg.qr(rng.randn(D, d))

        # -------------------------------------------------------------
        # Cluster Assignment Step: Evaluate (8)
        # -------------------------------------------------------------
        dist_kss = np.empty((K, N))
        for l in range(K):
            dist_kss[l, :] = compute_subspace_distance_sq(X, U[l])

        total_cost = np.empty((K, N))
        for l in range(K):
            seq_loss = np.convolve(dist_kss[l], seq_kernel, mode="same")
            total_cost[l, :] = dist_kss[l] + lam * seq_loss

        labels = np.argmin(total_cost, axis=0)

        if np.array_equal(labels, labels_prev):
            break

    return U, labels


def tkss_cluster(X, k=None, d=1, lam=1.0, s=1, max_iter=50, random_state=None,
                 k_max=None, method='eigengap', min_k=2, penalty=0.0):
    """Fit TKSS and return labels plus per-point residual.

    Residual is the squared distance to the assigned subspace, used as an
    outlier score in the SBM case-4 protocol (higher = more outlier-like).

    If ``k`` is None it is inferred from the observation matrix ``X`` via
    ``ssc_tv.estimate_k_from_data`` (``method='eigengap'`` or ``'ncut'``).
    """
    if k is None:
        k = estimate_k_from_data(
            X, k_max=k_max, min_k=min_k, method=method, penalty=penalty,
        )
    k = int(k)
    U, labels = tkss(
        X, K=k, d=d, lam=lam, s=s, max_iter=max_iter, random_state=random_state,
    )
    n = X.shape[1]
    residual = np.zeros(n)
    for c in range(int(k)):
        mask = labels == c
        if np.any(mask):
            residual[mask] = compute_subspace_distance_sq(X[:, mask], U[c])
    return labels, residual


# ---------------------------------------------------------------------
# Example Execution on Synthetic 2D Sequential Data
# ---------------------------------------------------------------------
if __name__ == "__main__":
    np.random.seed(42)

    # Generate sequential data switching between two 1D lines in 2D space
    t1 = np.linspace(-1, 1, 20)
    line1 = np.vstack([t1, 0.5 * t1]) + np.random.normal(0, 0.05, (2, 20))  # Line 1

    t2 = np.linspace(-1, 1, 20)
    line2 = np.vstack([t2, -2.0 * t2]) + np.random.normal(0, 0.05, (2, 20))  # Line 2

    # Concatenate into sequence: Line 1 -> Line 2
    X_toy = np.hstack([line1, line2])  # Shape (2, 40)

    # Run TKSS with K=2, d=1, lambda=0.5, s=2
    bases, estimated_clusters = tkss(
        X_toy, K=2, d=1, lam=0.5, s=2, random_state=0
    )

    #breakpoint()

    print("Estimated Cluster Labels for Sequence:")
    print(estimated_clusters)
