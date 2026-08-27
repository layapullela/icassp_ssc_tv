"""SSC-TV-E1E21: hybrid L1 + L2,1 on E, L1 TV on P and Q.

See README.md for the shared ADMM formulation and how this file differs
from the other SSC-TV variants.
"""

import numpy as np

from ssc_tv import (
    soft_threshold,
    block_soft_threshold_cols,
    finite_diff_matrix,
)


def ssc_admm_nuc_tv_e1_e21(
    Y,
    lambda_e1=1.0,
    lambda_e21=1.0,
    lambda_z=0.1,
    gamma=0.1,
    mu=1.0,
    sigma=1.0,
    rho=1.0,
    max_iter=50,
    tol=1e-4,
):
    """
    SSC-ADMM with TV on C and a hybrid L1 + L2,1 (sparse-group-lasso)
    penalty on E: E is shrunk elementwise, then its columns are shrunk
    as groups, via an auxiliary variable F = E.

    Objective
    ---------
        min   λ_e1 ||E||_1 + λ_e21 ||F||_{2,1}
              + (λ_z/2) ||Y - YX - E||_F^2 + γ(||DC||_1 + ||CD^T||_1)
        s.t.  X = C_off,  DC = P,  CD^T = Q,  E = F,  diag(C) = 0

    Parameters
    ----------
    Y         : ndarray (n, N)   data matrix (columns = data points)
    lambda_e1 : float            weight on elementwise ||E||_1
    lambda_e21: float            weight on column-group ||F||_{2,1}
    lambda_z  : float            weight on reconstruction loss
    gamma     : float            TV regularisation weight
    mu        : float            ADMM penalty for X = C_off
    sigma     : float            ADMM penalty for the TV auxiliaries
    rho       : float            ADMM penalty for E = F
    max_iter  : int
    tol       : float            convergence tolerance

    Returns
    -------
    X, C, E, F : ndarrays
        F is returned separately from E for diagnostics: at convergence
        E ≈ F, and ``np.linalg.norm(F, axis=0)`` is a natural per-column
        outlier score (see ``outlier_scores`` below).
    """
    n, N = Y.shape

    D = finite_diff_matrix(N)
    K = D.T @ D
    eigs, V = np.linalg.eigh(K)

    denom = mu + sigma * (eigs[:, None] + eigs[None, :])
    A_inv = np.linalg.inv(lambda_z * (Y.T @ Y) + mu * np.eye(N))

    X = np.zeros((N, N))
    C = np.zeros((N, N))
    E = np.zeros((n, N))
    F = np.zeros((n, N))
    P = np.zeros((N - 1, N))
    Q = np.zeros((N, N - 1))

    Lambda = np.zeros((N, N))
    Pi_P = np.zeros((N - 1, N))
    Pi_Q = np.zeros((N, N - 1))
    Pi_F = np.zeros((n, N))

    for it in range(max_iter):
        X_prev = X

        # 1. X-update
        C_off = C - np.diag(np.diag(C))
        X = A_inv @ (lambda_z * (Y.T @ (Y - E)) + mu * C_off - Lambda)

        # 2. C-update (Sylvester equation via eigendecomposition of K)
        P_tilde = P - Pi_P / sigma
        Q_tilde = Q - Pi_Q / sigma
        RHS_C = mu * (X + Lambda / mu) + sigma * (D.T @ P_tilde + Q_tilde @ D)
        C = V @ ((V.T @ RHS_C @ V) / denom) @ V.T
        np.fill_diagonal(C, 0.0)

        # 3-4. P- and Q-updates
        DC = D @ C
        CDt = C @ D.T
        P = soft_threshold(DC + Pi_P / sigma, gamma / sigma)
        Q = soft_threshold(CDt + Pi_Q / sigma, gamma / sigma)

        # 5a. E-update -- elementwise soft threshold, folding in the E=F link
        R = Y - Y @ X
        F_tilde = F - Pi_F / rho
        E = soft_threshold(
            (lambda_z * R + rho * F_tilde) / (lambda_z + rho),
            lambda_e1 / (lambda_z + rho),
        )

        # 5b. F-update -- column-wise group soft threshold (outlier columns)
        F = block_soft_threshold_cols(E + Pi_F / rho, lambda_e21 / rho)

        # 6. Dual updates
        C_off = C - np.diag(np.diag(C))
        Lambda += mu * (X - C_off)
        Pi_P += sigma * (DC - P)
        Pi_Q += sigma * (CDt - Q)
        Pi_F += rho * (E - F)

        # Convergence check
        primal_res = max(
            np.linalg.norm(X - C_off, 'fro'),
            np.linalg.norm(DC - P, 'fro'),
            np.linalg.norm(CDt - Q, 'fro'),
            np.linalg.norm(E - F, 'fro'),
        )
        dual_res = mu * np.linalg.norm(X - X_prev, 'fro')
        if primal_res < tol and dual_res < tol:
            break

        mu_max, gamma_0 = 10.0, 1.1
        gamma_step = gamma_0 if max(primal_res, dual_res) < tol else 1.0
        mu = min(mu_max, gamma_step * mu)
        sigma = min(mu_max, gamma_step * sigma)

    return X, C, E, F


def outlier_scores(F):
    """Per-column L2 norm of F -- higher means more outlier-like."""
    return np.linalg.norm(F, axis=0)


def flag_outliers(F, frac=None, n_mad=3.0):
    """
    Flag outlier columns from the column-group residual F.

    If ``frac`` is given, flags the top ``frac`` fraction of columns by
    score (use when the approximate outlier rate is known, e.g. from a
    benchmark). Otherwise uses a robust MAD-based threshold:
    score > median + n_mad * 1.4826 * MAD.
    """
    scores = outlier_scores(F)
    if frac is not None:
        k = max(1, int(round(frac * len(scores))))
        idx = np.argsort(scores)[::-1][:k]
        mask = np.zeros(len(scores), dtype=bool)
        mask[idx] = True
        return mask, scores
    med = np.median(scores)
    mad = np.median(np.abs(scores - med)) + 1e-12
    thresh = med + n_mad * 1.4826 * mad
    return scores > thresh, scores