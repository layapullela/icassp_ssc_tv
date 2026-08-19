"""
OSC impelmentation, translated to python from https://github.com/sjtrny/SubKit/blob/master/osc/osc_exact.m
"""

import numpy as np
from sklearn.cluster import SpectralClustering

def solve_l1(x, lambda_1):
    # soft thresholding
    return np.sign(x) * np.maximum(np.abs(x) - lambda_1, 0.0)

def solve_l1l2(W, lambda_1): # soft thresolding for columns
    m, n = np.shape(W)
    E = W
    for i in range(n): 
        norm_col = np.linalg.norm(W[:, i])

        if norm_col > lambda_1: 
            E[:, i] = (norm_col - lambda_1) * W[:, i] / norm_col
        else: 
            E[:, i] = 0.0
    return E

def solve_l2(x, lambda_1): # normalize by 1 + lambda
    return x / (1.0 + lambda_1)

def norm_l1(x):
    return np.sum(np.abs(x))

def norm_l1l2(x):
    L = 0
    for i in range(np.shape(x)[1]):
        L += np.linalg.norm(x[:, i])
    return L


def osc_exact(X, lambda_1, lambda_2, mu=None, diagconstraint=False, max_iter=200):
    X = np.asarray(X, dtype=float)

    if diagconstraint is None:
        diagconstraint = False

    max_iterations = int(max_iter)
    func_vals = np.zeros(max_iterations)

    xm, xn = X.shape

    Z = np.zeros((xn, xn))
    Z_prev = Z.copy()

    E = np.zeros((xm, xn))
    E_prev = E.copy()

    ones_R = np.ones((xn, xn - 1))
    R = (
        np.triu(ones_R, k=1) - np.triu(ones_R)
        + np.triu(ones_R, k=-1) - np.triu(ones_R)
    )

    J = np.zeros((xn, xn - 1))
    J_prev = J.copy()

    Y_1 = np.zeros((xm, xn))
    Y_2 = np.zeros((xn, xn - 1))

    if mu is None:
        mu = 0.1

    mu_max = 10.0
    gamma_0 = 1.1

    normfX = np.linalg.norm(X, "fro")
    rho = (np.linalg.norm(X, 2) ** 2) * 1.1

    tol_1 = 1e-2
    tol_2 = 1e-4

    for k in range(max_iterations):

        # Update Z
        partial = mu * (
            X.T @ (
                X @ Z_prev
                - (X - E_prev - (1.0 / mu) * Y_1)
            )
            + (
                Z_prev @ R
                - (J_prev + (1.0 / mu) * Y_2)
            ) @ R.T
        )

        V = Z_prev - (1.0 / rho) * partial
        Z = solve_l1(V, lambda_1 / rho)

        if diagconstraint:
            np.fill_diagonal(Z, 0.0)

        # Update E
        V = -X @ Z_prev + X - (1.0 / mu) * Y_1
        E = solve_l2(V, 1.0 / mu)

        # Update J
        J = solve_l1l2(
            Z_prev @ R - (1.0 / mu) * Y_2,
            lambda_2 / mu,
        )

        # Update dual variables
        Y_1 = Y_1 + mu * (X @ Z - X + E)
        Y_2 = Y_2 + mu * (J - Z @ R)

        # Decide whether to increase mu
        change_before_mu_update = (
            mu
            * np.sqrt(rho)
            * max(
                np.linalg.norm(Z - Z_prev, "fro"),
                np.linalg.norm(E - E_prev, "fro"),
                np.linalg.norm(J - J_prev, "fro"),
            )
            / normfX
        )

        if change_before_mu_update < tol_2:
            gamma = gamma_0
        else:
            gamma = 1.0

        mu = min(mu_max, gamma * mu)

        # Objective
        func_vals[k] = (
            0.5 * np.linalg.norm(E, "fro") ** 2
            + lambda_1 * norm_l1(Z)
            + lambda_2 * norm_l1l2(Z @ R)
        )

        # Convergence
        primal_ok = (
            np.linalg.norm(X @ Z - X + E, "fro") / normfX < tol_1
            and np.linalg.norm(J - Z @ R, "fro") / normfX < tol_1
        )

        change_after_mu_update = (
            mu
            * np.sqrt(rho)
            * max(
                np.linalg.norm(Z - Z_prev, "fro"),
                np.linalg.norm(E - E_prev, "fro"),
                np.linalg.norm(J - J_prev, "fro"),
            )
            / normfX
        )

        if primal_ok and change_after_mu_update < tol_2:
            break

        Z_prev = Z.copy()
        E_prev = E.copy()
        J_prev = J.copy()

    return Z


def cluster_from_Z(Z, k=None, k_max=None):
    """Spectral clustering on W = |Z| + |Z|^T (OSC paper, Algorithm 3)."""
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

    sc = SpectralClustering(n_clusters=k, affinity="precomputed",
                            assign_labels="kmeans", random_state=0)
    return sc.fit_predict(W)