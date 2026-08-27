"""
SBM benchmark of OSC, SSC-TV-E1E21-L21-P (the SSC-TV variant used for
now), and Temporal K-Subspaces (TKSS).  Other SSC-TV ADMM files remain
selectable via ``--methods``.  TKSS is a sequential K-subspaces method:
it returns labels directly (no coefficient matrix / spectral clustering).
Subspace dimension d, sequential weight λ, and neighbor window s are the
tunable knobs; k is the true number of SBM blocks.

Protocol
--------
Undirected SBM test cases with Poisson observation noise (or Gaussian noise
for cases 24–25), rate/σ ∈ {0, 0.05, 0.10, 0.20, 0.30, 0.50, 0.75, 1.00,
1.50, 2.00, 3.00}  (λ = 0 / σ = 0 is the noiseless adjacency), 10
independent draws per (case, λ).  Spectral clustering (OSC / SSC-TV) uses
the true number of SBM blocks k on W = |C| + |C|^T (OSC: |Z| + |Z|^T).
TKSS is given the same k.  Pass ``--k none`` to drop the oracle k and
instead estimate it (``--k-method eigengap|ncut``, default eigengap with
``--min-k 2``).  OSC / SSC-TV infer on the coefficient affinity; TKSS on Y.

Metrics: ARI (adjusted Rand index) and NMI (normalised mutual information,
arithmetic average method) are recorded for every run.  Outlier cases also
record precision / recall / F1 for the per-column outlier score.

Cases
-----
Original
  1. 3 contiguous blocks (50, 60, 90), p_in=0.5, p_out=0.1
  2. Same partition, sparse: p_in=0.3, p_out=0.05
  3. 5 contiguous blocks (10, 20, 30, 50, 90), p_in=0.5, p_out=0.1
  4. Case 1 with 10% ER p=0.5 outlier nodes (ARI on inliers)

Size sweep (p_in=0.5, p_out=0.1).  Equal contiguous partitions are
avoided: TKSS is initialised to K equal-length blocks, so those
graphs would give it the ground-truth labels at iteration 0.
  5. Mild 3-block (55, 65, 80)
  6. Unbalanced 3-block (8, 32, 160)
  7. Two-block (80, 120)
  8. Eight-block (15, 18, 20, 22, 25, 28, 32, 40)
 13. Severe unbalanced 3-block (5, 15, 180)
 14. Twelve-block (8, 9, 10, 12, 13, 14, 16, 17, 19, 22, 25, 35)

Probability sweep (sizes 50, 60, 90)
  9.  Hard overlap: p_in=0.4, p_out=0.2
 10. Very sparse: p_in=0.15, p_out=0.03
 11. Dense: p_in=0.8, p_out=0.2
 12. Weak communities: p_in=0.25, p_out=0.12

Scalability sweep (3-block, proportional non-equal sizes, p=0.5/0.1)
 15. N=100   (27, 33, 40)
 16. N=400   (108, 132, 160)
 17. N=800   (216, 264, 320)
     [Case 5 (N=200) serves as the N=200 anchor in scalability plots]

Heterogeneous block model (different density per cluster pair; p_mat is K×K)
 18. 3-block hetero: block densities 0.70 / 0.40 / 0.20, varied cross-block
 19. 5-block hetero: diagonal 0.65→0.15, cross-block 0.01–0.07
 20. 3-block hetero sparse: diagonal 0.30 / 0.20 / 0.15

Degree-corrected SBM (theta_i ~ Gamma(alpha,1), normalised per block)
 21. DC-SBM mild  (alpha=1.0, moderate degree spread)
 22. DC-SBM heavy (alpha=0.3, heavy-tailed degree distribution / hub nodes)
 23. DC-SBM + hetero B (alpha=0.5, heterogeneous affinity matrix)

Gaussian noise model  (λ axis = σ, symmetric additive noise Y = A + σ Z_sym)
 24. 3-block p=0.5/0.1, Gaussian noise
 25. 3-block sparse p=0.15/0.03, Gaussian noise

Large K / extreme difficulty
 26. 20-block (N=350, sizes 8–27), p_in=0.5, p_out=0.05
 27. 3-block + 20% ER outlier nodes (heavy outlier stress test)
 28. Very weak communities p_in=0.15, p_out=0.10
 29. Imbalanced + weak: (10, 40, 150), p_in=0.25, p_out=0.12

Each observation matrix Y has columns scaled to unit ℓ2 norm before
clustering or tuning.  The default SSC-TV variant (SSC-TV-E1E21-L21-P) fixes λ_z = 1
and tunes the remaining penalties relative to it; λ_e21 stored in JSON
is the pre-scale value and is multiplied by sqrt(N) at call time
(N = number of columns of Y).

ADMM solvers are imported from the existing modules; SSC-TV max_iter is
raised to 200 to match OSC.  TKSS uses alternating subspace / assignment
updates (default 50 iters, early stop).  File defaults are used unless
``--params`` points at a JSON file written by ``tune_hyperparams.py``
(Optuna TPE, one vector per method).

Usage
-----
    python tune_hyperparams.py                  # Optuna TPE: OSC, SSC-TV-E1E21-L21-P, TKSS
    python tune_hyperparams.py --retune OSC     # one method; merge into JSON
    python benchmark_sbm.py --params results/best_hyperparams.json --out-dir results/tuned
    python benchmark_sbm.py --k none --k-method eigengap --min-k 2
    python benchmark_sbm.py --k none --k-method ncut
    python benchmark_sbm.py --trials 5 --out-dir results
    python benchmark_sbm.py --smoke             # one trial, λ=0, case 1 only
    python benchmark_sbm.py --cases 15_scale_100 5_mild_three 16_scale_400 17_scale_800
    python benchmark_sbm.py --cases 18_hetero_3block 21_dcsbm_mild 22_dcsbm_heavy
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from osc import cluster_from_Z, osc_exact  # noqa: E402
from bdosc import bd_qosc  # noqa: E402
from ssc_tv import (  # noqa: E402
    cluster_from_C,
    estimate_k_from_data,
    ssc_admm_nuc_tv as ssc_tv_admm,
)
from tkss import tkss_cluster  # noqa: E402


# ── module loading ────────────────────────────────────────────────────────────

def _load_py(alias: str, filename: str):
    path = ROOT / filename
    spec = importlib.util.spec_from_file_location(alias, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[alias] = mod
    spec.loader.exec_module(mod)
    return mod


_mod_l21_p = _load_py(
    "method_ssc_tv_l21_p",
    "ssc_tv_with_column_l21_on_columns_l1_on_rows.py",
)
_mod_l21_pq = _load_py(
    "method_ssc_tv_l21_pq",
    "ssc_tv_with_l21_on_rows_and_columns.py",
)
_mod_e1e21 = _load_py("method_ssc_tv_e1e21", "ssc_tv_e21_e1.py")
_mod_e1e21_l21_p = _load_py(
    "method_ssc_tv_e1e21_l21_p",
    "ssc_tv_e21_e1_and_l21_on_columns_l1_on_rows.py",
)
_mod_e1e21_l21_pq = _load_py(
    "method_ssc_tv_e1e21_l21_pq",
    "ssc_tv_e21_e1_and_l21_on_rows_and_columns.py",
)
_mod_l21_pq_sparse_c = _load_py(
    "method_ssc_tv_l21_pq_sparse_c",
    "ssc_tv_row_col_l21_c_sparse.py",
)
_mod_l21_pq_lowrank_c = _load_py(
    "method_ssc_tv_l21_pq_lowrank_c",
    "ssc-tv-nuclear-norm.py",
)


# lambda_z is fixed at 1; other ADMM penalties are tuned relative to it.
# lambda_e21 in defaults / JSON is the *pre-scale* value; make_solver multiplies
# by sqrt(N) at call time (N = number of columns of Y).
LAMBDA_Z = 1.0
SSC_DEFAULTS = dict(lambda_e=1.0, lambda_z=LAMBDA_Z, gamma=0.1, mu=0.1, sigma=1.0,
                    max_iter=200, tol=1e-4)
E1E21_DEFAULTS = dict(lambda_e1=1.0, lambda_e21=1.0, lambda_z=LAMBDA_Z, gamma=0.1,
                      mu=0.1, sigma=1.0, rho=1.0, max_iter=200, tol=1e-4)
# L21-PQ + explicit λ_c ||C||_1 split (S = C). rho is the ADMM penalty for that
# constraint and is not searched; lambda_c is the extra tuned hyperparameter.
SPARSE_C_DEFAULTS = dict(SSC_DEFAULTS, lambda_c=0.1, rho=1.0)
# Same split / rho, but λ_c weights ||C||_* (nuclear / low-rank) instead of ||C||_1.
LOWRANK_C_DEFAULTS = dict(SSC_DEFAULTS, lambda_c=0.1, rho=1.0)
# osc_exact (SubKit / osc.m): λ1 on ||Z||_1, λ2 on ||ZR||_{1,2}, mu default 1.0.
# Tuner / JSON still use lambda1/lambda2; make_solver maps them onto lambda_1/lambda_2.
OSC_DEFAULTS = dict(lambda1=0.1, lambda2=1.0, mu=0.1, max_iter=200, diagconstraint=True)
# BD-QOSC: λ1 on ||ZᵀZ||_1, λ2 on ||ZR||_{2,1}, gamma1 ADMM penalty, p growth factor.
# Requires k (block-diagonal projection). Tuner keys: lambda1/lambda2/gamma1/p.
BDOSC_DEFAULTS = dict(
    lambda1=0.1, lambda2=1.0, gamma1=0.1, p=1.1,
    max_iter=200, diagconstraint=True, pos=False,
)
TKSS_DEFAULTS = dict(lam=1.0, s=1, d=1, max_iter=200, random_state=0)

# Back-compat aliases used by older call sites / the tuner.
SSC_KW = SSC_DEFAULTS
E1E21_KW = E1E21_DEFAULTS
OSC_KW = OSC_DEFAULTS

METHOD_SPECS = [
    dict(name="OSC", kind="osc", solver=None, defaults=OSC_DEFAULTS),
    dict(name="BDOSC", kind="bdosc", solver=None, defaults=BDOSC_DEFAULTS),
    dict(name="SSC-TV", kind="ssc", solver=ssc_tv_admm, defaults=SSC_DEFAULTS),
    dict(name="SSC-TV-L21-P", kind="ssc", solver=_mod_l21_p.ssc_admm_nuc_tv,
         defaults=SSC_DEFAULTS),
    dict(name="SSC-TV-L21-PQ", kind="ssc", solver=_mod_l21_pq.ssc_admm_nuc_tv,
         defaults=SSC_DEFAULTS),
    dict(name="SSC-TV-L21-PQ-SparseC", kind="ssc",
         solver=_mod_l21_pq_sparse_c.ssc_admm_nuc_tv_sparse_c,
         defaults=SPARSE_C_DEFAULTS),
    dict(name="SSC-TV-L21-PQ-LowRankC", kind="ssc",
         solver=_mod_l21_pq_lowrank_c.ssc_admm_nuc_tv,
         defaults=LOWRANK_C_DEFAULTS),
    dict(name="SSC-TV-E1E21", kind="ssc", solver=_mod_e1e21.ssc_admm_nuc_tv_e1_e21,
         defaults=E1E21_DEFAULTS),
    dict(name="SSC-TV-E1E21-L21-P", kind="ssc",
         solver=_mod_e1e21_l21_p.ssc_admm_nuc_tv_e1_e21, defaults=E1E21_DEFAULTS),
    dict(name="SSC-TV-E1E21-L21-PQ", kind="ssc",
         solver=_mod_e1e21_l21_pq.ssc_admm_nuc_tv_e1_e21, defaults=E1E21_DEFAULTS),
    dict(name="TKSS", kind="tkss", solver=tkss_cluster, defaults=TKSS_DEFAULTS),
]

METHOD_KIND = {s["name"]: s["kind"] for s in METHOD_SPECS}
# Default comparison set: OSC and TKSS plus one SSC-TV variant
# (ssc_tv_e21_e1_and_l21_on_columns_l1_on_rows.py).  Other METHOD_SPECS
# names remain valid --methods choices.
DEFAULT_METHODS = ["OSC", "SSC-TV-E1E21-L21-P", "TKSS"]


def _osc_exact_kwargs(kw):
    """Map tuner / JSON keys onto ``osc_exact``'s MATLAB-style signature."""
    out = {}
    if "lambda_1" in kw:
        out["lambda_1"] = float(kw["lambda_1"])
    elif "lambda1" in kw:
        out["lambda_1"] = float(kw["lambda1"])
    if "lambda_2" in kw:
        out["lambda_2"] = float(kw["lambda_2"])
    elif "lambda2" in kw:
        out["lambda_2"] = float(kw["lambda2"])
    if "mu" in kw:
        out["mu"] = float(kw["mu"])
    if "diagconstraint" in kw:
        out["diagconstraint"] = bool(kw["diagconstraint"])
    elif "diag_zero" in kw:
        out["diagconstraint"] = bool(kw["diag_zero"])
    if "max_iter" in kw:
        out["max_iter"] = int(kw["max_iter"])
    return out


def normalize_Y(Y):
    """Scale each column of ``Y`` to unit ℓ2 norm (zero columns left as zeros)."""
    Y = np.asarray(Y, dtype=float)
    scale = np.linalg.norm(Y, axis=0, keepdims=True)
    scale[~np.isfinite(scale) | (scale == 0.0)] = 1.0
    return Y / scale


def _ssc_call_kwargs(kw, Y):
    """Fix ``lambda_z=1`` and scale ``lambda_e21`` by ``sqrt(N)`` (pre-scale in kw)."""
    call_kw = dict(kw)
    call_kw["lambda_z"] = LAMBDA_Z
    if "lambda_e21" in call_kw:
        n_cols = int(Y.shape[1])
        call_kw["lambda_e21"] = float(call_kw["lambda_e21"]) * np.sqrt(n_cols)
    return call_kw


def _bdosc_kwargs(kw):
    """Map tuner / JSON keys onto ``bd_qosc``'s signature."""
    out = {}
    if "lambda_1" in kw:
        out["lambda_1"] = float(kw["lambda_1"])
    elif "lambda1" in kw:
        out["lambda_1"] = float(kw["lambda1"])
    if "lambda_2" in kw:
        out["lambda_2"] = float(kw["lambda_2"])
    elif "lambda2" in kw:
        out["lambda_2"] = float(kw["lambda2"])
    if "gamma_1" in kw:
        out["gamma_1"] = float(kw["gamma_1"])
    elif "gamma1" in kw:
        out["gamma_1"] = float(kw["gamma1"])
    if "p" in kw:
        out["p"] = float(kw["p"])
    if "max_iter" in kw:
        out["max_iter"] = int(kw["max_iter"])
    if "diagconstraint" in kw:
        out["diagconstraint"] = bool(kw["diagconstraint"])
    if "pos" in kw:
        out["pos"] = bool(kw["pos"])
    return out


def make_solver(spec, kwargs):
    """Bind a solver spec to a kwargs dict.

    ADMM methods (OSC / SSC-TV): ``fn(Y) -> (coeff, E, F)``.
    BDOSC: ``fn(Y, k) -> (coeff, E, F)`` (needs k for block-diagonal projection).
    TKSS: ``fn(Y, k) -> (labels, residual)``.

    SSC variants always run with ``lambda_z=1``.  ``lambda_e21`` stored in
    kwargs / JSON is the value *before* the ``sqrt(N)`` column-count scale.
    """
    kw = dict(kwargs)
    if spec["kind"] == "ssc":
        kw["lambda_z"] = LAMBDA_Z
    if spec["kind"] == "osc":
        def fn(Y, _kw=kw):
            Z = osc_exact(Y, **_osc_exact_kwargs(_kw))
            E = Y - Y @ Z
            return Z, E, None
        return fn
    if spec["kind"] == "bdosc":
        def fn(Y, k, _kw=kw):
            Z, _, _ = bd_qosc(Y, int(k), **_bdosc_kwargs(_kw))
            E = Y - Y @ Z
            return Z, E, None
        return fn
    if spec["kind"] == "tkss":
        def fn(Y, k, _kw=kw):
            d = int(round(_kw.get("d", 1)))
            s = int(round(_kw.get("s", 1)))
            return tkss_cluster(
                Y,
                k=k,
                d=max(1, d),
                lam=float(_kw.get("lam", 1.0)),
                s=max(0, s),
                max_iter=int(_kw.get("max_iter", 50)),
                random_state=_kw.get("random_state", 0),
            )
        return fn
    solver = spec["solver"]

    def fn(Y, _kw=kw, _solver=solver):
        out = _solver(Y, **_ssc_call_kwargs(_kw, Y))
        C, E = out[1], out[2]
        F = out[3] if len(out) > 3 else None
        return C, E, F
    return fn


def build_methods(overrides=None, max_iter=None, names=None):
    """Construct ``[(name, solver), ...]`` with optional per-method kwargs."""
    overrides = overrides or {}
    wanted = list(names) if names is not None else [s["name"] for s in METHOD_SPECS]
    methods = []
    for spec in METHOD_SPECS:
        if spec["name"] not in wanted:
            continue
        kw = dict(spec["defaults"])
        if max_iter is not None and spec["kind"] not in ("tkss",):
            kw["max_iter"] = max_iter
        extra = overrides.get(spec["name"], {})
        kw.update(extra)
        if spec["kind"] == "ssc":
            kw["lambda_z"] = LAMBDA_Z
        methods.append((spec["name"], make_solver(spec, kw)))
    return methods


def load_param_overrides(path):
    data = json.loads(Path(path).read_text())
    methods = data.get("methods", data)
    overrides = {}
    for name, info in methods.items():
        params = dict(info.get("params", info))
        params.pop("lambda_z", None)  # always 1; ignore stale JSON values
        overrides[name] = params
    return overrides


METHODS = build_methods()  # defaults; rebuilt in main() if --params is given


# ── SBM generation ────────────────────────────────────────────────────────────

def generate_sbm(cluster_sizes, p_in=None, p_out=None, rng=None, *, p_mat=None):
    """Undirected SBM: Bernoulli upper triangle, symmetrised, zero diagonal.

    If *p_mat* (K×K array-like) is given it overrides p_in/p_out, enabling
    heterogeneous within- and between-cluster densities across every block pair.
    """
    labels = np.repeat(np.arange(len(cluster_sizes)), cluster_sizes)
    n = labels.size
    if p_mat is not None:
        pm = np.asarray(p_mat, dtype=float)
        probs = pm[labels[:, None], labels[None, :]]
    else:
        same = labels[:, None] == labels[None, :]
        probs = np.where(same, p_in, p_out)
    upper = np.triu(rng.random((n, n)) < probs, k=1).astype(float)
    A = upper + upper.T
    return A, labels


def generate_dcsbm(cluster_sizes, B, theta_alpha, rng):
    """Degree-corrected SBM with Gamma(theta_alpha, 1) degree heterogeneity.

    theta_i is drawn per node and normalised so the mean theta equals 1 within
    each block, preserving the same expected density as B[k,l].
    Edge probability: min(1, theta_i * theta_j * B[z_i, z_j]).
    Small theta_alpha (e.g. 0.3) gives a heavy-tailed degree distribution.
    """
    labels = np.repeat(np.arange(len(cluster_sizes)), cluster_sizes)
    n = labels.size
    theta = rng.gamma(theta_alpha, 1.0, size=n)
    for k_idx in range(len(cluster_sizes)):
        mask = labels == k_idx
        mu = theta[mask].mean()
        if mu > 0:
            theta[mask] /= mu
    Bm = np.asarray(B, dtype=float)
    probs = np.clip(
        theta[:, None] * theta[None, :] * Bm[labels[:, None], labels[None, :]],
        0.0, 1.0,
    )
    upper = np.triu(rng.random((n, n)) < probs, k=1).astype(float)
    A = upper + upper.T
    return A, labels


def add_poisson_noise(A, lam, rng):
    """Y = A + Poisson(λ) on the upper triangle, then symmetrised.

    λ = 0 leaves A unchanged (Poisson(0) ≡ 0).  Entries stay non-negative.
    """
    Y = A.astype(float, copy=True)
    if lam <= 0:
        return Y
    n = A.shape[0]
    noise = rng.poisson(lam, size=(n, n)).astype(float)
    noise = np.triu(noise, k=1)
    noise = noise + noise.T
    Y = Y + noise
    np.fill_diagonal(Y, 0.0)
    return Y


def add_gaussian_noise(A, sigma, rng):
    """Y = A + σ · Z_sym where Z_sym = (Z + Z.T) / √2, Z ~ N(0,1)^{n×n}.

    sigma = 0 leaves A unchanged.  Unlike Poisson noise, entries may be negative.
    """
    Y = A.astype(float, copy=True)
    if sigma <= 0:
        return Y
    n = A.shape[0]
    Z = rng.standard_normal((n, n))
    Z_sym = (Z + Z.T) / np.sqrt(2.0)
    np.fill_diagonal(Z_sym, 0.0)
    return Y + sigma * Z_sym


def inject_er_outliers(A, frac, p, rng):
    """Replace rows/cols of ``frac`` randomly chosen nodes by ER(p) edges. This is for outlier column test."""
    n = A.shape[0]
    n_out = int(round(frac * n))
    idx = np.sort(rng.choice(n, size=n_out, replace=False))
    inliers = np.setdiff1d(np.arange(n), idx, assume_unique=False)
    Y = A.astype(float, copy=True)

    er_oi = (rng.random((n_out, inliers.size)) < p).astype(float)
    Y[np.ix_(idx, inliers)] = er_oi
    Y[np.ix_(inliers, idx)] = er_oi.T

    er_oo = np.triu((rng.random((n_out, n_out)) < p).astype(float), k=1)
    Y[np.ix_(idx, idx)] = er_oo + er_oo.T
    np.fill_diagonal(Y, 0.0)

    mask = np.zeros(n, dtype=bool)
    mask[idx] = True
    return Y, mask


def outlier_prf(scores, true_mask):
    """Top-|true| columns by score vs the ground-truth outlier mask."""
    k = int(true_mask.sum())
    pred = np.zeros_like(true_mask)
    pred[np.argsort(scores)[::-1][:k]] = True
    tp = float(np.sum(pred & true_mask))
    prec = tp / max(float(pred.sum()), 1.0)
    rec = tp / max(float(true_mask.sum()), 1.0)
    f1 = 2.0 * prec * rec / max(prec + rec, 1e-12)
    return prec, rec, f1


CASES = {
    "1_three_block": {
        "sizes": (50, 60, 90),
        "p_in": 0.5,
        "p_out": 0.1,
        "outliers": False,
        "k": 3,
        "title": "3-block (50,60,90)  p=0.5/0.1",
    },
    "2_three_block_sparse": {
        "sizes": (50, 60, 90),
        "p_in": 0.3,
        "p_out": 0.05,
        "outliers": False,
        "k": 3,
        "title": "3-block sparse  p=0.3/0.05",
    },
    "3_five_block": {
        "sizes": (10, 20, 30, 50, 90),
        "p_in": 0.5,
        "p_out": 0.1,
        "outliers": False,
        "k": 5,
        "title": "5-block (10,20,30,50,90)  p=0.5/0.1",
    },
    "4_three_block_outliers": {
        "sizes": (50, 60, 90),
        "p_in": 0.5,
        "p_out": 0.1,
        "outliers": True,
        "outlier_frac": 0.10,
        "outlier_p": 0.5,
        "k": 3,
        "title": "3-block + 10% ER outliers",
    },
    "5_mild_three": {
        "sizes": (55, 65, 80),
        "p_in": 0.5,
        "p_out": 0.1,
        "outliers": False,
        "k": 3,
        "title": "mild 3-block (55,65,80)",
    },
    "6_unbalanced_three": {
        "sizes": (8, 32, 160),
        "p_in": 0.5,
        "p_out": 0.1,
        "outliers": False,
        "k": 3,
        "title": "unbalanced 3-block (8,32,160)",
    },
    "7_two_block": {
        "sizes": (80, 120),
        "p_in": 0.5,
        "p_out": 0.1,
        "outliers": False,
        "k": 2,
        "title": "2-block (80,120)",
    },
    "8_eight_block": {
        "sizes": (15, 18, 20, 22, 25, 28, 32, 40),
        "p_in": 0.5,
        "p_out": 0.1,
        "outliers": False,
        "k": 8,
        "title": "8-block (15–40)",
    },
    "9_hard_overlap": {
        "sizes": (50, 60, 90),
        "p_in": 0.4,
        "p_out": 0.2,
        "outliers": False,
        "k": 3,
        "title": "hard overlap  p=0.4/0.2",
    },
    "10_very_sparse": {
        "sizes": (50, 60, 90),
        "p_in": 0.15,
        "p_out": 0.03,
        "outliers": False,
        "k": 3,
        "title": "very sparse  p=0.15/0.03",
    },
    "11_dense": {
        "sizes": (50, 60, 90),
        "p_in": 0.8,
        "p_out": 0.2,
        "outliers": False,
        "k": 3,
        "title": "dense  p=0.8/0.2",
    },
    "12_weak_community": {
        "sizes": (40, 60, 100),
        "p_in": 0.25,
        "p_out": 0.12,
        "outliers": False,
        "k": 3,
        "title": "weak communities  p=0.25/0.12",
    },
    "13_severe_unbalanced": {
        "sizes": (5, 15, 180),
        "p_in": 0.5,
        "p_out": 0.1,
        "outliers": False,
        "k": 3,
        "title": "severe unbalanced (5,15,180)",
    },
    "14_twelve_block": {
        "sizes": (8, 9, 10, 12, 13, 14, 16, 17, 19, 22, 25, 35),
        "p_in": 0.5,
        "p_out": 0.1,
        "outliers": False,
        "k": 12,
        "title": "12-block (8–35)",
    },
    # ── Scalability sweep (3-block, proportional non-equal sizes, p=0.5/0.1) ──
    # TKSS initialises to K equal-length blocks; all sizes here are unequal.
    # Case 5_mild_three (N=200, sizes 55,65,80) serves as the N=200 anchor.
    "15_scale_100": {
        "sizes": (27, 33, 40),
        "p_in": 0.5,
        "p_out": 0.1,
        "outliers": False,
        "k": 3,
        "title": "scale N=100  (27,33,40)",
    },
    "16_scale_400": {
        "sizes": (108, 132, 160),
        "p_in": 0.5,
        "p_out": 0.1,
        "outliers": False,
        "k": 3,
        "title": "scale N=400  (108,132,160)",
    },
    "17_scale_800": {
        "sizes": (216, 264, 320),
        "p_in": 0.5,
        "p_out": 0.1,
        "outliers": False,
        "k": 3,
        "title": "scale N=800  (216,264,320)",
    },
    # ── Heterogeneous block model (different density per cluster pair) ────────
    # p_mat is K×K; diagonal = within-cluster density, off-diagonal = between.
    "18_hetero_3block": {
        "sizes": (50, 70, 80),
        "p_mat": [
            [0.70, 0.05, 0.02],
            [0.05, 0.40, 0.08],
            [0.02, 0.08, 0.20],
        ],
        "outliers": False,
        "k": 3,
        "title": "hetero 3-block (p_mat)",
    },
    "19_hetero_5block": {
        "sizes": (30, 40, 35, 45, 50),
        "p_mat": [
            [0.65, 0.04, 0.03, 0.02, 0.01],
            [0.04, 0.50, 0.05, 0.03, 0.02],
            [0.03, 0.05, 0.35, 0.06, 0.03],
            [0.02, 0.03, 0.06, 0.25, 0.07],
            [0.01, 0.02, 0.03, 0.07, 0.15],
        ],
        "outliers": False,
        "k": 5,
        "title": "hetero 5-block (p_mat)",
    },
    "20_hetero_sparse": {
        "sizes": (55, 65, 80),
        "p_mat": [
            [0.30, 0.02, 0.01],
            [0.02, 0.20, 0.04],
            [0.01, 0.04, 0.15],
        ],
        "outliers": False,
        "k": 3,
        "title": "hetero sparse 3-block (p_mat)",
    },
    # ── Degree-corrected SBM (heterogeneous degree within blocks) ─────────────
    # B plays the role of p_mat; theta_i ~ Gamma(theta_alpha, 1), norm per block.
    # theta_alpha=1.0: moderate heterogeneity; 0.3: heavy-tailed (hub nodes).
    "21_dcsbm_mild": {
        "sizes": (55, 65, 80),
        "dc_sbm": True,
        "B": [
            [0.50, 0.10, 0.05],
            [0.10, 0.50, 0.08],
            [0.05, 0.08, 0.50],
        ],
        "theta_alpha": 1.0,
        "outliers": False,
        "k": 3,
        "title": "DC-SBM mild (α=1.0)",
    },
    "22_dcsbm_heavy": {
        "sizes": (55, 65, 80),
        "dc_sbm": True,
        "B": [
            [0.50, 0.10, 0.05],
            [0.10, 0.50, 0.08],
            [0.05, 0.08, 0.50],
        ],
        "theta_alpha": 0.3,
        "outliers": False,
        "k": 3,
        "title": "DC-SBM heavy-tail (α=0.3)",
    },
    "23_dcsbm_hetero": {
        "sizes": (50, 70, 80),
        "dc_sbm": True,
        "B": [
            [0.70, 0.05, 0.02],
            [0.05, 0.40, 0.08],
            [0.02, 0.08, 0.20],
        ],
        "theta_alpha": 0.5,
        "outliers": False,
        "k": 3,
        "title": "DC-SBM + hetero B (α=0.5)",
    },
    # ── Gaussian noise model (λ interpreted as σ) ─────────────────────────────
    "24_gaussian_3block": {
        "sizes": (50, 60, 90),
        "p_in": 0.5,
        "p_out": 0.1,
        "noise_model": "gaussian",
        "outliers": False,
        "k": 3,
        "title": "3-block Gaussian noise  σ sweep",
    },
    "25_gaussian_sparse": {
        "sizes": (50, 60, 90),
        "p_in": 0.15,
        "p_out": 0.03,
        "noise_model": "gaussian",
        "outliers": False,
        "k": 3,
        "title": "sparse 3-block Gaussian noise  σ sweep",
    },
    # ── Large number of clusters ───────────────────────────────────────────────
    "26_twenty_block": {
        "sizes": (8, 9, 10, 11, 12, 13, 14, 15, 16, 17,
                  18, 19, 20, 21, 22, 23, 24, 25, 26, 27),
        "p_in": 0.5,
        "p_out": 0.05,
        "outliers": False,
        "k": 20,
        "title": "20-block (N=350, p=0.5/0.05)",
    },
    # ── Heavy outliers (20%) ───────────────────────────────────────────────────
    "27_heavy_outliers": {
        "sizes": (50, 60, 90),
        "p_in": 0.5,
        "p_out": 0.1,
        "outliers": True,
        "outlier_frac": 0.20,
        "outlier_p": 0.5,
        "k": 3,
        "title": "3-block + 20% ER outliers",
    },
    # ── Very weak clusters (p_in barely above p_out) ─────────────────────────
    "28_very_weak": {
        "sizes": (55, 65, 80),
        "p_in": 0.15,
        "p_out": 0.10,
        "outliers": False,
        "k": 3,
        "title": "very weak  p=0.15/0.10",
    },
    # ── Severely imbalanced + weak (stress test) ──────────────────────────────
    "29_imbalanced_weak": {
        "sizes": (10, 40, 150),
        "p_in": 0.25,
        "p_out": 0.12,
        "outliers": False,
        "k": 3,
        "title": "imbalanced+weak (10,40,150)  p=0.25/0.12",
    },
}

# LAMBDAS = (0.0, 0.05, 0.10, 0.20, 0.30, 0.50, 0.75, 1.00, 1.50, 2.00, 3.00)
LAMBDAS =  (0.0, 0.25, 0.5, 1.0, 2.0, 3.0)

# ARI vs λ is split across figures so the cases stay readable.
# Cases named "scale_*" are handled by plot_scalability (ARI/NMI vs N).
ARI_PLOT_GROUPS = [
    (
        "ari_vs_lambda.png",
        (
            "1_three_block",
            "2_three_block_sparse",
            "3_five_block",
            "4_three_block_outliers",
            "9_hard_overlap",
            "10_very_sparse",
            "11_dense",
            "12_weak_community",
        ),
    ),
    (
        "ari_vs_lambda_size.png",
        (
            "5_mild_three",
            "6_unbalanced_three",
            "7_two_block",
            "8_eight_block",
            "13_severe_unbalanced",
            "14_twelve_block",
        ),
    ),
    (
        "ari_vs_lambda_hetero.png",
        (
            "18_hetero_3block",
            "19_hetero_5block",
            "20_hetero_sparse",
            "21_dcsbm_mild",
            "22_dcsbm_heavy",
            "23_dcsbm_hetero",
        ),
    ),
    (
        "ari_vs_lambda_noise_outliers.png",
        (
            "24_gaussian_3block",
            "25_gaussian_sparse",
            "26_twenty_block",
            "27_heavy_outliers",
            "28_very_weak",
            "29_imbalanced_weak",
        ),
    ),
]

# Scalability group: these cases + 5_mild_three (N=200 anchor) are plotted
# by plot_scalability as ARI/NMI vs N at selected noise levels.
SCALE_CASES = ("15_scale_100", "5_mild_three", "16_scale_400", "17_scale_800")
SCALE_N = {
    "15_scale_100": 100,
    "5_mild_three": 200,
    "16_scale_400": 400,
    "17_scale_800": 800,
}
FIELDNAMES = [
    "case", "lambda", "trial", "method", "k",
    "ari", "nmi", "precision", "recall", "f1", "seconds", "error",
]


def parse_k_arg(value):
    """CLI / JSON k: ``'oracle'``, ``'none'`` (infer), or a positive int."""
    s = str(value).strip().lower()
    if s == "oracle":
        return "oracle"
    if s in ("none", "null"):
        return None
    try:
        k = int(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(
            f"k must be an integer, 'none', or 'oracle' (got {value!r})"
        ) from exc
    if k < 1:
        raise argparse.ArgumentTypeError("k must be >= 1")
    return k


def resolve_run_k(k, oracle_k):
    """Map CLI k onto the value passed to ``run_one``.

    ``'oracle'`` (default) uses the true SBM block count; ``None`` leaves k
    unset so the solver infers it via ``--k-method``.
    """
    if k == "oracle":
        return int(oracle_k)
    return k


def run_one(Y, labels, k, method_name, solver, outlier_mask,
            k_method="eigengap", min_k=2, penalty=0.0):
    Y = normalize_Y(Y)
    t0 = time.perf_counter()
    kind = METHOD_KIND.get(method_name)
    if kind == "tkss":
        if k is None:
            k = estimate_k_from_data(
                Y, method=k_method, min_k=min_k, penalty=penalty,
            )
        pred, residual = solver(Y, k)
        elapsed = time.perf_counter() - t0
        scores = residual
    elif kind == "bdosc":
        # Block-diagonal projection needs k during the ADMM solve.
        if k is None:
            k = estimate_k_from_data(
                Y, method=k_method, min_k=min_k, penalty=penalty,
            )
        coeff, E, F = solver(Y, k)
        elapsed = time.perf_counter() - t0
        pred = cluster_from_Z(
            coeff, k=k, method=k_method, min_k=min_k, penalty=penalty,
        )
        scores = np.linalg.norm(F if F is not None else E, axis=0)
    else:
        coeff, E, F = solver(Y)
        elapsed = time.perf_counter() - t0
        if method_name == "OSC":
            pred = cluster_from_Z(
                coeff, k=k, method=k_method, min_k=min_k, penalty=penalty,
            )
        else:
            pred = cluster_from_C(
                coeff, k=k, method=k_method, min_k=min_k, penalty=penalty,
            )
        if k is None:
            k = int(np.unique(pred).size)
        scores = np.linalg.norm(F if F is not None else E, axis=0)

    if outlier_mask is None:
        ari = float(adjusted_rand_score(labels, pred))
        nmi = float(normalized_mutual_info_score(labels, pred, average_method="arithmetic"))
        prec = rec = f1 = float("nan")
    else:
        inliers = ~outlier_mask
        ari = float(adjusted_rand_score(labels[inliers], pred[inliers]))
        nmi = float(normalized_mutual_info_score(labels[inliers], pred[inliers], average_method="arithmetic"))
        prec, rec, f1 = outlier_prf(scores, outlier_mask)

    return {
        "ari": ari,
        "nmi": nmi,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "seconds": elapsed,
        "k": int(k),
        "error": "",
    }


def make_observation(cfg, lam, rng):
    if cfg.get("dc_sbm"):
        A, labels = generate_dcsbm(
            cfg["sizes"], cfg["B"], cfg.get("theta_alpha", 0.5), rng,
        )
    elif cfg.get("p_mat") is not None:
        A, labels = generate_sbm(cfg["sizes"], rng=rng, p_mat=cfg["p_mat"])
    else:
        A, labels = generate_sbm(cfg["sizes"], cfg["p_in"], cfg["p_out"], rng)
    outlier_mask = None
    if cfg.get("outliers"):
        A, outlier_mask = inject_er_outliers(
            A, cfg["outlier_frac"], cfg["outlier_p"], rng,
        )
    noise_model = cfg.get("noise_model", "poisson")
    if noise_model == "gaussian":
        Y = add_gaussian_noise(A, lam, rng)
    else:
        Y = add_poisson_noise(A, lam, rng)
    Y = normalize_Y(Y)
    return Y, labels, outlier_mask


def run_benchmark(cases, lambdas, n_trials, methods, seed, out_dir, k="oracle",
                  k_method="eigengap", min_k=2, penalty=0.0):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "sbm_benchmark.csv"

    rows = []
    n_jobs = len(cases) * len(lambdas) * n_trials * len(methods)
    done = 0
    t_start = time.perf_counter()

    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDNAMES)
        writer.writeheader()

        for case_name in cases:
            cfg = CASES[case_name]
            for lam in lambdas:
                for trial in range(n_trials):
                    case_id = list(CASES).index(case_name)
                    rng = np.random.default_rng(
                        seed + 10_000 * case_id
                        + int(round(lam * 1000)) * 17
                        + trial
                    )
                    Y, labels, outlier_mask = make_observation(cfg, lam, rng)

                    for method_name, solver in methods:
                        rec = {
                            "case": case_name,
                            "lambda": lam,
                            "trial": trial,
                            "method": method_name,
                            "k": "",
                            "ari": float("nan"),
                            "nmi": float("nan"),
                            "precision": float("nan"),
                            "recall": float("nan"),
                            "f1": float("nan"),
                            "seconds": float("nan"),
                            "error": "",
                        }
                        try:
                            rec.update(run_one(
                                Y, labels, resolve_run_k(k, cfg["k"]),
                                method_name, solver, outlier_mask,
                                k_method=k_method, min_k=min_k, penalty=penalty,
                            ))
                        except Exception as exc:
                            rec["error"] = f"{type(exc).__name__}: {exc}"
                            traceback.print_exc()

                        writer.writerow(rec)
                        fh.flush()
                        rows.append(rec)
                        done += 1
                        elapsed = time.perf_counter() - t_start
                        eta = (elapsed / done) * (n_jobs - done) if done else 0
                        status = (
                            f"[{done}/{n_jobs}] {case_name}  λ={lam}  "
                            f"trial={trial}  {method_name}  k={rec.get('k', '')}  "
                            f"ARI={rec['ari']:.3f}"
                            if rec["error"] == ""
                            else f"[{done}/{n_jobs}] {method_name} FAILED: {rec['error']}"
                        )
                        print(f"{status}   ({elapsed:.0f}s elapsed, ETA {eta:.0f}s)",
                              flush=True)

    method_names = [m for m, _ in methods]
    write_summary(rows, out_dir, method_names)
    try:
        plot_results(rows, out_dir, method_names)
    except Exception:
        traceback.print_exc()
        print("Plotting skipped (matplotlib missing or plot error).", flush=True)
    return rows


def _finite(vals):
    arr = np.asarray(vals, dtype=float)
    arr = arr[np.isfinite(arr)]
    return arr


def write_summary(rows, out_dir, method_names=None):
    path = out_dir / "sbm_benchmark_summary.csv"
    keys = sorted({(r["case"], r["lambda"], r["method"]) for r in rows})
    with path.open("w", newline="") as fh:
        fields = [
            "case", "lambda", "method", "n",
            "ari_mean", "ari_std",
            "nmi_mean", "nmi_std",
            "precision_mean", "precision_std",
            "recall_mean", "recall_std",
            "f1_mean", "f1_std",
            "seconds_mean",
        ]
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for case, lam, method in keys:
            sub = [r for r in rows
                   if r["case"] == case and r["lambda"] == lam
                   and r["method"] == method and r["error"] == ""]
            ari = _finite([r["ari"] for r in sub])
            nmi = _finite([r.get("nmi", float("nan")) for r in sub])
            prec = _finite([r["precision"] for r in sub])
            rec = _finite([r["recall"] for r in sub])
            f1 = _finite([r["f1"] for r in sub])
            sec = _finite([r["seconds"] for r in sub])

            def mean_std(a):
                if a.size == 0:
                    return float("nan"), float("nan")
                return float(a.mean()), float(a.std(ddof=1) if a.size > 1 else 0.0)

            ari_m, ari_s = mean_std(ari)
            nmi_m, nmi_s = mean_std(nmi)
            p_m, p_s = mean_std(prec)
            r_m, r_s = mean_std(rec)
            f_m, f_s = mean_std(f1)
            s_m, _ = mean_std(sec)
            writer.writerow({
                "case": case, "lambda": lam, "method": method, "n": len(sub),
                "ari_mean": ari_m, "ari_std": ari_s,
                "nmi_mean": nmi_m, "nmi_std": nmi_s,
                "precision_mean": p_m, "precision_std": p_s,
                "recall_mean": r_m, "recall_std": r_s,
                "f1_mean": f_m, "f1_std": f_s,
                "seconds_mean": s_m,
            })
    print(f"Wrote {path}", flush=True)

    # Compact stdout table: ARI mean ± std and NMI mean per case, per λ.
    print("\n=== ARI mean ± std  /  NMI mean (over trials, per λ) ===", flush=True)
    methods = method_names or [s["name"] for s in METHOD_SPECS]
    present = {r["case"] for r in rows}
    case_order = [c for c in CASES if c in present] or sorted(present)
    for case in case_order:
        print(f"\n{case}", flush=True)
        header = f"{'λ':>8}" + "".join(f"{m:>22}" for m in methods)
        print(header, flush=True)
        lams = sorted({r["lambda"] for r in rows if r["case"] == case})
        for lam in lams:
            cells = [f"{lam:8.2f}"]
            for method in methods:
                sub = [r for r in rows
                       if r["case"] == case and r["lambda"] == lam
                       and r["method"] == method and r["error"] == ""]
                a = _finite([r["ari"] for r in sub])
                nm = _finite([r.get("nmi", float("nan")) for r in sub])
                if a.size == 0:
                    cells.append(f"{'n/a':>22}")
                else:
                    m = a.mean()
                    s = a.std(ddof=1) if a.size > 1 else 0.0
                    nmi_str = f"{nm.mean():.3f}" if nm.size else "n/a"
                    cells.append(f"{m:6.3f}±{s:<5.3f}NMI{nmi_str}".rjust(22))
            print("".join(cells), flush=True)

    outlier_cases = [c for c in ["4_three_block_outliers", "27_heavy_outliers"]
                     if any(r["case"] == c for r in rows)]
    for outlier_case in outlier_cases:
        outlier_rows = [r for r in rows if r["case"] == outlier_case]
        print(f"\n=== {outlier_case} outlier F1 mean ± std ===", flush=True)
        header = f"{'λ':>8}" + "".join(f"{m:>22}" for m in methods)
        print(header, flush=True)
        lams = sorted({r["lambda"] for r in outlier_rows})
        for lam in lams:
            cells = [f"{lam:8.2f}"]
            for method in methods:
                sub = [r["f1"] for r in outlier_rows
                       if r["lambda"] == lam and r["method"] == method
                       and r["error"] == ""]
                a = _finite(sub)
                if a.size == 0:
                    cells.append(f"{'n/a':>22}")
                else:
                    m = a.mean()
                    s = a.std(ddof=1) if a.size > 1 else 0.0
                    cells.append(f"{m:8.3f} ± {s:<6.3f}".rjust(22))
            print("".join(cells), flush=True)

    print("\n=== Runtime mean ± std (seconds, pooled over cases / λ / trials) ===",
          flush=True)
    header = f"{'method':24s} {'n':>4} {'mean':>8} {'std':>8} {'median':>8} {'min':>8} {'max':>8}"
    print(header, flush=True)
    for method in methods:
        a = _finite([r["seconds"] for r in rows
                     if r["method"] == method and r["error"] == ""])
        if a.size == 0:
            continue
        std = float(a.std(ddof=1)) if a.size > 1 else 0.0
        print(
            f"{method:24s} {a.size:4d} {a.mean():8.3f} {std:8.3f} "
            f"{float(np.median(a)):8.3f} {a.min():8.3f} {a.max():8.3f}",
            flush=True,
        )


def _plot_ari_grid(rows, cases, methods, markers, out_path):
    """One ARI-vs-λ figure for a subset of cases."""
    import matplotlib.pyplot as plt

    n = len(cases)
    if n == 0:
        return
    ncols = 3 if n > 4 else 2
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(4.4 * ncols, 3.15 * nrows), sharey=True,
    )
    axes = np.atleast_1d(axes).ravel()
    for ax in axes[n:]:
        ax.set_visible(False)

    for ax, case in zip(axes, cases):
        lams = sorted({r["lambda"] for r in rows if r["case"] == case})
        for i, method in enumerate(methods):
            means, stds = [], []
            for lam in lams:
                a = _finite([
                    r["ari"] for r in rows
                    if r["case"] == case and r["lambda"] == lam
                    and r["method"] == method and r["error"] == ""
                ])
                means.append(a.mean() if a.size else np.nan)
                stds.append(a.std(ddof=1) if a.size > 1 else 0.0)
            ax.errorbar(
                lams, means, yerr=stds, marker=markers[i % len(markers)],
                capsize=3, label=method, linewidth=1.4,
            )
        title = CASES.get(case, {}).get("title", case)
        ax.set_title(title, fontsize=10)
        noise_label = ("Gaussian σ"
                       if CASES.get(case, {}).get("noise_model") == "gaussian"
                       else "Poisson rate λ")
        ax.set_xlabel(noise_label)
        ax.set_ylabel("ARI")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, 1.01))
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}", flush=True)


def plot_scalability(rows, out_dir, methods, markers, lambdas_to_show=(0.0, 0.5, 1.0)):
    """ARI and NMI vs N for SCALE_CASES at selected noise levels."""
    import matplotlib.pyplot as plt

    present = {r["case"] for r in rows}
    scale_cases = [c for c in SCALE_CASES if c in present]
    if not scale_cases:
        return

    ns = [SCALE_N[c] for c in scale_cases]
    lams_present = sorted({r["lambda"] for r in rows
                           if r["case"] in scale_cases})
    show_lams = [lam for lam in lambdas_to_show if lam in lams_present]
    if not show_lams:
        show_lams = lams_present[:3]

    for metric, ylabel, fname in [
        ("ari", "ARI", "scalability_ari_vs_n.png"),
        ("nmi", "NMI", "scalability_nmi_vs_n.png"),
    ]:
        fig, axes = plt.subplots(
            1, len(show_lams),
            figsize=(4.8 * len(show_lams), 4.0),
            sharey=True,
        )
        axes = np.atleast_1d(axes).ravel()
        for ax, lam in zip(axes, show_lams):
            for i, method in enumerate(methods):
                vals = []
                for case in scale_cases:
                    a = _finite([
                        r[metric] for r in rows
                        if r["case"] == case and r["lambda"] == lam
                        and r["method"] == method and r["error"] == ""
                    ])
                    vals.append(a.mean() if a.size else np.nan)
                ax.plot(ns, vals, marker=markers[i % len(markers)],
                        linewidth=1.4, label=method)
            ax.set_title(f"Poisson λ={lam}", fontsize=10)
            ax.set_xlabel("N (number of nodes)")
            ax.set_ylabel(ylabel)
            ax.set_ylim(-0.05, 1.05)
            ax.grid(True, alpha=0.3)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False,
                   bbox_to_anchor=(0.5, 1.04))
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        out_path = Path(out_dir) / fname
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Wrote {out_path}", flush=True)


def plot_results(rows, out_dir, method_names=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    methods = method_names or [s["name"] for s in METHOD_SPECS]
    markers = ["o", "s", "D", "^", "v", "P", "X", "*"]
    present = {r["case"] for r in rows}
    grouped = set()
    for filename, group in ARI_PLOT_GROUPS:
        cases = [c for c in group if c in present]
        grouped.update(cases)
        _plot_ari_grid(rows, cases, methods, markers, Path(out_dir) / filename)

    leftover = [c for c in CASES if c in present and c not in grouped]
    leftover += [c for c in sorted(present) if c not in grouped and c not in CASES]
    if leftover:
        _plot_ari_grid(
            rows, leftover, methods, markers,
            Path(out_dir) / "ari_vs_lambda_other.png",
        )

    for outlier_case in ["4_three_block_outliers", "27_heavy_outliers"]:
        case_rows = [r for r in rows if r["case"] == outlier_case]
        if not case_rows:
            continue
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), sharey=True)
        for ax, metric, title in zip(
            axes,
            ("precision", "recall", "f1"),
            ("Outlier precision", "Outlier recall", "Outlier F1"),
        ):
            lams = sorted({r["lambda"] for r in case_rows})
            for i, method in enumerate(methods):
                means, stds = [], []
                for lam in lams:
                    a = _finite([
                        r[metric] for r in case_rows
                        if r["lambda"] == lam and r["method"] == method
                        and r["error"] == ""
                    ])
                    means.append(a.mean() if a.size else np.nan)
                    stds.append(a.std(ddof=1) if a.size > 1 else 0.0)
                ax.errorbar(
                    lams, means, yerr=stds, marker=markers[i % len(markers)],
                    capsize=3, label=method, linewidth=1.4,
                )
            ax.set_title(title)
            ax.set_xlabel("Poisson rate λ")
            ax.set_ylim(-0.05, 1.05)
            ax.grid(True, alpha=0.3)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False,
                   bbox_to_anchor=(0.5, 1.08))
        fig.tight_layout()
        out_path = Path(out_dir) / f"{outlier_case}_detection.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Wrote {out_path}", flush=True)

    plot_scalability(rows, out_dir, methods, markers)
    plot_runtime(rows, out_dir, methods)


def plot_runtime(rows, out_dir, methods):
    """Mean wall time per method, and ARI vs time (quality–cost)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    means, stds, aris = [], [], []
    for method in methods:
        t = _finite([r["seconds"] for r in rows
                     if r["method"] == method and r["error"] == ""])
        a = _finite([r["ari"] for r in rows
                     if r["method"] == method and r["error"] == ""])
        means.append(float(t.mean()) if t.size else np.nan)
        stds.append(float(t.std(ddof=1)) if t.size > 1 else 0.0)
        aris.append(float(a.mean()) if a.size else np.nan)

    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    x = np.arange(len(methods))
    ax.bar(x, means, yerr=stds, capsize=3, width=0.7, color="0.45")
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=25, ha="right")
    ax.set_ylabel("Wall time (s)")
    ax.set_title("Clustering, 200×200")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    bar_path = out_dir / "runtime_by_method.png"
    fig.savefig(bar_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {bar_path}", flush=True)

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    markers = ["o", "s", "D", "^", "v", "P", "X", "*"]
    for i, method in enumerate(methods):
        ax.scatter(
            means[i], aris[i], s=70, marker=markers[i % len(markers)],
            label=method, zorder=3,
        )
        ax.annotate(method, (means[i], aris[i]),
                    textcoords="offset points", xytext=(6, 4), fontsize=8)
    ax.set_xlabel("Mean wall time (s)")
    ax.set_ylabel("Mean ARI (all cases, λ, trials)")
    ax.set_title("Quality vs cost")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    scatter_path = out_dir / "ari_vs_runtime.png"
    fig.savefig(scatter_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {scatter_path}", flush=True)

    # Time vs noise: pooled over cases/trials.
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    lams = sorted({r["lambda"] for r in rows})
    for i, method in enumerate(methods):
        tmeans = []
        for lam in lams:
            t = _finite([
                r["seconds"] for r in rows
                if r["method"] == method and r["lambda"] == lam
                and r["error"] == ""
            ])
            tmeans.append(t.mean() if t.size else np.nan)
        ax.plot(lams, tmeans, marker=markers[i % len(markers)],
                linewidth=1.4, label=method)
    ax.set_xlabel("Poisson rate λ")
    ax.set_ylabel("Mean wall time (s)")
    ax.set_title("Runtime vs noise level")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", ncol=2, frameon=False, fontsize=8)
    fig.tight_layout()
    vs_path = out_dir / "runtime_vs_lambda.png"
    fig.savefig(vs_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {vs_path}", flush=True)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--trials", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", type=str, default=str(ROOT / "results"))
    p.add_argument(
        "--cases", nargs="+", default=list(CASES),
        choices=list(CASES),
    )
    p.add_argument(
        "--methods", nargs="+", default=list(DEFAULT_METHODS),
        choices=[s["name"] for s in METHOD_SPECS],
    )
    p.add_argument(
        "--lambdas", type=float, nargs="+", default=list(LAMBDAS),
    )
    p.add_argument(
        "--smoke", action="store_true",
        help="One trial, λ=0, case 1 only — check imports and a single ADMM run.",
    )
    p.add_argument(
        "--k", type=parse_k_arg, default="oracle",
        help="Cluster count: 'oracle' (default, true SBM k), 'none' (infer "
             "via --k-method), or a positive integer.",
    )
    p.add_argument(
        "--k-method", type=str, default="eigengap",
        choices=["eigengap", "ncut"],
        help="How to infer k when --k none (default: eigengap).",
    )
    p.add_argument(
        "--min-k", type=int, default=2,
        help="Minimum k when inferring (default 2; avoids trivial eigengap at 1).",
    )
    p.add_argument(
        "--k-penalty", type=float, default=0.0,
        help="Optional linear penalty for method=ncut (cost + penalty*k).",
    )
    p.add_argument(
        "--params", type=str, default=None,
        help="JSON from tune_hyperparams.py with per-method 'params' dicts.",
    )
    p.add_argument(
        "--plot-only", action="store_true",
        help="Recompute summary tables and plots from out-dir/sbm_benchmark.csv.",
    )
    return p.parse_args()


def load_benchmark_csv(path):
    rows = []
    with Path(path).open(newline="") as fh:
        for r in csv.DictReader(fh):
            rec = dict(r)
            rec["lambda"] = float(rec["lambda"])
            rec["trial"] = int(rec["trial"])
            if rec.get("k") not in (None, "", "nan"):
                rec["k"] = int(float(rec["k"]))
            for key in ("ari", "nmi", "precision", "recall", "f1", "seconds"):
                rec[key] = float(rec[key]) if rec[key] not in ("", "nan") else float("nan")
            rows.append(rec)
    return rows


def main():
    args = parse_args()
    if args.plot_only:
        out_dir = Path(args.out_dir)
        csv_path = out_dir / "sbm_benchmark.csv"
        rows = load_benchmark_csv(csv_path)
        names = []
        for spec in METHOD_SPECS:
            if any(r["method"] == spec["name"] for r in rows):
                names.append(spec["name"])
        write_summary(rows, out_dir, names)
        plot_results(rows, out_dir, names)
        return

    cases = args.cases
    lambdas = args.lambdas
    n_trials = args.trials
    if args.smoke:
        cases = ["1_three_block"]
        lambdas = [0.0]
        n_trials = 1
        print("Smoke mode: case 1, λ=0, 1 trial, all methods.", flush=True)

    overrides = load_param_overrides(args.params) if args.params else {}
    if overrides:
        print(f"Loaded hyperparameters from {args.params}", flush=True)
        for name in args.methods:
            if name in overrides:
                print(f"  {name}: {overrides[name]}", flush=True)
    methods = build_methods(overrides=overrides, names=args.methods)
    n = len(cases) * len(lambdas) * n_trials * len(methods)
    if args.k is None:
        k_s = f"none ({args.k_method}, min_k={args.min_k})"
    else:
        k_s = str(args.k)
    print(
        f"Running {n} jobs  "
        f"({len(cases)} cases × {len(lambdas)} λ × {n_trials} trials "
        f"× {len(methods)} methods)  k={k_s}",
        flush=True,
    )
    run_benchmark(
        cases, lambdas, n_trials, methods, args.seed, args.out_dir, k=args.k,
        k_method=args.k_method, min_k=args.min_k, penalty=args.k_penalty,
    )


if __name__ == "__main__":
    main()
