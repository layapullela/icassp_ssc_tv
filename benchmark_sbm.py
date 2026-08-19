"""
SBM benchmark of OSC, six SSC-TV ADMM variants, and Temporal K-Subspaces
(TKSS).  TKSS is a sequential K-subspaces method: it returns labels
directly (no coefficient matrix / spectral clustering).  Subspace
dimension d, sequential weight λ, and neighbor window s are the
tunable knobs; k is the true number of SBM blocks.

Protocol
--------
200×200 undirected SBM test cases, Poisson observation noise with
rate λ ∈ {0, 0.05, 0.10, 0.20, 0.30, 0.50, 0.75, 1.00}
(λ = 0 is the noiseless adjacency), 5 independent draws per (case, λ).
Spectral clustering (OSC / SSC-TV) uses the true number of SBM blocks k
on W = |C| + |C|^T (OSC: |Z| + |Z|^T).  TKSS is given the same k.

Cases
-----
Original
  1. 3 contiguous blocks (50, 60, 90), p_in=0.5, p_out=0.1
  2. Same partition, sparse: p_in=0.3, p_out=0.05
  3. 5 contiguous blocks (30, 35, 40, 45, 50), p_in=0.5, p_out=0.1
  4. Case 1 with 10% ER p=0.5 outlier nodes (ARI on inliers)

Size sweep (p_in=0.5, p_out=0.1)
  5. Equal 3-block (67, 67, 66)
  6. Unbalanced 3-block (20, 50, 130)
  7. Two-block (100, 100)
  8. Eight-block (8 × 25)

Probability sweep (sizes 50, 60, 90)
  9.  Hard overlap: p_in=0.4, p_out=0.2
 10. Very sparse: p_in=0.15, p_out=0.03
 11. Dense: p_in=0.8, p_out=0.2
 12. Weak communities: p_in=0.25, p_out=0.12

Each observation matrix Y is Frobenius-normalised before clustering or
tuning.  SSC-TV variants fix λ_z = 1 and tune the remaining penalties
relative to it; λ_e21 stored in JSON is the pre-scale value and is
multiplied by √N at call time (N = number of columns of Y).

ADMM solvers are imported from the existing modules; SSC-TV max_iter is
raised to 200 to match OSC.  TKSS uses alternating subspace / assignment
updates (default 50 iters, early stop).  File defaults are used unless
``--params`` points at a JSON file written by ``tune_hyperparams.py``
(Optuna TPE, one vector per method).

Usage
-----
    python tune_hyperparams.py                  # Optuna TPE, all 8 methods
    python tune_hyperparams.py --retune OSC     # one method; merge into JSON
    python benchmark_sbm.py --params results/best_hyperparams.json --out-dir results/tuned
    python benchmark_sbm.py --trials 5 --out-dir results
    python benchmark_sbm.py --smoke             # one trial, λ=0, case 1 only
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
from sklearn.metrics import adjusted_rand_score

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from osc import cluster_from_Z, osc_exact  # noqa: E402
from ssc_tv import cluster_from_C, ssc_admm_nuc_tv as ssc_tv_admm  # noqa: E402
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


# lambda_z is fixed at 1; other ADMM penalties are tuned relative to it.
# lambda_e21 in defaults / JSON is the *pre-scale* value; make_solver multiplies
# by sqrt(N) at call time (N = number of columns of Y).
LAMBDA_Z = 1.0
SSC_DEFAULTS = dict(lambda_e=1.0, lambda_z=LAMBDA_Z, gamma=0.1, mu=1.0, sigma=1.0,
                    max_iter=200, tol=1e-4)
E1E21_DEFAULTS = dict(lambda_e1=1.0, lambda_e21=1.0, lambda_z=LAMBDA_Z, gamma=0.1,
                      mu=1.0, sigma=1.0, rho=1.0, max_iter=200, tol=1e-4)
# osc_exact (SubKit / osc.m): λ1 on ||Z||_1, λ2 on ||ZR||_{1,2}, mu default 0.1.
# Tuner / JSON still use lambda1/lambda2; make_solver maps them onto lambda_1/lambda_2.
OSC_DEFAULTS = dict(lambda1=0.1, lambda2=1.0, mu=0.1, max_iter=200)
TKSS_DEFAULTS = dict(lam=1.0, s=1, d=1, max_iter=200, random_state=0)

# Back-compat aliases used by older call sites / the tuner.
SSC_KW = SSC_DEFAULTS
E1E21_KW = E1E21_DEFAULTS
OSC_KW = OSC_DEFAULTS

METHOD_SPECS = [
    dict(name="OSC", kind="osc", solver=None, defaults=OSC_DEFAULTS),
    dict(name="SSC-TV", kind="ssc", solver=ssc_tv_admm, defaults=SSC_DEFAULTS),
    dict(name="SSC-TV-L21-P", kind="ssc", solver=_mod_l21_p.ssc_admm_nuc_tv,
         defaults=SSC_DEFAULTS),
    dict(name="SSC-TV-L21-PQ", kind="ssc", solver=_mod_l21_pq.ssc_admm_nuc_tv,
         defaults=SSC_DEFAULTS),
    dict(name="SSC-TV-E1E21", kind="ssc", solver=_mod_e1e21.ssc_admm_nuc_tv_e1_e21,
         defaults=E1E21_DEFAULTS),
    dict(name="SSC-TV-E1E21-L21-P", kind="ssc",
         solver=_mod_e1e21_l21_p.ssc_admm_nuc_tv_e1_e21, defaults=E1E21_DEFAULTS),
    dict(name="SSC-TV-E1E21-L21-PQ", kind="ssc",
         solver=_mod_e1e21_l21_pq.ssc_admm_nuc_tv_e1_e21, defaults=E1E21_DEFAULTS),
    dict(name="TKSS", kind="tkss", solver=tkss_cluster, defaults=TKSS_DEFAULTS),
]

METHOD_KIND = {s["name"]: s["kind"] for s in METHOD_SPECS}


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
    """Unit-Frobenius scale so ADMM penalties stay relative to ``lambda_z=1``."""
    Y = np.asarray(Y, dtype=float)
    scale = np.linalg.norm(Y, "fro")
    if not np.isfinite(scale) or scale == 0.0:
        return Y
    return Y / scale


def _ssc_call_kwargs(kw, Y):
    """Fix ``lambda_z=1`` and scale ``lambda_e21`` by ``sqrt(N)`` (pre-scale in kw)."""
    call_kw = dict(kw)
    call_kw["lambda_z"] = LAMBDA_Z
    if "lambda_e21" in call_kw:
        n_cols = int(Y.shape[1])
        call_kw["lambda_e21"] = float(call_kw["lambda_e21"]) * np.sqrt(n_cols)
    return call_kw


def make_solver(spec, kwargs):
    """Bind a solver spec to a kwargs dict.

    ADMM methods (OSC / SSC-TV): ``fn(Y) -> (coeff, E, F)``.
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
    if spec["kind"] == "tkss":
        def fn(Y, k, _kw=kw):
            d = int(round(_kw.get("d", 1)))
            s = int(round(_kw.get("s", 1)))
            return tkss_cluster(
                Y,
                k=int(k),
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
        if max_iter is not None and spec["kind"] != "tkss":
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

def generate_sbm(cluster_sizes, p_in, p_out, rng):
    """Undirected SBM: Bernoulli upper triangle, symmetrised, zero diagonal."""
    labels = np.repeat(np.arange(len(cluster_sizes)), cluster_sizes)
    n = labels.size
    same = labels[:, None] == labels[None, :]
    probs = np.where(same, p_in, p_out)
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
        "sizes": (30, 35, 40, 45, 50),
        "p_in": 0.5,
        "p_out": 0.1,
        "outliers": False,
        "k": 5,
        "title": "5-block  p=0.5/0.1",
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
    "5_equal_three": {
        "sizes": (67, 67, 66),
        "p_in": 0.5,
        "p_out": 0.1,
        "outliers": False,
        "k": 3,
        "title": "equal 3-block (67,67,66)",
    },
    "6_unbalanced_three": {
        "sizes": (20, 50, 130),
        "p_in": 0.5,
        "p_out": 0.1,
        "outliers": False,
        "k": 3,
        "title": "unbalanced 3-block (20,50,130)",
    },
    "7_two_block": {
        "sizes": (100, 100),
        "p_in": 0.5,
        "p_out": 0.1,
        "outliers": False,
        "k": 2,
        "title": "2-block (100,100)",
    },
    "8_eight_block": {
        "sizes": (25, 25, 25, 25, 25, 25, 25, 25),
        "p_in": 0.5,
        "p_out": 0.1,
        "outliers": False,
        "k": 8,
        "title": "8-block (8×25)",
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
        "sizes": (50, 60, 90),
        "p_in": 0.25,
        "p_out": 0.12,
        "outliers": False,
        "k": 3,
        "title": "weak communities  p=0.25/0.12",
    },
}

LAMBDAS = (0.0, 0.05, 0.10, 0.20, 0.30, 0.50, 0.75, 1.00)
FIELDNAMES = [
    "case", "lambda", "trial", "method",
    "ari", "precision", "recall", "f1", "seconds", "error",
]


def run_one(Y, labels, k, method_name, solver, outlier_mask):
    Y = normalize_Y(Y)
    t0 = time.perf_counter()
    if METHOD_KIND.get(method_name) == "tkss":
        pred, residual = solver(Y, k)
        elapsed = time.perf_counter() - t0
        scores = residual
    else:
        coeff, E, F = solver(Y)
        elapsed = time.perf_counter() - t0
        if method_name == "OSC":
            pred = cluster_from_Z(coeff, k=k)
        else:
            pred = cluster_from_C(coeff, k=k)
        scores = np.linalg.norm(F if F is not None else E, axis=0)

    if outlier_mask is None:
        ari = float(adjusted_rand_score(labels, pred))
        prec = rec = f1 = float("nan")
    else:
        inliers = ~outlier_mask
        ari = float(adjusted_rand_score(labels[inliers], pred[inliers]))
        prec, rec, f1 = outlier_prf(scores, outlier_mask)

    return {
        "ari": ari,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "seconds": elapsed,
        "error": "",
    }


def make_observation(cfg, lam, rng):
    A, labels = generate_sbm(cfg["sizes"], cfg["p_in"], cfg["p_out"], rng)
    outlier_mask = None
    if cfg.get("outliers"):
        A, outlier_mask = inject_er_outliers(
            A, cfg["outlier_frac"], cfg["outlier_p"], rng,
        )
    Y = add_poisson_noise(A, lam, rng)
    Y = normalize_Y(Y)
    return Y, labels, outlier_mask


def run_benchmark(cases, lambdas, n_trials, methods, seed, out_dir):
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
                            "ari": float("nan"),
                            "precision": float("nan"),
                            "recall": float("nan"),
                            "f1": float("nan"),
                            "seconds": float("nan"),
                            "error": "",
                        }
                        try:
                            rec.update(run_one(
                                Y, labels, cfg["k"],
                                method_name, solver, outlier_mask,
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
                            f"trial={trial}  {method_name}  "
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
            prec = _finite([r["precision"] for r in sub])
            rec = _finite([r["recall"] for r in sub])
            f1 = _finite([r["f1"] for r in sub])
            sec = _finite([r["seconds"] for r in sub])

            def mean_std(a):
                if a.size == 0:
                    return float("nan"), float("nan")
                return float(a.mean()), float(a.std(ddof=1) if a.size > 1 else 0.0)

            ari_m, ari_s = mean_std(ari)
            p_m, p_s = mean_std(prec)
            r_m, r_s = mean_std(rec)
            f_m, f_s = mean_std(f1)
            s_m, _ = mean_std(sec)
            writer.writerow({
                "case": case, "lambda": lam, "method": method, "n": len(sub),
                "ari_mean": ari_m, "ari_std": ari_s,
                "precision_mean": p_m, "precision_std": p_s,
                "recall_mean": r_m, "recall_std": r_s,
                "f1_mean": f_m, "f1_std": f_s,
                "seconds_mean": s_m,
            })
    print(f"Wrote {path}", flush=True)

    # Compact stdout table: ARI mean ± std per case, averaged over λ, then
    # the full per-λ grid.
    print("\n=== ARI mean ± std (over trials, per λ) ===", flush=True)
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
                sub = [r["ari"] for r in rows
                       if r["case"] == case and r["lambda"] == lam
                       and r["method"] == method and r["error"] == ""]
                a = _finite(sub)
                if a.size == 0:
                    cells.append(f"{'n/a':>22}")
                else:
                    m = a.mean()
                    s = a.std(ddof=1) if a.size > 1 else 0.0
                    cells.append(f"{m:8.3f} ± {s:<6.3f}".rjust(22))
            print("".join(cells), flush=True)

    outlier_rows = [r for r in rows if r["case"] == "4_three_block_outliers"]
    if outlier_rows:
        print("\n=== Case 4 outlier F1 mean ± std ===", flush=True)
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


def plot_results(rows, out_dir, method_names=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    methods = method_names or [s["name"] for s in METHOD_SPECS]
    markers = ["o", "s", "D", "^", "v", "P", "X", "*"]
    cases = [c for c in CASES if c in {r["case"] for r in rows}]
    if not cases:
        cases = sorted({r["case"] for r in rows})

    n = len(cases)
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
        ax.set_xlabel("Poisson rate λ")
        ax.set_ylabel("ARI")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, 1.01))
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    ari_path = out_dir / "ari_vs_lambda.png"
    fig.savefig(ari_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {ari_path}", flush=True)

    case4 = [r for r in rows if r["case"] == "4_three_block_outliers"]
    if case4:
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), sharey=True)
        for ax, metric, title in zip(
            axes,
            ("precision", "recall", "f1"),
            ("Outlier precision", "Outlier recall", "Outlier F1"),
        ):
            lams = sorted({r["lambda"] for r in case4})
            for i, method in enumerate(methods):
                means, stds = [], []
                for lam in lams:
                    a = _finite([
                        r[metric] for r in case4
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
        out_path = out_dir / "case4_outlier_detection.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Wrote {out_path}", flush=True)

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
    p.add_argument("--trials", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", type=str, default=str(ROOT / "results"))
    p.add_argument(
        "--cases", nargs="+", default=list(CASES),
        choices=list(CASES),
    )
    p.add_argument(
        "--methods", nargs="+", default=[m for m, _ in METHODS],
        choices=[m for m, _ in METHODS],
    )
    p.add_argument(
        "--lambdas", type=float, nargs="+", default=list(LAMBDAS),
    )
    p.add_argument(
        "--smoke", action="store_true",
        help="One trial, λ=0, case 1 only — check imports and a single ADMM run.",
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
            for key in ("ari", "precision", "recall", "f1", "seconds"):
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
    print(
        f"Running {n} jobs  "
        f"({len(cases)} cases × {len(lambdas)} λ × {n_trials} trials "
        f"× {len(methods)} methods)",
        flush=True,
    )
    run_benchmark(cases, lambdas, n_trials, methods, args.seed, args.out_dir)


if __name__ == "__main__":
    main()
