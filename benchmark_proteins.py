"""
Protein-multimer subunit recovery: DP-Y vs OSC vs SSC-TV-L21-PQ vs TKSS.

Each Biological Assembly 1 is turned into a Cα contact graph (default 8 Å).
Residues are ordered by chain, then sequence id — the same contiguous-block
layout OSC / SSC-TV / TKSS assume.  Ground-truth labels are polymer chain IDs.
Spectral clustering (OSC / SSC-TV) uses the true number of remaining chains k
on W = |C| + |C|^T (OSC: |Z| + |Z|^T).  TKSS is given the same k and returns
labels directly.  DP-Y is the trivial baseline: the same contiguous DP NCut
on the contact matrix Y itself (no coefficient matrix).  Pass ``--k none``
to drop that oracle k and infer it (``--k-method eigengap|ncut``); the true
subunit labels are then used only to score ARI / NMI.

Tiny peptide-sized chains are dropped (default min length 20) so k matches
the subunits we actually want to recover.  Large assemblies are uniformly
subsampled to --max-n residues so the N×N ADMM stays tractable.

Protocol
--------
Assemblies are split 30/70 *within each oligomer class* (seeded): 30% for
hyperparameter search, 70% held-out test.  Each contact matrix Y has
columns scaled to unit ℓ2 norm.  OSC, SSC-TV-L21-PQ, and TKSS
hyperparameters are selected on the tune split by mean ARI with Optuna TPE
(log-uniform [1e-4, 10]; TKSS: λ same range; s, d integer).  SSC-TV λ_z is
fixed at 1.  SSC-TV-L21-PQ tunes λ_e and γ (no λ_e21).  The held-out split
is the only number used to report protein contact-graph recovery.  Pass
``--params`` to reuse a previous split / already-tuned methods and search
only the methods that are still missing.

Reporting
---------
TKSS is initialised to K equal-length contiguous blocks, so equal-size
subunits (chain lengths differ by at most 1 residue) match that init and
are not a fair TKSS test.  Printed tables and plots therefore split:

  * equal-size subunits  — OSC / SSC-TV only
  * unequal-size subunits — OSC / SSC-TV plus TKSS as a comparison

Usage
-----
    python benchmark_proteins.py --smoke
    python benchmark_proteins.py --out-dir results/proteins_tuned
    python benchmark_proteins.py --n-trials 40 --out-dir results/proteins_optuna
    python benchmark_proteins.py --params results/proteins_tuned/best_hyperparams.json
    python benchmark_proteins.py --k none --k-method eigengap --min-k 2 \\
        --tune-frac 0.3 --methods OSC SSC-TV-L21-PQ TKSS \\
        --out-dir results/proteins-unk-k-eigengap-l21pq
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import traceback
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import optuna
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "protein_data") not in sys.path:
    sys.path.insert(0, str(ROOT / "protein_data"))

from osc import cluster_from_Z  # noqa: E402
from ssc_tv import cluster_from_C, estimate_k_from_data  # noqa: E402
from visualize_contact_maps import (  # noqa: E402
    contact_adjacency,
    extract_ca_coords,
    load_metadata,
    pairwise_distances,
)

import benchmark_sbm as bench  # noqa: E402

METHOD_CHOICES = [
    "DP-Y",
    "OSC", "BDOSC", "SSC-TV", "SSC-TV-L21-PQ", "SSC-TV-L21-PQ-SparseC",
    "SSC-TV-L21-PQ-LowRankC",
    "SSC-TV-E1E21-L21-P", "SSC-TV-E1E21-L21-PQ", "TKSS",
]
DEFAULT_METHODS = ["OSC", "SSC-TV-L21-PQ", "TKSS"]
OLIGOMER_ORDER = [
    "dimer", "trimer", "tetramer", "pentamer", "hexamer", "heptamer", "octamer",
]
SPLIT_SEED = 0
TUNE_FRAC = 0.3
TUNE_MAX_ITER = 200
N_TRIALS = 40
N_STARTUP_TRIALS = 10

# Log-uniform ranges.  λ_z is fixed at 1 (not searched).
# E1E21 variants also tune λ_e21 (pre-√N scale); L21-PQ tunes λ_e and γ
# with γ floored at 1e-2; L21-PQ-SparseC additionally tunes λ_c (||C||_1);
# L21-PQ-LowRankC tunes λ_c as the nuclear-norm weight (||C||_*).
# BDOSC tunes λ1/λ2 (like OSC) plus ADMM γ₁ and growth factor p.
PARAM_RANGE = (1e-3, 10.0)
GAMMA_RANGE = (1e-2, 10.0)
SEARCH_SPACES = {
    "OSC": {
        "lambda1": PARAM_RANGE,
        "lambda2": PARAM_RANGE,
    },
    "BDOSC": {
        "lambda1": PARAM_RANGE,
        "lambda2": PARAM_RANGE,
        "gamma1": PARAM_RANGE,
        "p": (1.01, 1.5),
    },
    "SSC-TV": {
        "lambda_e": PARAM_RANGE,
        "gamma": GAMMA_RANGE,
    },
    "SSC-TV-L21-PQ": {
        "lambda_e": PARAM_RANGE,
        "gamma": GAMMA_RANGE,
    },
    "SSC-TV-E1E21": {
        "lambda_e1": PARAM_RANGE,
        "lambda_e21": PARAM_RANGE,
        "gamma": GAMMA_RANGE,
    },
    "TKSS": {
        "lam": PARAM_RANGE,
        "s": (1, 8),
        "d": (1, 6),
    },
}
SEARCH_SPACES["SSC-TV-L21-PQ-SparseC"] = {
    "lambda_e": PARAM_RANGE,
    "lambda_c": PARAM_RANGE,
    "gamma": GAMMA_RANGE,
}
SEARCH_SPACES["SSC-TV-L21-PQ-LowRankC"] = SEARCH_SPACES["SSC-TV-L21-PQ-SparseC"]
SEARCH_SPACES["SSC-TV-E1E21-L21-P"] = SEARCH_SPACES["SSC-TV-E1E21"]
SEARCH_SPACES["SSC-TV-E1E21-L21-PQ"] = SEARCH_SPACES["SSC-TV-E1E21"]

INT_SEARCH_KEYS = {"s", "d"}

FIELDNAMES = [
    "split", "pdb_id", "oligomer_label", "n_subunits_meta", "n_chains",
    "k_hat", "n_residues", "n_residues_raw", "stride", "chain_sizes",
    "cutoff", "method", "ari", "nmi", "seconds", "error",
]


def subsample_stride(n: int, max_n: int) -> int:
    if max_n is None or max_n <= 0 or n <= max_n:
        return 1
    return int(np.ceil(n / max_n))


def load_contact_graph(
    cif_path: Path,
    cutoff: float,
    min_chain_len: int,
    max_n: int,
):
    coords, _seq, chain_ids = extract_ca_coords(cif_path)
    counts = Counter(chain_ids)
    keep = np.array([counts[c] >= min_chain_len for c in chain_ids], dtype=bool)
    if keep.sum() < 2:
        raise RuntimeError(
            f"fewer than 2 residues after min-chain-len={min_chain_len}"
        )
    coords = coords[keep]
    chain_ids = [c for c, ok in zip(chain_ids, keep) if ok]

    n_raw = len(chain_ids)
    stride = subsample_stride(n_raw, max_n)
    if stride > 1:
        idx = np.arange(0, n_raw, stride)
        coords = coords[idx]
        chain_ids = [chain_ids[i] for i in idx]

    remaining = Counter(chain_ids)
    # A stride can empty a short chain; drop empties.
    keep2 = np.array([remaining[c] >= 1 for c in chain_ids], dtype=bool)
    coords = coords[keep2]
    chain_ids = [c for c, ok in zip(chain_ids, keep2) if ok]
    remaining = Counter(chain_ids)
    if len(remaining) < 2:
        raise RuntimeError("fewer than 2 chains after filtering / stride")

    dist = pairwise_distances(coords)
    adj = contact_adjacency(dist, cutoff).astype(np.float64)
    labels, sizes = encode_labels(chain_ids)
    return adj, labels, sizes, n_raw, stride


def encode_labels(chain_ids):
    order = []
    seen = set()
    for c in chain_ids:
        if c not in seen:
            seen.add(c)
            order.append(c)
    mapping = {c: i for i, c in enumerate(order)}
    labels = np.array([mapping[c] for c in chain_ids], dtype=int)
    sizes = [int(np.sum(labels == i)) for i in range(len(order))]
    return labels, sizes


def cluster_coeff(coeff, k, method_name, k_method="eigengap", min_k=2, penalty=0.0):
    kw = dict(k=k, method=k_method, min_k=min_k, penalty=penalty)
    if method_name in ("OSC", "BDOSC"):
        return cluster_from_Z(coeff, **kw)
    return cluster_from_C(coeff, **kw)


def run_one(Y, labels, method_name, solver, k="oracle",
            k_method="eigengap", min_k=2, penalty=0.0):
    Y = bench.normalize_Y(Y)
    run_k = bench.resolve_run_k(k, int(np.unique(labels).size))
    t0 = time.perf_counter()
    kind = bench.METHOD_KIND.get(method_name)
    if kind == "dp_y":
        pred = cluster_from_C(
            Y, k=run_k, method=k_method, min_k=min_k, penalty=penalty,
        )
        if run_k is None:
            run_k = int(np.unique(pred).size)
    elif kind == "tkss":
        if run_k is None:
            run_k = estimate_k_from_data(
                Y, method=k_method, min_k=min_k, penalty=penalty,
            )
        pred, _residual = solver(Y, run_k)
    elif kind == "bdosc":
        if run_k is None:
            run_k = estimate_k_from_data(
                Y, method=k_method, min_k=min_k, penalty=penalty,
            )
        coeff, _E, _F = solver(Y, run_k)
        pred = cluster_coeff(
            coeff, run_k, method_name,
            k_method=k_method, min_k=min_k, penalty=penalty,
        )
    else:
        coeff, _E, _F = solver(Y)
        pred = cluster_coeff(
            coeff, run_k, method_name,
            k_method=k_method, min_k=min_k, penalty=penalty,
        )
        if run_k is None:
            run_k = int(np.unique(pred).size)
    elapsed = time.perf_counter() - t0
    ari = float(adjusted_rand_score(labels, pred))
    # NaN from divide-by-zero / degenerate partitions → -1 (Optuna moves away).
    if not np.isfinite(ari):
        ari = -1.0
    return {
        "ari": ari,
        "nmi": float(normalized_mutual_info_score(labels, pred)),
        "seconds": elapsed,
        "k_hat": int(run_k),
        "error": "",
    }


def stratified_split(meta_rows, frac, seed):
    """Split assemblies within each oligomer class so both halves stay balanced."""
    if frac <= 0:
        return [], list(meta_rows)
    if frac >= 1:
        return list(meta_rows), []

    rng = np.random.default_rng(seed)
    tune, test = [], []
    groups = {}
    for row in meta_rows:
        groups.setdefault(row.get("oligomer_label", ""), []).append(row)

    labels = [lab for lab in OLIGOMER_ORDER if lab in groups]
    labels.extend(lab for lab in groups if lab not in labels)
    for lab in labels:
        group = groups[lab]
        order = rng.permutation(len(group))
        n_tune = int(round(len(group) * frac))
        if len(group) >= 2:
            n_tune = min(max(n_tune, 1), len(group) - 1)
        chosen = [group[i] for i in order]
        tune.extend(chosen[:n_tune])
        test.extend(chosen[n_tune:])
    return tune, test


def load_assembly(meta, datadir, cutoff, min_chain_len, max_n):
    rec = {
        "meta": meta,
        "pdb_id": meta["pdb_id"],
        "oligomer_label": meta.get("oligomer_label", ""),
        "n_subunits_meta": int(meta.get("protein_subunit_count") or 0),
        "n_chains": 0,
        "n_residues": 0,
        "n_residues_raw": 0,
        "stride": 1,
        "chain_sizes": [],
        "cutoff": cutoff,
        "Y": None,
        "labels": None,
        "error": "",
    }
    cif_path = datadir / meta["cif_file"]
    try:
        Y, labels, sizes, n_raw, stride = load_contact_graph(
            cif_path, cutoff, min_chain_len, max_n,
        )
        Y = bench.normalize_Y(Y)
        rec.update({
            "Y": Y,
            "labels": labels,
            "chain_sizes": sizes,
            "n_chains": int(np.unique(labels).size),
            "n_residues": int(Y.shape[0]),
            "n_residues_raw": int(n_raw),
            "stride": int(stride),
        })
    except Exception as exc:
        rec["error"] = f"{type(exc).__name__}: {exc}"
        # Expected skip: peptide filtering / stride emptied all but one chain.
        if "fewer than 2" in str(exc):
            print(f"  skip {rec['pdb_id']}: {exc}", flush=True)
        else:
            traceback.print_exc()
    return rec


def mean_ari(solver, method_name, graphs, k="oracle",
             k_method="eigengap", min_k=2, penalty=0.0):
    aris = []
    for g in graphs:
        if g["error"] or g["Y"] is None:
            aris.append(float("nan"))
            continue
        try:
            rec = run_one(
                g["Y"], g["labels"], method_name, solver, k=k,
                k_method=k_method, min_k=min_k, penalty=penalty,
            )
            aris.append(rec["ari"])
        except Exception:
            aris.append(float("nan"))
    a = np.array(aris, dtype=float)
    if not np.isfinite(a).any():
        return -1.0
    return float(np.nanmean(a))


def spec_by_name(name):
    for spec in bench.METHOD_SPECS:
        if spec["name"] == name:
            return spec
    raise KeyError(name)


def suggest_params(trial, name):
    space = SEARCH_SPACES[name]
    params = {}
    for key, (lo, hi) in space.items():
        if key in INT_SEARCH_KEYS:
            params[key] = trial.suggest_int(key, int(lo), int(hi))
        else:
            params[key] = trial.suggest_float(key, lo, hi, log=True)
    return params


def default_search_params(name):
    spec = spec_by_name(name)
    out = {}
    for key in SEARCH_SPACES[name]:
        v = spec["defaults"][key]
        out[key] = int(v) if key in INT_SEARCH_KEYS else float(v)
    return out


def eval_config(name, extra, graphs, max_iter, k="oracle",
                k_method="eigengap", min_k=2, penalty=0.0):
    spec = spec_by_name(name)
    kw = dict(spec["defaults"])
    if spec["kind"] != "tkss":
        kw["max_iter"] = max_iter
    kw.update(extra)
    if spec["kind"] == "ssc":
        kw["lambda_z"] = bench.LAMBDA_Z
    solver = bench.make_solver(spec, kw)
    return mean_ari(
        solver, name, graphs, k=k,
        k_method=k_method, min_k=min_k, penalty=penalty,
    )


def tune_method_optuna(
    name, graphs, max_iter, n_trials, n_startup, seed, storage_path, resume,
    k="oracle", k_method="eigengap", min_k=2, penalty=0.0,
):
    ok = [g for g in graphs if not g["error"] and g["Y"] is not None]
    if name not in SEARCH_SPACES:
        raise KeyError(f"no Optuna search space for {name}")

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", optuna.exceptions.ExperimentalWarning)
        sampler = optuna.samplers.TPESampler(
            seed=seed,
            multivariate=True,
            n_startup_trials=n_startup,
        )
    db_url = f"sqlite:///{Path(storage_path).resolve()}"
    storage = optuna.storages.RDBStorage(url=db_url)
    if not resume:
        try:
            optuna.delete_study(study_name=name, storage=storage)
        except (KeyError, optuna.exceptions.OptunaError):
            pass
    study = optuna.create_study(
        study_name=name,
        storage=storage,
        direction="maximize",
        sampler=sampler,
        load_if_exists=resume,
    )
    if not resume or len(study.trials) == 0:
        study.enqueue_trial(default_search_params(name))

    def n_finished(s):
        finished = {
            optuna.trial.TrialState.COMPLETE,
            optuna.trial.TrialState.FAIL,
            optuna.trial.TrialState.PRUNED,
        }
        return sum(t.state in finished for t in s.trials)

    t0 = time.perf_counter()
    n_done_start = n_finished(study)
    remaining = n_trials if not resume else max(n_trials - n_done_start, 0)
    target = n_done_start + remaining

    def objective(trial):
        extra = suggest_params(trial, name)
        ari = eval_config(
            name, extra, ok, max_iter, k=k,
            k_method=k_method, min_k=min_k, penalty=penalty,
        )
        # NaN ARI (e.g. divide-by-zero) → -1 so Optuna moves away.
        if not np.isfinite(ari):
            return -1.0
        return ari

    def callback(study, trial):
        done = n_finished(study)
        elapsed = time.perf_counter() - t0
        left = max(target - done, 0)
        eta = (elapsed / max(done - n_done_start, 1)) * left
        value = trial.value
        ari_s = f"{value:.3f}" if value is not None and np.isfinite(value) else "nan"
        try:
            best_s = f"{study.best_value:.3f}"
        except ValueError:
            best_s = "n/a"
        print(
            f"  [{done}/{target}] {name}  {trial.params}  "
            f"tune ARI={ari_s}  best={best_s}  ETA {eta:.0f}s",
            flush=True,
        )

    if remaining:
        study.optimize(objective, n_trials=remaining, callbacks=[callback])

    best_trial = study.best_trial
    params = {}
    for k, v in best_trial.params.items():
        params[k] = int(v) if k in INT_SEARCH_KEYS else float(v)
    best = {
        "params": params,
        "val_ari": float(best_trial.value),
        "n_trials": len(study.trials),
    }
    rows = []
    for t in study.trials:
        rec = {"method": name, "trial": t.number, "val_ari": t.value}
        rec.update(t.params)
        rows.append(rec)
    return best, rows


def parse_chain_sizes(value):
    if value is None or value == "":
        return []
    if isinstance(value, (list, tuple, np.ndarray)):
        return [int(x) for x in value]
    return [int(x) for x in str(value).split()]


def equal_size_subunits(sizes):
    """True when every chain length is the same up to a 1-residue remainder.

    That is the partition TKSS uses at initialisation (equal-length contiguous
    blocks), so those assemblies are excluded from TKSS accuracy.
    """
    sizes = [int(s) for s in sizes]
    if len(sizes) < 2:
        return False
    return max(sizes) - min(sizes) <= 1


def row_equal_size(row):
    return equal_size_subunits(parse_chain_sizes(row.get("chain_sizes", "")))


def ssc_method_names(method_names):
    """OSC / SSC-TV variants; TKSS is reported only on unequal-size assemblies."""
    return [m for m in method_names if m != "TKSS"]


def split_by_subunit_balance(rows):
    balanced, unbalanced = [], []
    for r in rows:
        (balanced if row_equal_size(r) else unbalanced).append(r)
    return balanced, unbalanced


def rows_for_eval(rows, split="test"):
    ok = [r for r in rows if r.get("error", "") == ""]
    tagged = [r for r in ok if r.get("split")]
    if not tagged:
        return ok
    chosen = [r for r in ok if r.get("split") == split]
    return chosen if chosen else ok


def graph_to_row_base(g, split, cutoff):
    return {
        "split": split,
        "pdb_id": g["pdb_id"],
        "oligomer_label": g["oligomer_label"],
        "n_subunits_meta": g["n_subunits_meta"],
        "n_chains": g["n_chains"],
        "n_residues": g["n_residues"],
        "n_residues_raw": g["n_residues_raw"],
        "stride": g["stride"],
        "chain_sizes": " ".join(str(s) for s in g["chain_sizes"]),
        "cutoff": cutoff,
    }


def eval_graphs(graphs, methods, split, writer, fh, rows, n_jobs, done, t_start,
                cutoff, k="oracle", k_method="eigengap", min_k=2, penalty=0.0):
    for g in graphs:
        rec_base = graph_to_row_base(g, split, cutoff)
        if g["error"] or g["Y"] is None:
            for method_name, _solver in methods:
                rec = dict(rec_base)
                rec.update({
                    "method": method_name,
                    "k_hat": "",
                    "ari": float("nan"),
                    "nmi": float("nan"),
                    "seconds": float("nan"),
                    "error": g["error"],
                })
                writer.writerow(rec)
                fh.flush()
                rows.append(rec)
                done += 1
            continue

        print(
            f"{split:4s}  {g['pdb_id']}  {g['oligomer_label']:<10s}  "
            f"N={g['n_residues']} (raw {g['n_residues_raw']}, stride {g['stride']})  "
            f"k_true={g['n_chains']}  sizes={g['chain_sizes']}",
            flush=True,
        )
        for method_name, solver in methods:
            rec = dict(rec_base)
            rec.update({
                "method": method_name,
                "k_hat": "",
                "ari": float("nan"),
                "nmi": float("nan"),
                "seconds": float("nan"),
                "error": "",
            })
            try:
                rec.update(run_one(
                    g["Y"], g["labels"], method_name, solver, k=k,
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
            if rec["error"]:
                status = (
                    f"[{done}/{n_jobs}] {split} {g['pdb_id']} "
                    f"{method_name} FAILED: {rec['error']}"
                )
            else:
                status = (
                    f"[{done}/{n_jobs}] {split} {g['pdb_id']}  "
                    f"{method_name:16s}  k_hat={rec.get('k_hat', '')}  "
                    f"ARI={rec['ari']:.3f}  {rec['seconds']:.1f}s"
                )
            print(f"{status}   ({elapsed:.0f}s elapsed, ETA {eta:.0f}s)", flush=True)
    return done


def _row_score(row, score_fn):
    if score_fn is None:
        return row["ari"]
    return score_fn(row)


def plot_results(rows, out_dir, method_names):
    _plot_results_score(
        rows, out_dir, method_names,
        score_fn=None, ylabel="ARI", stem="ari",
    )
    _plot_results_score(
        rows, out_dir, method_names,
        score_fn=bench.row_sqrt_ari_nmi, ylabel=bench.SQRT_ARI_NMI_LABEL,
        stem="sqrt_ari_nmi",
    )


def _plot_results_score(rows, out_dir, method_names, score_fn, ylabel, stem):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ok = rows_for_eval(rows, split="test")
    if not ok:
        return
    balanced, unbalanced = split_by_subunit_balance(ok)
    ssc_names = ssc_method_names(method_names)
    markers = ["o", "s", "D", "^", "v", "P"]

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.6), sharey=True)
    _plot_oligomer_bars(
        axes[0], balanced, ssc_names,
        "Equal-size subunits (OSC / SSC-TV)",
        score_fn=score_fn, ylabel=ylabel,
    )
    _plot_oligomer_bars(
        axes[1], unbalanced, method_names,
        "Unequal-size subunits (TKSS included)",
        score_fn=score_fn, ylabel=ylabel,
    )
    fig.tight_layout()
    path = out_dir / f"{stem}_by_oligomer.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Wrote {path}", flush=True)

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.8), sharey=True)
    for ax, sub, names, title in (
        (axes[0], balanced, ssc_names, "Equal-size subunits"),
        (axes[1], unbalanced, method_names, "Unequal-size subunits"),
    ):
        for i, method in enumerate(names):
            pts = [r for r in sub if r["method"] == method]
            ax.scatter(
                [r["n_residues"] for r in pts],
                [_row_score(r, score_fn) for r in pts],
                s=28, marker=markers[i % len(markers)], label=method, alpha=0.85,
            )
        ax.set_xlabel("Residues after filter / stride (N)")
        ax.set_ylabel(ylabel)
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend(frameon=False)
    fig.tight_layout()
    path = out_dir / f"{stem}_vs_n.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Wrote {path}", flush=True)

    _plot_paired(
        balanced, ssc_names, out_dir / f"{stem}_paired_balanced.png",
        "Equal-size subunits", score_fn=score_fn, ylabel=ylabel,
    )
    _plot_paired(
        unbalanced, method_names, out_dir / f"{stem}_paired_unbalanced.png",
        "Unequal-size subunits", score_fn=score_fn, ylabel=ylabel,
    )


def _plot_oligomer_bars(ax, ok, method_names, title, score_fn=None, ylabel="ARI"):
    if not method_names:
        ax.set_visible(False)
        return
    x = np.arange(len(OLIGOMER_ORDER))
    width = 0.35 if len(method_names) == 2 else 0.8 / max(len(method_names), 1)
    for i, method in enumerate(method_names):
        means, stds = [], []
        for lab in OLIGOMER_ORDER:
            a = np.array([
                _row_score(r, score_fn) for r in ok
                if r["method"] == method and r["oligomer_label"] == lab
            ], dtype=float)
            a = a[np.isfinite(a)]
            means.append(float(a.mean()) if a.size else np.nan)
            stds.append(float(a.std(ddof=1)) if a.size > 1 else 0.0)
        offset = (i - (len(method_names) - 1) / 2) * width
        ax.bar(x + offset, means, width=width, yerr=stds, capsize=3, label=method)
    ax.set_xticks(x)
    ax.set_xticklabels(OLIGOMER_ORDER, rotation=25, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(frameon=False)


def _plot_paired(ok, method_names, path, title, score_fn=None, ylabel="ARI"):
    import matplotlib.pyplot as plt

    pairs = [
        (method_names[i], method_names[j])
        for i in range(len(method_names))
        for j in range(i + 1, len(method_names))
    ]
    if not pairs:
        return
    n_pairs = len(pairs)
    fig, axes = plt.subplots(
        1, n_pairs, figsize=(5.4 * n_pairs, 5.4), squeeze=False,
    )
    axes = axes.ravel()
    paired = {}
    for r in ok:
        paired.setdefault(r["pdb_id"], {})[r["method"]] = _row_score(r, score_fn)
    for ax, (a_name, b_name) in zip(axes, pairs):
        xs, ys = [], []
        for vals in paired.values():
            if a_name in vals and b_name in vals:
                xs.append(vals[a_name])
                ys.append(vals[b_name])
        if not xs:
            ax.set_visible(False)
            continue
        ax.scatter(xs, ys, s=28, alpha=0.85)
        ax.plot([-0.05, 1.05], [-0.05, 1.05], color="0.6", linewidth=1)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel(f"{a_name} {ylabel}")
        ax.set_ylabel(f"{b_name} {ylabel}")
        ax.set_title(title)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Wrote {path}", flush=True)


def _print_ari_table(ok, method_names, title):
    print(f"\n=== {title} ===", flush=True)
    if not ok or not method_names:
        print("  (no assemblies)", flush=True)
        return
    n_asm = len({r["pdb_id"] for r in ok})
    print(f"  n assemblies = {n_asm}", flush=True)
    header = f"{'class':>12}" + "".join(f"{m:>22}" for m in method_names)
    print(header, flush=True)
    for lab in ["all"] + OLIGOMER_ORDER:
        cells = [f"{lab:>12}"]
        any_val = False
        for method in method_names:
            if lab == "all":
                sub = [r["ari"] for r in ok if r["method"] == method]
            else:
                sub = [
                    r["ari"] for r in ok
                    if r["method"] == method and r["oligomer_label"] == lab
                ]
            a = np.array(sub, dtype=float)
            a = a[np.isfinite(a)]
            if a.size == 0:
                cells.append(f"{'n/a':>22}")
            else:
                any_val = True
                s = a.std(ddof=1) if a.size > 1 else 0.0
                cells.append(f"{a.mean():8.3f} ± {s:<6.3f}".rjust(22))
        if lab == "all" or any_val:
            print("".join(cells), flush=True)


def _paired_wins(ok, method_names):
    for i, a_name in enumerate(method_names):
        for b_name in method_names[i + 1:]:
            _paired_wins_two(ok, a_name, b_name)


def _paired_wins_two(ok, a_name, b_name):
    paired = {}
    for r in ok:
        paired.setdefault(r["pdb_id"], {})[r["method"]] = r["ari"]
    n_a = n_b = n_tie = 0
    diffs = []
    for vals in paired.values():
        if a_name not in vals or b_name not in vals:
            continue
        if not (np.isfinite(vals[a_name]) and np.isfinite(vals[b_name])):
            continue
        diffs.append(vals[b_name] - vals[a_name])
        if vals[b_name] > vals[a_name] + 1e-12:
            n_b += 1
        elif vals[a_name] > vals[b_name] + 1e-12:
            n_a += 1
        else:
            n_tie += 1
    if not diffs:
        return
    d = np.array(diffs, dtype=float)
    print(
        f"\n=== Paired held-out comparison ({a_name} vs {b_name}) ===",
        flush=True,
    )
    print(
        f"  {a_name} wins {n_a}   {b_name} wins {n_b}   ties {n_tie}   "
        f"n={len(diffs)}",
        flush=True,
    )
    print(
        f"  mean ARI({b_name}) − ARI({a_name}) = {d.mean():+.3f}  "
        f"(median {float(np.median(d)):+.3f})",
        flush=True,
    )


def _print_k_recovery(ok, method_names):
    rows = [
        r for r in ok
        if r.get("k_hat") not in ("", None) and r.get("n_chains") not in ("", None)
    ]
    if not rows:
        return
    print("\n=== Estimated k vs true n_chains ===", flush=True)
    for method in method_names:
        sub = [r for r in rows if r["method"] == method]
        if not sub:
            continue
        hats, trues = [], []
        for r in sub:
            try:
                hats.append(int(r["k_hat"]))
                trues.append(int(r["n_chains"]))
            except (TypeError, ValueError):
                continue
        if not hats:
            continue
        hats = np.array(hats)
        trues = np.array(trues)
        exact = float(np.mean(hats == trues))
        mae = float(np.mean(np.abs(hats - trues)))
        print(
            f"  {method:16s}  exact={exact:.3f}  MAE={mae:.2f}  "
            f"mean k_hat={hats.mean():.2f}  mean k_true={trues.mean():.2f}",
            flush=True,
        )


def write_summary(rows, out_dir, method_names):
    splits = []
    for name in ("tune", "test"):
        if any(r.get("split") == name for r in rows):
            splits.append(name)
    if not splits:
        splits = [None]

    path = out_dir / "protein_benchmark_summary.csv"
    fields = [
        "split", "scope", "method", "n", "ari_mean", "ari_std", "ari_median",
        "nmi_mean", "seconds_mean",
    ]
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()

        def dump(split, scope, sub, method):
            ari = np.array([r["ari"] for r in sub], dtype=float)
            ari = ari[np.isfinite(ari)]
            nmi = np.array([r["nmi"] for r in sub], dtype=float)
            nmi = nmi[np.isfinite(nmi)]
            sec = np.array([r["seconds"] for r in sub], dtype=float)
            sec = sec[np.isfinite(sec)]
            if ari.size == 0:
                return
            writer.writerow({
                "split": split or "all",
                "scope": scope,
                "method": method,
                "n": int(ari.size),
                "ari_mean": float(ari.mean()),
                "ari_std": float(ari.std(ddof=1) if ari.size > 1 else 0.0),
                "ari_median": float(np.median(ari)),
                "nmi_mean": float(nmi.mean()) if nmi.size else float("nan"),
                "seconds_mean": float(sec.mean()) if sec.size else float("nan"),
            })

        for split in splits:
            if split:
                ok = rows_for_eval(rows, split=split)
            else:
                ok = [r for r in rows if r.get("error", "") == ""]
            balanced, unbalanced = split_by_subunit_balance(ok)
            ssc_names = ssc_method_names(method_names)
            for method in ssc_names:
                dump(split, "balanced", [r for r in balanced if r["method"] == method], method)
                for lab in OLIGOMER_ORDER:
                    dump(
                        split,
                        f"balanced:{lab}",
                        [r for r in balanced if r["method"] == method and r["oligomer_label"] == lab],
                        method,
                    )
            for method in method_names:
                dump(split, "unbalanced", [r for r in unbalanced if r["method"] == method], method)
                for lab in OLIGOMER_ORDER:
                    dump(
                        split,
                        f"unbalanced:{lab}",
                        [r for r in unbalanced if r["method"] == method and r["oligomer_label"] == lab],
                        method,
                    )

    print(f"Wrote {path}", flush=True)

    def _report_split(ok, label):
        balanced, unbalanced = split_by_subunit_balance(ok)
        ssc_names = ssc_method_names(method_names)
        _print_ari_table(
            balanced, ssc_names,
            f"{label} ARI — equal-size subunits (OSC / SSC-TV)",
        )
        _print_ari_table(
            unbalanced, method_names,
            f"{label} ARI — unequal-size subunits (TKSS included)",
        )
        print(f"\n--- {label} paired, equal-size subunits ---", flush=True)
        _paired_wins(balanced, ssc_names)
        print(f"\n--- {label} paired, unequal-size subunits ---", flush=True)
        _paired_wins(unbalanced, method_names)
        _print_k_recovery(ok, method_names)

    test_ok = rows_for_eval(rows, split="test")
    has_tune = any(r.get("split") == "tune" for r in rows)
    if has_tune:
        _report_split(rows_for_eval(rows, split="tune"), "Tune-split")
    _report_split(test_ok, "Held-out test")

    print("\n=== Runtime (seconds, held-out test) ===", flush=True)
    for method in method_names:
        a = np.array(
            [r["seconds"] for r in test_ok if r["method"] == method], dtype=float,
        )
        a = a[np.isfinite(a)]
        if a.size == 0:
            continue
        std = float(a.std(ddof=1)) if a.size > 1 else 0.0
        print(
            f"{method:16s}  n={a.size:3d}  mean={a.mean():7.2f}  "
            f"std={std:6.2f}  median={float(np.median(a)):7.2f}",
            flush=True,
        )


def write_tune_artifacts(out_dir, selected, all_rows, protocol, merge=False):
    csv_path = out_dir / "tune_grid.csv"
    existing_rows = []
    if merge and csv_path.exists():
        with csv_path.open(newline="") as fh:
            existing_rows = [
                r for r in csv.DictReader(fh)
                if r.get("method") not in selected
            ]
    combined = existing_rows + all_rows
    param_keys = sorted({
        k for r in combined for k in r
        if k not in ("method", "val_ari") and r.get(k) not in ("", None)
    })
    with csv_path.open("w", newline="") as fh:
        fields = ["method", *param_keys, "val_ari"]
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for r in combined:
            writer.writerow(r)
    print(f"Wrote {csv_path}", flush=True)

    json_path = out_dir / "best_hyperparams.json"
    if merge and json_path.exists():
        payload = json.loads(json_path.read_text())
        payload.setdefault("methods", {}).update(selected)
        payload.setdefault("protocol", {}).update(protocol)
    else:
        payload = {"protocol": protocol, "methods": selected}
    json_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Wrote {json_path}", flush=True)


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--datadir", type=Path,
        default=ROOT / "protein_data" / "pdb_multimers",
    )
    p.add_argument("--out-dir", type=Path, default=ROOT / "results" / "proteins")
    p.add_argument("--cutoff", type=float, default=8.0)
    p.add_argument("--min-chain-len", type=int, default=20)
    p.add_argument(
        "--max-n", type=int, default=400,
        help="Uniformly stride residues so N ≤ this (0 disables). Default 400.",
    )
    p.add_argument("--max-iter", type=int, default=200)
    p.add_argument(
        "--tune-max-iter", type=int, default=TUNE_MAX_ITER,
        help="ADMM iterations during the hyperparameter search (default 80).",
    )
    p.add_argument(
        "--n-trials", type=int, default=N_TRIALS,
        help="Optuna trials per method. Default 40.",
    )
    p.add_argument(
        "--n-startup-trials", type=int, default=N_STARTUP_TRIALS,
        help="Random TPE warmup trials before Bayesian proposals. Default 10.",
    )
    p.add_argument(
        "--tune-seed", type=int, default=0,
        help="Optuna TPESampler seed.",
    )
    p.add_argument(
        "--resume", action="store_true",
        help="Continue Optuna studies stored in out-dir/optuna.db.",
    )
    p.add_argument(
        "--tune-frac", type=float, default=TUNE_FRAC,
        help="Fraction of assemblies used to tune (rest is held-out test). "
             "Default 0.3.  0 skips the search and evaluates on all data.",
    )
    p.add_argument("--split-seed", type=int, default=SPLIT_SEED)
    p.add_argument(
        "--k", type=bench.parse_k_arg, default="oracle",
        help="Cluster count: 'oracle' (default, true remaining-chain k), "
             "'none' (infer via --k-method), or a positive integer.  True "
             "subunit labels are always used for ARI / NMI.",
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
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--pdb", nargs="+", default=None, help="Optional PDB ID subset.")
    p.add_argument("--methods", nargs="+", default=DEFAULT_METHODS, choices=METHOD_CHOICES)
    p.add_argument(
        "--params", type=str, default="",
        help="Skip the search and use this JSON (protein or SBM tuner output). "
             "Empty (default) tunes on the train split.",
    )
    p.add_argument(
        "--smoke", action="store_true",
        help="One small dimer (1A4C), selected methods — check a single run.",
    )
    p.add_argument(
        "--plot-only", action="store_true",
        help="Recompute summary and plots from out-dir/protein_benchmark.csv.",
    )
    p.add_argument(
        "--append", action="store_true",
        help="Evaluate only --methods and merge into existing "
             "protein_benchmark.csv / best_hyperparams.json (do not overwrite "
             "rows for other methods).",
    )
    return p.parse_args()


def load_rows(path):
    rows = []
    with Path(path).open(newline="") as fh:
        for r in csv.DictReader(fh):
            rec = dict(r)
            rec.setdefault("split", "")
            rec.setdefault("k_hat", "")
            for key in ("n_subunits_meta", "n_chains", "n_residues", "n_residues_raw", "stride"):
                rec[key] = int(rec[key]) if rec[key] not in ("", None) else 0
            if rec.get("k_hat") not in ("", None, "nan"):
                rec["k_hat"] = int(float(rec["k_hat"]))
            rec["cutoff"] = float(rec["cutoff"])
            for key in ("ari", "nmi", "seconds"):
                rec[key] = float(rec[key]) if rec[key] not in ("", "nan") else float("nan")
            rows.append(rec)
    return rows


def apply_saved_split(meta_rows, payload):
    proto = payload.get("protocol", {})
    tune_ids = proto.get("tune_pdb_ids")
    test_ids = proto.get("test_pdb_ids")
    if not tune_ids and not test_ids:
        return None, None
    by_id = {r["pdb_id"].upper(): r for r in meta_rows}
    tune = [by_id[i.upper()] for i in (tune_ids or []) if i.upper() in by_id]
    test = [by_id[i.upper()] for i in (test_ids or []) if i.upper() in by_id]
    return tune, test


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.plot_only:
        rows = load_rows(out_dir / "protein_benchmark.csv")
        names = []
        for name in METHOD_CHOICES:
            if any(r["method"] == name for r in rows):
                names.append(name)
        write_summary(rows, out_dir, names)
        plot_results(rows, out_dir, names)
        return

    meta_rows = load_metadata(args.datadir / "metadata.csv")
    if args.pdb:
        wanted = {x.upper() for x in args.pdb}
        meta_rows = [r for r in meta_rows if r["pdb_id"].upper() in wanted]
    if args.smoke:
        meta_rows = [r for r in meta_rows if r["pdb_id"] == "1A4C"] or meta_rows[:1]
        print("Smoke mode: 1A4C only (no hyperparameter search).", flush=True)
    if args.limit:
        meta_rows = meta_rows[: args.limit]

    params_path = args.params.strip() if args.params else ""
    params_payload = {}
    overrides = {}
    if params_path:
        path = Path(params_path)
        if path.exists():
            params_payload = json.loads(path.read_text())
            overrides = bench.load_param_overrides(path)
            print(f"Loaded hyperparameters from {path}", flush=True)
            for name in args.methods:
                if name in overrides:
                    print(f"  {name}: {overrides[name]}", flush=True)
        else:
            print(f"No params file at {path}; will tune or use solver defaults.", flush=True)

    do_tune_names = [
        m for m in args.methods
        if m not in overrides and m in SEARCH_SPACES
    ]
    do_tune = (not args.smoke) and bool(do_tune_names) and args.tune_frac > 0
    if args.smoke:
        tune_meta, test_meta = [], meta_rows
    elif args.tune_frac <= 0:
        tune_meta, test_meta = [], meta_rows
    else:
        saved = apply_saved_split(meta_rows, params_payload) if params_payload else (None, None)
        if saved[0] is not None:
            tune_meta, test_meta = saved
            print(
                f"Reusing saved split: {len(tune_meta)} tune / {len(test_meta)} test",
                flush=True,
            )
        else:
            tune_meta, test_meta = stratified_split(
                meta_rows, args.tune_frac, args.split_seed,
            )

    print(
        f"Split  tune={len(tune_meta)}  test={len(test_meta)}  "
        f"(frac={args.tune_frac:g}, seed={args.split_seed})",
        flush=True,
    )
    if args.k is None:
        k_s = f"none ({args.k_method}, min_k={args.min_k})"
    elif args.k == "oracle":
        k_s = "oracle (true remaining-chain count)"
    else:
        k_s = str(args.k)
    print(f"k={k_s}  (true subunit labels used only for ARI / NMI)", flush=True)
    for lab in OLIGOMER_ORDER:
        n_tr = sum(1 for r in tune_meta if r.get("oligomer_label") == lab)
        n_te = sum(1 for r in test_meta if r.get("oligomer_label") == lab)
        if n_tr or n_te:
            print(f"  {lab:<10s}  tune {n_tr:2d}  test {n_te:2d}", flush=True)

    print("Loading contact graphs …", flush=True)
    tune_graphs = [
        load_assembly(m, args.datadir, args.cutoff, args.min_chain_len, args.max_n)
        for m in tune_meta
    ]
    test_graphs = [
        load_assembly(m, args.datadir, args.cutoff, args.min_chain_len, args.max_n)
        for m in test_meta
    ]
    n_ok_tune = sum(1 for g in tune_graphs if not g["error"])
    n_ok_test = sum(1 for g in test_graphs if not g["error"])
    print(
        f"  loaded {n_ok_tune}/{len(tune_graphs)} tune, "
        f"{n_ok_test}/{len(test_graphs)} test",
        flush=True,
    )
    if do_tune and n_ok_tune < 1:
        print("No usable tune graphs; skipping search and using solver defaults.", flush=True)
        do_tune = False

    selected = {}
    tune_rows = []
    if do_tune:
        t_tune = time.perf_counter()
        storage_path = out_dir / "optuna.db"
        for name in args.methods:
            if name in overrides:
                print(f"Reusing saved params for {name}: {overrides[name]}", flush=True)
                if name in params_payload.get("methods", {}):
                    selected[name] = params_payload["methods"][name]
                continue
            if name not in SEARCH_SPACES:
                print(f"No Optuna space for {name}; keeping solver defaults.", flush=True)
                continue
            print("", flush=True)
            print(
                f"=== Tuning {name}  (Optuna TPE, {args.n_trials} trials × "
                f"{n_ok_tune} graphs, max_iter={args.tune_max_iter}) ===",
                flush=True,
            )
            best, rows = tune_method_optuna(
                name, tune_graphs, args.tune_max_iter,
                n_trials=args.n_trials,
                n_startup=args.n_startup_trials,
                seed=args.tune_seed + args.methods.index(name),
                storage_path=storage_path,
                resume=args.resume,
                k=args.k,
                k_method=args.k_method,
                min_k=args.min_k,
                penalty=args.k_penalty,
            )
            tune_rows.extend(rows)
            selected[name] = best
            overrides[name] = best["params"]
            print(
                f"  → best {best['params']}  tune ARI={best['val_ari']:.3f}",
                flush=True,
            )
        protocol = {
            "search": "optuna",
            "tune_frac": args.tune_frac,
            "split_seed": args.split_seed,
            "tune_seed": args.tune_seed,
            "tune_pdb_ids": [r["pdb_id"] for r in tune_meta],
            "test_pdb_ids": [r["pdb_id"] for r in test_meta],
            "max_iter_tune": args.tune_max_iter,
            "max_iter_test": args.max_iter,
            "cutoff": args.cutoff,
            "min_chain_len": args.min_chain_len,
            "max_n": args.max_n,
            "objective": "mean ARI on the tune split",
            "elapsed_seconds": time.perf_counter() - t_tune,
            "sampler": "TPESampler(multivariate=True)",
            "n_trials": args.n_trials,
            "n_startup_trials": args.n_startup_trials,
            "search_spaces": SEARCH_SPACES,
            "y_normalization": "frobenius",
            "lambda_z": bench.LAMBDA_Z,
            "lambda_e21_scale": "sqrt(N)",
            "k": "none" if args.k is None else args.k,
            "k_method": args.k_method,
            "min_k": args.min_k,
            "k_penalty": args.k_penalty,
            "n_configs": {
                m: args.n_trials for m in do_tune_names
            },
            "reused_methods": [
                m for m in args.methods
                if m not in do_tune_names and m in overrides
            ],
        }
        write_tune_artifacts(
            out_dir, selected, tune_rows, protocol, merge=args.append,
        )
        print("\n=== Selected hyperparameters ===", flush=True)
        for name in args.methods:
            if name in selected:
                info = selected[name]
                print(
                    f"{name:16s}  {info['params']}  "
                    f"tune ARI={info['val_ari']:.3f}",
                    flush=True,
                )

    methods = bench.build_methods(
        overrides=overrides, max_iter=args.max_iter, names=args.methods,
    )

    eval_pairs = []
    if tune_graphs:
        eval_pairs.append(("tune", tune_graphs))
    if test_graphs:
        eval_pairs.append(("test", test_graphs))
    n_jobs = sum(len(gs) for _, gs in eval_pairs) * len(methods)
    print(
        f"\nEvaluating selected params  "
        f"({n_jobs} jobs, max_iter={args.max_iter}"
        f"{', append' if args.append else ''})",
        flush=True,
    )

    csv_path = out_dir / "protein_benchmark.csv"
    kept_rows = []
    if args.append and csv_path.exists():
        kept_rows = [
            r for r in load_rows(csv_path) if r.get("method") not in args.methods
        ]
        print(
            f"Append: keeping {len(kept_rows)} existing rows; "
            f"replacing methods {args.methods}",
            flush=True,
        )

    rows = list(kept_rows)
    done = 0
    t_start = time.perf_counter()

    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDNAMES)
        writer.writeheader()
        for r in kept_rows:
            writer.writerow(r)
        for split, graphs in eval_pairs:
            done = eval_graphs(
                graphs, methods, split, writer, fh, rows,
                n_jobs, done, t_start, args.cutoff,
                k=args.k, k_method=args.k_method,
                min_k=args.min_k, penalty=args.k_penalty,
            )

    # Prefer a stable method order: existing methods first, then newly run.
    method_names = []
    for name in METHOD_CHOICES:
        if any(r.get("method") == name for r in rows):
            method_names.append(name)
    write_summary(rows, out_dir, method_names)
    try:
        plot_results(rows, out_dir, method_names)
    except Exception:
        traceback.print_exc()
        print("Plotting skipped.", flush=True)


if __name__ == "__main__":
    main()
