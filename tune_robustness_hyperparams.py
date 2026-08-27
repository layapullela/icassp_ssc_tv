"""
Held-out hyperparameter search validated against the robustness suite
(``benchmark_sbm_robustness.py``) instead of the case-based benchmark.

This does NOT reimplement the Optuna search loop — it imports
``tune_hyperparams.py`` and reuses its ``SEARCH_SPACES``,
``default_search_params``, and (most importantly) ``tune_method_optuna``
as-is. The only thing this script replaces is where the validation graphs
come from: instead of ``bench.CASES`` × ``lambdas``, graphs are drawn from
the 5 robustness axes (``num_clusters``, ``num_nodes``, ``gaussian_noise``,
``imbalance``, ``density_ratio``), subsampled for tuning speed.

Why a separate validation set from the robustness *report*
------------------------------------------------------------
``benchmark_sbm_robustness.py`` uses seed=0 by default and its full sweep
grids when you run it for real. This tuner uses a different seed
(``VAL_SEED`` below) and a coarser, subsampled grid per axis (spread across
easy -> hard rather than every point), the same way the original
``tune_hyperparams.py`` excludes the slow N=400/800 cases and validates on
a different seed from the test benchmark. So hyperparameters are never
selected on the exact draws you'll later report.

Usage
-----
    python tune_hyperparams_robustness.py                 # all 5 axes
    python tune_hyperparams_robustness.py --axes num_clusters imbalance
    python tune_hyperparams_robustness.py --k none --k-method eigengap --min-k 2
    python tune_hyperparams_robustness.py --quick          # sanity-check the loop
    python tune_hyperparams_robustness.py --retune OSC     # merge one method in
    python benchmark_sbm_robustness.py \\
        --params results/robustness_tuned/best_hyperparams_robustness.json \\
        --out-dir results/robustness_tuned
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

import benchmark_sbm as bench
import benchmark_sbm_robustness as rbench
import tune_hyperparams as tuner  # reuse SEARCH_SPACES / tune_method_optuna as-is

ROOT = Path(__file__).resolve().parent

VAL_SEED = 3_000_003       # distinct from robustness-report seed (0) and
                           # tune_hyperparams.py's own VAL_SEED (1_000_003)
TUNE_MAX_ITER = 200
N_TRIALS = 40
N_STARTUP_TRIALS = 10
TUNE_POINTS_PER_AXIS = 4   # subsample each axis's sweep grid to this many points
TUNE_GRAPH_TRIALS = 2


# ── Validation-graph construction from the robustness axes ────────────────

def _subsample(seq, n):
    """n evenly-spaced points spanning the easy->hard range, endpoints included."""
    seq = list(seq)
    if n >= len(seq):
        return seq
    if n <= 1:
        return [seq[len(seq) // 2]]
    idx = np.linspace(0, len(seq) - 1, n)
    idx = sorted({int(round(i)) for i in idx})
    return [seq[i] for i in idx]


def subsampled_axis_cfgs(axis_name, points_per_axis):
    """Subsample an axis's full sweep grid, respecting sweep_group (e.g.
    the two Gaussian-noise structures), so tuning stays fast but still
    spans each axis's easy -> hard range.
    """
    full_cfgs = rbench.AXES[axis_name]["build"]()
    groups = sorted({c.get("sweep_group", "") for c in full_cfgs})
    out = []
    for group in groups:
        group_cfgs = [c for c in full_cfgs if c.get("sweep_group", "") == group]
        group_cfgs = sorted(group_cfgs, key=lambda c: c["sweep_value"])
        out.extend(_subsample(group_cfgs, points_per_axis))
    return out


def build_val_graphs(axes, points_per_axis, n_trials, seed, k="oracle"):
    """Mirrors ``tune_hyperparams.build_val_graphs``'s graph dict shape
    (Y / labels / k / outlier_mask) so it can be handed straight to
    ``tuner.tune_method_optuna`` / ``tuner.eval_solver`` unmodified.
    """
    graphs = []
    for axis_idx, axis_name in enumerate(axes):
        cfgs = subsampled_axis_cfgs(axis_name, points_per_axis)
        for point_idx, cfg in enumerate(cfgs):
            for trial in range(n_trials):
                rng = np.random.default_rng(
                    seed
                    + 1_000_000 * axis_idx
                    + 1_000 * point_idx
                    + trial
                )
                A, labels = bench.generate_sbm(
                    cfg["sizes"], cfg["p_in"], cfg["p_out"], rng,
                )
                if cfg["noise_model"] == "gaussian":
                    Y = bench.add_gaussian_noise(A, cfg["lambda"], rng)
                else:
                    Y = bench.add_poisson_noise(A, cfg["lambda"], rng)
                Y = bench.normalize_Y(Y)
                graphs.append({
                    "axis": axis_name,
                    "sweep_group": cfg.get("sweep_group", ""),
                    "sweep_value": cfg["sweep_value"],
                    "trial": trial,
                    "k": bench.resolve_run_k(k, cfg["k"]),
                    "Y": Y,
                    "labels": labels,
                    "outlier_mask": None,   # robustness axes have no outlier cases
                })
    return graphs


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--out-dir", type=str,
                   default=str(ROOT / "results" / "robustness_tuned"))
    p.add_argument("--seed", type=int, default=VAL_SEED)
    p.add_argument("--trials", type=int, default=TUNE_GRAPH_TRIALS,
                   help="Graph draws per (axis, sweep-value). Default 2.")
    p.add_argument("--points-per-axis", type=int, default=TUNE_POINTS_PER_AXIS,
                   help="Subsampled sweep points per axis/group (spans "
                        "easy->hard). Default 4.")
    p.add_argument("--max-iter", type=int, default=TUNE_MAX_ITER)
    p.add_argument(
        "--axes", nargs="+", default=list(rbench.AXES),
        choices=list(rbench.AXES),
    )
    p.add_argument(
        "--methods", nargs="+",
        default=list(bench.DEFAULT_METHODS),
        choices=[s["name"] for s in bench.METHOD_SPECS],
    )
    p.add_argument(
        "--retune", nargs="+", default=None,
        choices=[s["name"] for s in bench.METHOD_SPECS],
        metavar="METHOD",
        help="Only search these methods and merge the winners into existing "
             "best_hyperparams_robustness.json / tune_grid_robustness.csv.",
    )
    p.add_argument("--n-trials", type=int, default=N_TRIALS,
                   help="Optuna trials per method. Default 40.")
    p.add_argument("--n-startup-trials", type=int, default=N_STARTUP_TRIALS)
    p.add_argument("--prior-weight", type=float, default=1.0)
    p.add_argument("--tune-seed", type=int, default=0)
    p.add_argument("--resume", action="store_true")
    p.add_argument(
        "--quick", action="store_true",
        help="1 axis (num_clusters), 2 points, 1 graph trial, 3 Optuna trials.",
    )
    p.add_argument("--f1-weight", type=float, default=0.0,
                   help="Robustness axes have no outlier cases; leave at 0.")
    p.add_argument("--nmi-weight", type=float, default=0.0,
                   help="Weight on mean NMI in the selection score (default 0).")
    p.add_argument(
        "--k", type=bench.parse_k_arg, default="oracle",
        help="'oracle' (default), 'none' (infer via --k-method), or an int.",
    )
    p.add_argument("--k-method", type=str, default="eigengap",
                   choices=["eigengap", "ncut"])
    p.add_argument("--min-k", type=int, default=2)
    p.add_argument("--k-penalty", type=float, default=0.0)
    p.add_argument(
        "--merge", action="store_true",
        help="Update existing JSON/CSV instead of replacing them.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    tuner.F1_WEIGHT = args.f1_weight
    tuner.NMI_WEIGHT = args.nmi_weight
    if args.retune:
        args.methods = list(args.retune)
        args.merge = True
        print(f"Retune: {', '.join(args.methods)}  (merging into existing files)",
              flush=True)

    axes = args.axes
    points_per_axis = args.points_per_axis
    n_graph_trials = args.trials
    n_optuna_trials = args.n_trials
    if args.quick:
        axes = ["num_clusters"]
        points_per_axis = 2
        n_graph_trials = 1
        if args.n_trials == N_TRIALS:
            n_optuna_trials = 3
        print(
            f"Quick mode: axis=num_clusters, 2 points, 1 graph trial, "
            f"{n_optuna_trials} Optuna trials.",
            flush=True,
        )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.k is None:
        k_s = f"none ({args.k_method}, min_k={args.min_k})"
    else:
        k_s = str(args.k)
    print(
        f"Building validation graphs from robustness axes "
        f"({axes}, {points_per_axis} pts/axis × {n_graph_trials} trials, "
        f"seed={args.seed}, k={k_s}) …",
        flush=True,
    )
    graphs = build_val_graphs(
        axes, points_per_axis, n_graph_trials, args.seed, k=args.k,
    )
    print(f"  {len(graphs)} graphs, Y shape {graphs[0]['Y'].shape}", flush=True)

    n_jobs = n_optuna_trials * len(args.methods) * len(graphs)
    print(
        f"Search: Optuna TPE, {n_optuna_trials} trials × {len(args.methods)} "
        f"methods × {len(graphs)} graphs = {n_jobs} solver runs "
        f"(max_iter={args.max_iter})",
        flush=True,
    )

    all_rows = []
    selected = {}
    t_start = time.perf_counter()
    storage_path = out_dir / "optuna_robustness.db"
    for name in args.methods:
        print(
            f"\n=== {name}  (Optuna TPE, {n_optuna_trials} trials × "
            f"{len(graphs)} graphs, max_iter={args.max_iter}) ===",
            flush=True,
        )
        best, rows = tuner.tune_method_optuna(
            name, graphs, args.max_iter,
            n_trials=n_optuna_trials,
            n_startup=args.n_startup_trials,
            prior_weight=args.prior_weight,
            seed=args.tune_seed,
            storage_path=storage_path,
            resume=args.resume,
            k_method=args.k_method,
            min_k=args.min_k,
            penalty=args.k_penalty,
        )
        all_rows.extend(rows)
        selected[name] = best
        print(
            f"  → best {best['params']}  "
            f"ARI={best['val_ari']:.3f}  NMI={best['val_nmi']:.3f}  "
            f"score={best['score']:.3f}",
            flush=True,
        )

    csv_path = out_dir / "tune_grid_robustness.csv"
    existing_rows = []
    if args.merge and csv_path.exists():
        with csv_path.open(newline="") as fh:
            existing_rows = [
                r for r in csv.DictReader(fh) if r.get("method") not in args.methods
            ]
    combined = existing_rows + all_rows
    skip = {"method", "val_ari", "val_nmi", "val_f1", "score"}
    param_keys = sorted({
        k for r in combined for k in r
        if k not in skip and r.get(k) not in ("", None)
    })
    with csv_path.open("w", newline="") as fh:
        fields = ["method", *param_keys, "val_ari", "val_nmi", "val_f1", "score"]
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for r in combined:
            writer.writerow(r)
    print(f"\nWrote {csv_path}", flush=True)

    json_path = out_dir / "best_hyperparams_robustness.json"
    elapsed = time.perf_counter() - t_start
    protocol_extra = {
        "search": "optuna",
        "sampler": (
            f"TPESampler(multivariate=True, prior_weight={args.prior_weight})"
        ),
        "n_optuna_trials": n_optuna_trials,
        "n_startup_trials": args.n_startup_trials,
        "prior_weight": args.prior_weight,
        "search_spaces": {m: tuner.SEARCH_SPACES[m] for m in args.methods},
        "y_normalization": "frobenius",
        "lambda_z": bench.LAMBDA_Z,
        "lambda_e21_scale": "sqrt(N)",
        "nmi_weight": args.nmi_weight,
        "k": "none" if args.k is None else args.k,
        "k_method": args.k_method,
        "min_k": args.min_k,
        "k_penalty": args.k_penalty,
        "validated_against": "benchmark_sbm_robustness.py",
        "axes": axes,
        "points_per_axis": points_per_axis,
    }
    if args.merge and json_path.exists():
        payload = json.loads(json_path.read_text())
        payload.setdefault("methods", {}).update(selected)
        payload.setdefault("protocol", {}).update({
            "elapsed_seconds": elapsed,
            "retune": {"methods": list(args.methods), **protocol_extra},
        })
    else:
        payload = {
            "protocol": {
                "val_seed": args.seed,
                "robustness_report_seed": 0,
                "trials": n_graph_trials,
                "max_iter_tune": args.max_iter,
                "f1_weight": args.f1_weight,
                "nmi_weight": args.nmi_weight,
                "objective": "mean_ARI + nmi_weight * mean_NMI",
                "elapsed_seconds": elapsed,
                **protocol_extra,
            },
            "methods": selected,
        }
    json_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Wrote {json_path}", flush=True)

    print("\n=== Selected hyperparameters ===", flush=True)
    for name in args.methods:
        info = selected[name]
        print(
            f"{name:24s}  {info['params']}  "
            f"val ARI={info['val_ari']:.3f}  NMI={info['val_nmi']:.3f}",
            flush=True,
        )
    print(
        "\nRun the robustness suite with these params:\n"
        f"  python benchmark_sbm_robustness.py --params {json_path} "
        f"--out-dir results/robustness_tuned",
        flush=True,
    )


if __name__ == "__main__":
    main()