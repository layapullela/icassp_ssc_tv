"""
Robustness benchmark: five sweeps isolating one difficulty axis at a time,
run at both oracle-k and inferred-k (eigengap / ncut), so you can see not
just "which method wins" but "how much does each method degrade once it
has to infer k itself."

This script reuses the SBM generators, ADMM solvers, k-estimation, and
metric code from ``benchmark_sbm.py`` (import as ``bench``) rather than
reimplementing them, so it stays consistent with the main benchmark and
picks up any new SSC-TV variant you register in ``bench.METHOD_SPECS``
automatically.

Axes
----
1. num_clusters   : p_in/p_out fixed, k swept upward, moderate-but-consistent
                    imbalance (max/min block-size ratio fixed at 3x) at every k.
2. num_nodes      : k fixed at 3, mild imbalance fixed, N swept 200 -> 1000.
3. gaussian_noise : structure fixed (two base cases: moderate + weak
                    separation), Gaussian noise sigma swept.
4. imbalance      : k and N fixed, max/min block-size ratio swept upward.
5. density_ratio  : sizes/k fixed, p_in fixed, p_out/p_in ratio swept
                    upward from near-0 (well separated) to near-1 (no signal).

Each (axis, sweep value, trial) draws ONE graph and evaluates every method
under BOTH k modes on that same graph:
    k_mode='oracle'   -> true SBM block count handed to the method
    k_mode='inferred' -> k estimated via --k-method (eigengap by default)
so the oracle-vs-inferred gap is directly comparable per draw, not just
per aggregate.

Usage
-----
    python benchmark_sbm_robustness.py                        # all 5 axes
    python benchmark_sbm_robustness.py --axes num_clusters
    python benchmark_sbm_robustness.py --smoke                # fast sanity check
    python benchmark_sbm_robustness.py --methods OSC SSC-TV-L21-PQ TKSS DP-Y
    python benchmark_sbm_robustness.py --trials 3 --k-method eigengap --min-k 2
    python benchmark_sbm_robustness.py --plot-only --out-dir results/robustness
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import benchmark_sbm as bench  # noqa: E402


# ── Sizing helpers ────────────────────────────────────────────────────────

def sizes_for_ratio(n_total, k, ratio):
    """k block sizes summing to ``n_total`` with max/min size == ``ratio``.

    Geometric spacing between 1x and ``ratio``x (ratio=1 -> perfectly
    balanced). Rounds to ints and pushes any rounding remainder onto the
    largest block so the sizes always sum exactly to n_total.
    """
    if k <= 1:
        return [int(n_total)]
    weights = np.geomspace(1.0, max(ratio, 1.0), k)
    raw = weights / weights.sum() * n_total
    sizes = np.maximum(np.round(raw).astype(int), 1)
    diff = int(n_total) - int(sizes.sum())
    sizes[int(np.argmax(sizes))] += diff
    return sizes.tolist()


# ── Axis 1: number of clusters ───────────────────────────────────────────
# p_in/p_out held constant; k swept; imbalance ratio held constant (3x) at
# every k so the sweep isolates "more blocks", not "more imbalance".

AXIS1_K_GRID = [2, 3, 4, 5, 6, 8, 10, 12, 16, 20]
AXIS1_AVG_BLOCK = 25       # N scales with k so per-block difficulty stays ~fixed
AXIS1_RATIO = 3.0          # moderate, consistent imbalance
AXIS1_P_IN = 0.5
AXIS1_P_OUT = 0.1
AXIS1_LAMBDA = 0.5         # light-moderate background Poisson noise


def axis_num_clusters():
    cases = []
    for k in AXIS1_K_GRID:
        n_total = AXIS1_AVG_BLOCK * k
        sizes = sizes_for_ratio(n_total, k, AXIS1_RATIO)
        cases.append({
            "sweep_value": k,
            "sizes": sizes,
            "p_in": AXIS1_P_IN,
            "p_out": AXIS1_P_OUT,
            "k": k,
            "lambda": AXIS1_LAMBDA,
            "noise_model": "poisson",
            "title": f"k={k}  N={n_total}  sizes~ratio{AXIS1_RATIO}x",
        })
    return cases


# ── Axis 2: number of nodes ───────────────────────────────────────────────
# k fixed at 3, mild fixed imbalance (1.5x), N swept 200 -> 1000.

AXIS2_N_GRID = [200, 300, 400, 600, 800, 1000]
AXIS2_K = 3
AXIS2_RATIO = 1.5
AXIS2_P_IN = 0.5
AXIS2_P_OUT = 0.1
AXIS2_LAMBDA = 0.5


def axis_num_nodes():
    cases = []
    for n_total in AXIS2_N_GRID:
        sizes = sizes_for_ratio(n_total, AXIS2_K, AXIS2_RATIO)
        cases.append({
            "sweep_value": n_total,
            "sizes": sizes,
            "p_in": AXIS2_P_IN,
            "p_out": AXIS2_P_OUT,
            "k": AXIS2_K,
            "lambda": AXIS2_LAMBDA,
            "noise_model": "poisson",
            "title": f"N={n_total}  k={AXIS2_K}  sizes={sizes}",
        })
    return cases


# ── Axis 3: Gaussian noise ────────────────────────────────────────────────
# Structure fixed (two base cases spanning moderate -> weak separation),
# Gaussian sigma swept as the sweep variable itself.

AXIS3_SIGMA_GRID = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0]
AXIS3_STRUCTURES = {
    "moderate": {"sizes": (50, 60, 90), "p_in": 0.5, "p_out": 0.1, "k": 3},
    "weak": {"sizes": (55, 65, 80), "p_in": 0.25, "p_out": 0.12, "k": 3},
}


def axis_gaussian_noise():
    cases = []
    for struct_name, struct in AXIS3_STRUCTURES.items():
        for sigma in AXIS3_SIGMA_GRID:
            cases.append({
                "sweep_value": sigma,
                "sweep_group": struct_name,
                "sizes": list(struct["sizes"]),
                "p_in": struct["p_in"],
                "p_out": struct["p_out"],
                "k": struct["k"],
                "lambda": sigma,
                "noise_model": "gaussian",
                "title": f"Gaussian σ={sigma}  ({struct_name})",
            })
    return cases


# ── Axis 4: imbalance level ───────────────────────────────────────────────
# k and N fixed, p_in/p_out fixed, max/min block-size ratio swept upward.

AXIS4_K = 3
AXIS4_N = 200
AXIS4_RATIO_GRID = [1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0, 30.0, 40.0]
AXIS4_P_IN = 0.5
AXIS4_P_OUT = 0.1
AXIS4_LAMBDA = 0.5


def axis_imbalance():
    cases = []
    for ratio in AXIS4_RATIO_GRID:
        sizes = sizes_for_ratio(AXIS4_N, AXIS4_K, ratio)
        cases.append({
            "sweep_value": ratio,
            "sizes": sizes,
            "p_in": AXIS4_P_IN,
            "p_out": AXIS4_P_OUT,
            "k": AXIS4_K,
            "lambda": AXIS4_LAMBDA,
            "noise_model": "poisson",
            "title": f"imbalance ratio={ratio}x  sizes={sizes}",
        })
    return cases


# ── Axis 5: cluster density (p_out / p_in ratio) ─────────────────────────
# sizes/k fixed, p_in fixed; p_out/p_in ratio swept from well-separated
# (low ratio) toward indistinguishable (ratio -> 1).

AXIS5_SIZES = (50, 60, 90)
AXIS5_K = 3
AXIS5_P_IN = 0.5
AXIS5_RATIO_GRID = [0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8]
AXIS5_LAMBDA = 0.5


def axis_density_ratio():
    cases = []
    for ratio in AXIS5_RATIO_GRID:
        p_out = AXIS5_P_IN * ratio
        cases.append({
            "sweep_value": ratio,
            "sizes": list(AXIS5_SIZES),
            "p_in": AXIS5_P_IN,
            "p_out": p_out,
            "k": AXIS5_K,
            "lambda": AXIS5_LAMBDA,
            "noise_model": "poisson",
            "title": f"p_out/p_in={ratio}  (p_in={AXIS5_P_IN}, p_out={p_out:.3f})",
        })
    return cases


AXES = {
    "num_clusters": {
        "build": axis_num_clusters,
        "xlabel": "number of clusters (k)",
        "desc": "Robustness to number of clusters (density fixed, moderate "
                "consistent imbalance, k swept upward).",
    },
    "num_nodes": {
        "build": axis_num_nodes,
        "xlabel": "N (number of nodes)",
        "desc": "Robustness to graph size (k=3 fixed, mild imbalance fixed, "
                "N swept 200 -> 1000).",
    },
    "gaussian_noise": {
        "build": axis_gaussian_noise,
        "xlabel": "Gaussian noise σ",
        "desc": "Robustness to Gaussian noise (structure fixed at moderate "
                "and weak separation, σ swept).",
    },
    "imbalance": {
        "build": axis_imbalance,
        "xlabel": "max/min block-size ratio",
        "desc": "Robustness to imbalance (k=3, N=200 fixed, block-size "
                "ratio swept upward).",
    },
    "density_ratio": {
        "build": axis_density_ratio,
        "xlabel": "p_out / p_in",
        "desc": "Robustness to cluster density (sizes/k fixed, p_in fixed, "
                "p_out/p_in swept from well-separated to indistinguishable).",
    },
}


# ── Run harness ────────────────────────────────────────────────────────────

FIELDNAMES = [
    "axis", "sweep_group", "sweep_value", "trial", "method", "k_mode",
    "k_true", "k", "ari", "nmi", "seconds", "error",
]


def run_axis(axis_name, methods, n_trials, seed, k_method, min_k, penalty,
            writer, fh):
    cfgs = AXES[axis_name]["build"]()
    rows = []
    n_jobs = len(cfgs) * n_trials * len(methods) * 2  # oracle + inferred
    done = 0
    t_start = time.perf_counter()

    for ci, cfg in enumerate(cfgs):
        sweep_group = cfg.get("sweep_group", "")
        for trial in range(n_trials):
            rng = np.random.default_rng(
                seed + 10_000 * ci + int(round(cfg["lambda"] * 1000)) * 17 + trial
            )
            A, labels = bench.generate_sbm(
                cfg["sizes"], cfg["p_in"], cfg["p_out"], rng,
            )
            if cfg["noise_model"] == "gaussian":
                Y = bench.add_gaussian_noise(A, cfg["lambda"], rng)
            else:
                Y = bench.add_poisson_noise(A, cfg["lambda"], rng)
            Y = bench.normalize_Y(Y)
            k_true = int(cfg["k"])

            for method_name, solver in methods:
                for k_mode, run_k in (("oracle", k_true), ("inferred", None)):
                    rec = {
                        "axis": axis_name,
                        "sweep_group": sweep_group,
                        "sweep_value": cfg["sweep_value"],
                        "trial": trial,
                        "method": method_name,
                        "k_mode": k_mode,
                        "k_true": k_true,
                        "k": "",
                        "ari": float("nan"),
                        "nmi": float("nan"),
                        "seconds": float("nan"),
                        "error": "",
                    }
                    try:
                        out = bench.run_one(
                            Y, labels, run_k, method_name, solver, None,
                            k_method=k_method, min_k=min_k, penalty=penalty,
                        )
                        rec["k"] = out["k"]
                        rec["ari"] = out["ari"]
                        rec["nmi"] = out["nmi"]
                        rec["seconds"] = out["seconds"]
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
                        f"[{axis_name} {done}/{n_jobs}] "
                        f"{cfg['title']}  trial={trial}  {method_name}  "
                        f"k_mode={k_mode}  k={rec['k']}  ARI={rec['ari']:.3f}"
                        if rec["error"] == ""
                        else f"[{axis_name} {done}/{n_jobs}] "
                             f"{method_name} FAILED: {rec['error']}"
                    )
                    print(f"{status}   ({elapsed:.0f}s elapsed, ETA {eta:.0f}s)",
                          flush=True)
    return rows


def _finite(vals):
    arr = np.asarray(vals, dtype=float)
    return arr[np.isfinite(arr)]


def print_axis_summary(axis_name, rows, method_names):
    print(f"\n=== {axis_name}: ARI mean (oracle-k | inferred-k)  "
          f"['|Δk|' = mean |k_hat - k_true| under inferred-k] ===", flush=True)
    groups = sorted({r["sweep_group"] for r in rows}) or [""]
    for group in groups:
        if group:
            print(f"\n  -- group: {group} --", flush=True)
        sub_rows = [r for r in rows if r["sweep_group"] == group]
        values = sorted({r["sweep_value"] for r in sub_rows})
        header = f"{'value':>10}" + "".join(f"{m:>26}" for m in method_names)
        print(header, flush=True)
        for v in values:
            cells = [f"{v:>10}"]
            for m in method_names:
                orc = _finite([r["ari"] for r in sub_rows
                               if r["sweep_value"] == v and r["method"] == m
                               and r["k_mode"] == "oracle" and r["error"] == ""])
                inf = _finite([r["ari"] for r in sub_rows
                               if r["sweep_value"] == v and r["method"] == m
                               and r["k_mode"] == "inferred" and r["error"] == ""])
                dk = _finite([
                    abs(r["k"] - r["k_true"]) for r in sub_rows
                    if r["sweep_value"] == v and r["method"] == m
                    and r["k_mode"] == "inferred" and r["error"] == ""
                    and r["k"] != ""
                ])
                o_s = f"{orc.mean():.3f}" if orc.size else "n/a"
                i_s = f"{inf.mean():.3f}" if inf.size else "n/a"
                dk_s = f"{dk.mean():.2f}" if dk.size else "n/a"
                cells.append(f"{o_s}|{i_s} Δk{dk_s}".rjust(26))
            print("".join(cells), flush=True)


def _row_score(row, score_fn):
    if score_fn is None:
        return row["ari"]
    return score_fn(row)


def plot_axis(axis_name, rows, method_names, out_dir):
    _plot_axis_score(
        axis_name, rows, method_names, out_dir,
        score_fn=None, ylabel="ARI",
        filename=f"robustness_{axis_name}.png",
    )
    _plot_axis_score(
        axis_name, rows, method_names, out_dir,
        score_fn=bench.row_sqrt_ari_nmi, ylabel=bench.SQRT_ARI_NMI_LABEL,
        filename=f"robustness_{axis_name}_sqrt_ari_nmi.png",
    )
    _plot_axis_kerror(axis_name, rows, method_names, out_dir)


def _plot_axis_score(axis_name, rows, method_names, out_dir, score_fn, ylabel,
                     filename):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    markers = ["o", "s", "D", "^", "v", "P", "X", "*"]
    groups = sorted({r["sweep_group"] for r in rows}) or [""]
    xlabel = AXES[axis_name]["xlabel"]

    fig, axes = plt.subplots(
        len(groups), 2, figsize=(10, 4.0 * len(groups)), squeeze=False,
    )
    for gi, group in enumerate(groups):
        sub_rows = [r for r in rows if r["sweep_group"] == group]
        values = sorted({r["sweep_value"] for r in sub_rows})
        for col, k_mode in enumerate(("oracle", "inferred")):
            ax = axes[gi][col]
            for mi, m in enumerate(method_names):
                means, stds = [], []
                for v in values:
                    a = _finite([
                        _row_score(r, score_fn) for r in sub_rows
                        if r["sweep_value"] == v and r["method"] == m
                        and r["k_mode"] == k_mode and r["error"] == ""
                    ])
                    means.append(a.mean() if a.size else np.nan)
                    stds.append(a.std(ddof=1) if a.size > 1 else 0.0)
                ax.errorbar(
                    values, means, yerr=stds, marker=markers[mi % len(markers)],
                    capsize=3, label=m, linewidth=1.4,
                )
            title = f"k = {k_mode}"
            if group:
                title = f"{group}  ({title})"
            ax.set_title(title, fontsize=10)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_ylim(-0.05, 1.05)
            ax.grid(True, alpha=0.3)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=min(len(labels), 5),
               frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle(AXES[axis_name]["desc"], fontsize=10, y=1.06)
    fig.tight_layout()
    out_path = Path(out_dir) / filename
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}", flush=True)


def _plot_axis_kerror(axis_name, rows, method_names, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    markers = ["o", "s", "D", "^", "v", "P", "X", "*"]
    groups = sorted({r["sweep_group"] for r in rows}) or [""]
    xlabel = AXES[axis_name]["xlabel"]

    # |k_hat - k_true| vs sweep value (inferred-k only).
    fig, axes = plt.subplots(1, len(groups), figsize=(5.5 * len(groups), 4.2),
                             squeeze=False)
    for gi, group in enumerate(groups):
        ax = axes[0][gi]
        sub_rows = [r for r in rows if r["sweep_group"] == group]
        values = sorted({r["sweep_value"] for r in sub_rows})
        for mi, m in enumerate(method_names):
            means = []
            for v in values:
                dk = _finite([
                    abs(r["k"] - r["k_true"]) for r in sub_rows
                    if r["sweep_value"] == v and r["method"] == m
                    and r["k_mode"] == "inferred" and r["error"] == ""
                    and r["k"] != ""
                ])
                means.append(dk.mean() if dk.size else np.nan)
            ax.plot(values, means, marker=markers[mi % len(markers)],
                    linewidth=1.4, label=m)
        ax.set_title(f"k-estimation error{'  (' + group + ')' if group else ''}",
                     fontsize=10)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("mean |k_hat - k_true|")
        ax.grid(True, alpha=0.3)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=min(len(labels), 5),
               frameon=False, bbox_to_anchor=(0.5, 1.05))
    fig.tight_layout()
    out_path = Path(out_dir) / f"robustness_{axis_name}_kerror.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}", flush=True)


def load_csv(path):
    rows = []
    with Path(path).open(newline="") as fh:
        for r in csv.DictReader(fh):
            rec = dict(r)
            rec["sweep_value"] = float(rec["sweep_value"])
            rec["trial"] = int(rec["trial"])
            rec["k_true"] = int(rec["k_true"]) if rec["k_true"] else 0
            if rec.get("k") not in (None, "", "nan"):
                rec["k"] = int(float(rec["k"]))
            for key in ("ari", "nmi", "seconds"):
                rec[key] = float(rec[key]) if rec[key] not in ("", "nan") else float("nan")
            rows.append(rec)
    return rows


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--axes", nargs="+", default=list(AXES),
                   choices=list(AXES))
    p.add_argument("--methods", nargs="+", default=list(bench.DEFAULT_METHODS),
                   choices=[s["name"] for s in bench.METHOD_SPECS])
    p.add_argument("--trials", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", type=str,
                   default=str(ROOT / "results" / "robustness"))
    p.add_argument("--k-method", type=str, default="eigengap",
                   choices=["eigengap", "ncut"])
    p.add_argument("--min-k", type=int, default=2)
    p.add_argument("--k-penalty", type=float, default=0.0)
    p.add_argument("--params", type=str, default=None,
                   help="JSON from tune_hyperparams.py with per-method params.")
    p.add_argument("--smoke", action="store_true",
                   help="1 trial, num_clusters axis only, small k grid.")
    p.add_argument("--plot-only", action="store_true",
                   help="Recompute summary/plots from out-dir/robustness.csv.")
    p.add_argument(
        "--append", action="store_true",
        help="Run only --methods and merge into existing robustness.csv "
             "(keep rows for other methods).",
    )
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "robustness.csv"

    if args.plot_only:
        rows = load_csv(csv_path)
        for axis_name in args.axes:
            sub = [r for r in rows if r["axis"] == axis_name]
            if not sub:
                continue
            names = [s["name"] for s in bench.METHOD_SPECS
                     if any(r["method"] == s["name"] for r in sub)]
            print_axis_summary(axis_name, sub, names)
            plot_axis(axis_name, sub, names, out_dir)
        return

    axes = args.axes
    if args.smoke:
        axes = ["num_clusters"]
        global AXIS1_K_GRID
        AXIS1_K_GRID = [2, 4, 8]
        args.trials = 1
        print("Smoke mode: num_clusters axis, k in [2,4,8], 1 trial.", flush=True)

    overrides = bench.load_param_overrides(args.params) if args.params else {}
    methods = bench.build_methods(overrides=overrides, names=args.methods)

    kept_rows = []
    if args.append and csv_path.exists():
        kept_rows = [
            r for r in load_csv(csv_path) if r.get("method") not in args.methods
        ]
        print(
            f"Append: keeping {len(kept_rows)} existing rows; "
            f"replacing methods {args.methods}",
            flush=True,
        )

    all_rows = list(kept_rows)
    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDNAMES)
        writer.writeheader()
        for r in kept_rows:
            writer.writerow(r)
        for axis_name in axes:
            print(f"\n########## AXIS: {axis_name} ##########", flush=True)
            print(AXES[axis_name]["desc"], flush=True)
            rows = run_axis(
                axis_name, methods, args.trials, args.seed,
                args.k_method, args.min_k, args.k_penalty, writer, fh,
            )
            all_rows.extend(rows)
            # Summarize / plot all methods present after this axis (merged).
            axis_rows = [r for r in all_rows if r["axis"] == axis_name]
            plot_names = []
            for s in bench.METHOD_SPECS:
                if any(r["method"] == s["name"] for r in axis_rows):
                    plot_names.append(s["name"])
            print_axis_summary(axis_name, axis_rows, plot_names)
            try:
                plot_axis(axis_name, axis_rows, plot_names, out_dir)
            except Exception:
                traceback.print_exc()
                print("Plotting skipped (matplotlib missing or plot error).",
                      flush=True)

    print(f"\nWrote {csv_path}", flush=True)


if __name__ == "__main__":
    main()