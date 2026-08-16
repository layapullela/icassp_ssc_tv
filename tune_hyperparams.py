"""
Held-out hyperparameter search for the seven ADMM methods.

Protocol
--------
One hyperparameter vector per method (not per case / not per noise level).
Validation graphs use a different RNG seed from the test benchmark so the
selected values are not fit on the numbers we later report.

Validation design
    cases   : all four SBM cases
    λ       : {0.0, 0.10}   (noiseless + moderate Poisson; not the full test grid)
    trials  : 2
    seed    : 1_000_003     (benchmark test seed is 0)
    max_iter: 80            (faster; winners are re-evaluated at 200)

Objective (to maximize)
    score = mean ARI  +  0.25 × mean case-4 outlier F1
          (F1 is 0 when a config never produces a finite F1)

ARI is primary so clustering quality still dominates; the F1 bonus is
there so SSC variants are not rewarded for zeroing E.

Usage
-----
    python tune_hyperparams.py
    python tune_hyperparams.py --quick          # 1 trial, cases 1 and 4 only
    python benchmark_sbm.py --params results/best_hyperparams.json \\
        --out-dir results/tuned
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import time
from pathlib import Path

import numpy as np

import benchmark_sbm as bench

ROOT = Path(__file__).resolve().parent

# Compact log-style grids.  Defaults from each solver file are included.
# OSC is searched as densely as the 3-weight SSC grids (30 vs 27 points):
# its objective only has two model weights (λ1 on ||Z||_1, λ2 on ||ZR||_{2,1}).
GRIDS = {
    "OSC": {
        "lambda1": (0.01, 0.05, 0.1, 0.5, 1.0),
        "lambda2": (0.1, 0.5, 1.0, 2.0, 5.0, 10.0),
    },
    "SSC-TV": {
        "lambda_e": (0.05, 0.2, 1.0),
        "lambda_z": (0.05, 0.1, 0.5),
        "gamma": (0.01, 0.1, 0.5),
    },
    "SSC-TV-L21-P": {
        "lambda_e": (0.05, 0.2, 1.0),
        "lambda_z": (0.05, 0.1, 0.5),
        "gamma": (0.01, 0.1, 0.5),
    },
    "SSC-TV-L21-PQ": {
        "lambda_e": (0.05, 0.2, 1.0),
        "lambda_z": (0.05, 0.1, 0.5),
        "gamma": (0.01, 0.1, 0.5),
    },
    # λ_z is shared with the matching SSC-TV default scale; γ and the two
    # E-weights are the degrees of freedom that actually distinguish E1E21.
    "SSC-TV-E1E21": {
        "lambda_e1": (0.05, 0.2, 1.0),
        "lambda_e21": (0.05, 0.2, 1.0),
        "lambda_z": (0.1,),
        "gamma": (0.01, 0.1, 0.5),
    },
    "SSC-TV-E1E21-L21-P": {
        "lambda_e1": (0.05, 0.2, 1.0),
        "lambda_e21": (0.05, 0.2, 1.0),
        "lambda_z": (0.1,),
        "gamma": (0.01, 0.1, 0.5),
    },
    "SSC-TV-E1E21-L21-PQ": {
        "lambda_e1": (0.05, 0.2, 1.0),
        "lambda_e21": (0.05, 0.2, 1.0),
        "lambda_z": (0.1,),
        "gamma": (0.01, 0.1, 0.5),
    },
}

F1_WEIGHT = 0.25
VAL_SEED = 1_000_003
TUNE_MAX_ITER = 80


def configs_from_grid(grid):
    keys = list(grid)
    for values in itertools.product(*[grid[k] for k in keys]):
        yield dict(zip(keys, values))


def spec_by_name(name):
    for spec in bench.METHOD_SPECS:
        if spec["name"] == name:
            return spec
    raise KeyError(name)


def build_val_graphs(cases, lambdas, n_trials, seed):
    graphs = []
    for case_name in cases:
        cfg = bench.CASES[case_name]
        case_id = list(bench.CASES).index(case_name)
        for lam in lambdas:
            for trial in range(n_trials):
                rng = np.random.default_rng(
                    seed + 10_000 * case_id
                    + int(round(lam * 1000)) * 17
                    + trial
                )
                Y, labels, outlier_mask = bench.make_observation(cfg, lam, rng)
                graphs.append({
                    "case": case_name,
                    "lambda": lam,
                    "trial": trial,
                    "k": cfg["k"],
                    "Y": Y,
                    "labels": labels,
                    "outlier_mask": outlier_mask,
                })
    return graphs


def eval_solver(solver, name, graphs):
    aris, f1s = [], []
    for g in graphs:
        try:
            rec = bench.run_one(
                g["Y"], g["labels"], g["k"], name, solver, g["outlier_mask"],
            )
        except Exception:
            aris.append(float("nan"))
            continue
        aris.append(rec["ari"])
        if np.isfinite(rec["f1"]):
            f1s.append(rec["f1"])
    ari = float(np.nanmean(aris)) if aris else float("nan")
    f1 = float(np.mean(f1s)) if f1s else 0.0
    score = (0.0 if not np.isfinite(ari) else ari) + F1_WEIGHT * f1
    return ari, f1, score


def tune_method(name, grid, graphs, max_iter):
    spec = spec_by_name(name)
    best = None
    rows = []
    configs = list(configs_from_grid(grid))
    t0 = time.perf_counter()
    for i, extra in enumerate(configs, 1):
        kw = dict(spec["defaults"])
        kw["max_iter"] = max_iter
        kw.update(extra)
        solver = bench.make_solver(spec, kw)
        ari, f1, score = eval_solver(solver, name, graphs)
        rec = {
            "method": name,
            **{k: extra[k] for k in extra},
            "val_ari": ari,
            "val_f1": f1,
            "score": score,
        }
        rows.append(rec)
        elapsed = time.perf_counter() - t0
        eta = (elapsed / i) * (len(configs) - i)
        print(
            f"  [{i}/{len(configs)}] {name}  {extra}  "
            f"ARI={ari:.3f}  F1={f1:.3f}  score={score:.3f}  "
            f"ETA {eta:.0f}s",
            flush=True,
        )
        if best is None or score > best["score"] + 1e-12:
            best = {
                "params": extra,
                "val_ari": ari,
                "val_f1": f1,
                "score": score,
            }
        elif best is not None and abs(score - best["score"]) <= 1e-12:
            # Tie-break: prefer the file default when scores match, else fewer
            # large weights (sum of log params as a roughness penalty).
            def roughness(p):
                return sum(abs(np.log10(max(v, 1e-12))) for v in p.values())
            if roughness(extra) < roughness(best["params"]):
                best = {
                    "params": extra,
                    "val_ari": ari,
                    "val_f1": f1,
                    "score": score,
                }
    return best, rows


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir", type=str, default=str(ROOT / "results"))
    p.add_argument("--seed", type=int, default=VAL_SEED)
    p.add_argument("--trials", type=int, default=2)
    p.add_argument("--max-iter", type=int, default=TUNE_MAX_ITER)
    p.add_argument(
        "--lambdas", type=float, nargs="+", default=[0.0, 0.10],
    )
    p.add_argument(
        "--cases", nargs="+", default=list(bench.CASES),
        choices=list(bench.CASES),
    )
    p.add_argument(
        "--methods", nargs="+",
        default=[s["name"] for s in bench.METHOD_SPECS],
        choices=[s["name"] for s in bench.METHOD_SPECS],
    )
    p.add_argument(
        "--quick", action="store_true",
        help="1 trial, cases 1 and 4 only — sanity-check the search loop.",
    )
    p.add_argument(
        "--f1-weight", type=float, default=F1_WEIGHT,
        help="Weight on mean case-4 F1 in the selection score (default 0.25).",
    )
    p.add_argument(
        "--merge", action="store_true",
        help="Update existing best_hyperparams.json / tune_grid.csv instead of replacing them.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    global F1_WEIGHT
    F1_WEIGHT = args.f1_weight

    cases = args.cases
    n_trials = args.trials
    if args.quick:
        cases = ["1_three_block", "4_three_block_outliers"]
        n_trials = 1
        print("Quick mode: cases 1+4, 1 trial.", flush=True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"Building validation graphs  "
        f"({len(cases)} cases × {len(args.lambdas)} λ × {n_trials} trials, "
        f"seed={args.seed}) …",
        flush=True,
    )
    graphs = build_val_graphs(cases, args.lambdas, n_trials, args.seed)
    print(f"  {len(graphs)} graphs, Y shape {graphs[0]['Y'].shape}", flush=True)

    n_configs = sum(len(list(configs_from_grid(GRIDS[m]))) for m in args.methods)
    n_jobs = n_configs * len(graphs)
    print(
        f"Search: {n_configs} configs × {len(graphs)} graphs "
        f"= {n_jobs} ADMM runs  (max_iter={args.max_iter})",
        flush=True,
    )

    all_rows = []
    selected = {}
    t_start = time.perf_counter()
    for name in args.methods:
        print(f"\n=== {name}  ({len(list(configs_from_grid(GRIDS[name])))} configs) ===",
              flush=True)
        best, rows = tune_method(
            name, GRIDS[name], graphs, args.max_iter,
        )
        all_rows.extend(rows)
        selected[name] = best
        print(
            f"  → best {best['params']}  "
            f"ARI={best['val_ari']:.3f}  F1={best['val_f1']:.3f}  "
            f"score={best['score']:.3f}",
            flush=True,
        )

    # Wide CSV: union of param keys.  --merge keeps rows for methods not retuned.
    csv_path = out_dir / "tune_grid.csv"
    existing_rows = []
    if args.merge and csv_path.exists():
        with csv_path.open(newline="") as fh:
            existing_rows = [
                r for r in csv.DictReader(fh) if r.get("method") not in args.methods
            ]
    combined = existing_rows + all_rows
    param_keys = sorted({
        k for r in combined for k in r
        if k not in ("method", "val_ari", "val_f1", "score") and r.get(k) not in ("", None)
    })
    with csv_path.open("w", newline="") as fh:
        fields = ["method", *param_keys, "val_ari", "val_f1", "score"]
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for r in combined:
            writer.writerow(r)
    print(f"\nWrote {csv_path}", flush=True)

    json_path = out_dir / "best_hyperparams.json"
    if args.merge and json_path.exists():
        payload = json.loads(json_path.read_text())
        payload.setdefault("methods", {}).update(selected)
        payload.setdefault("protocol", {})["osc_retune"] = {
            "n_configs": n_configs,
            "elapsed_seconds": time.perf_counter() - t_start,
            "grid": GRIDS.get("OSC"),
        }
    else:
        payload = {
            "protocol": {
                "val_seed": args.seed,
                "test_seed": 0,
                "cases": cases,
                "lambdas": list(args.lambdas),
                "trials": n_trials,
                "max_iter_tune": args.max_iter,
                "f1_weight": F1_WEIGHT,
                "objective": "mean_ARI + f1_weight * mean_case4_F1",
                "elapsed_seconds": time.perf_counter() - t_start,
                "n_configs": {m: len(list(configs_from_grid(GRIDS[m])))
                              for m in args.methods},
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
            f"val ARI={info['val_ari']:.3f}  F1={info['val_f1']:.3f}",
            flush=True,
        )
    print(
        "\nRe-run the held-out test with:\n"
        f"  python benchmark_sbm.py --params {json_path} --out-dir results/tuned",
        flush=True,
    )


if __name__ == "__main__":
    main()
