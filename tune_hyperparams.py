"""
Held-out hyperparameter search for OSC, the SSC-TV ADMM variants, and TKSS.

Protocol
--------
One hyperparameter vector per method (not per case / not per noise level).
Validation graphs use a different RNG seed from the test benchmark so the
selected values are not fit on the numbers we later report.

Each Y is Frobenius-normalised.  SSC-TV λ_z is fixed at 1 (not searched);
other penalties are log-uniform on [1e-5, 10].  λ_e21 is tuned pre-scale
and multiplied by sqrt(N) at solve time.

Validation design
    cases   : original four SBM cases (held out from the size/probability sweeps)
    λ       : {0.0, 0.10, 0.50, 1.00}  (noiseless, moderate, and high Poisson;
              not the full test grid — 0.05 / 0.20 / 0.30 / 0.75 stay held out)
    trials  : 2
    seed    : 1_000_003     (benchmark test seed is 0)
    max_iter: 80            (faster; winners are re-evaluated at 200)
    search  : Optuna TPE over every method (40 trials/method)

Objective (to maximize)
    score = mean ARI + (also can look at F1 score for outlier detection, but this is disabled for now)
          (F1 is 0 when a config never produces a finite F1)

ARI is primary so clustering quality still dominates; the F1 bonus is
there so SSC variants are not rewarded for zeroing E.

Usage
-----
    python tune_hyperparams.py                  # Optuna TPE, all methods
    python tune_hyperparams.py --quick          # 1 graph trial, cases 1 and 4 only
    python tune_hyperparams.py --retune OSC     # one method; merge into existing JSON
    python benchmark_sbm.py --params results/best_hyperparams.json \\
        --out-dir results/tuned
"""

from __future__ import annotations

import argparse
import csv
import json
import time
import warnings
from pathlib import Path

import numpy as np
import optuna

import benchmark_sbm as bench

ROOT = Path(__file__).resolve().parent

F1_WEIGHT = 0 # TODO: decide if we want this in the loss
VAL_SEED = 1_000_003
TUNE_MAX_ITER = 80
N_TRIALS = 40
N_STARTUP_TRIALS = 10
TUNE_CASES = [
    "1_three_block",
    "2_three_block_sparse",
    "3_five_block",
    "4_three_block_outliers",
]
TUNE_LAMBDAS = [0.0, 0.10, 0.50, 1.00]

# Log-uniform Optuna ranges.  λ_z is fixed at 1 (not searched); λ_e21 is the
# pre-√N scale.  One space per METHOD_SPECS entry in benchmark_sbm.py.
PARAM_RANGE = (1e-5, 10.0)
SEARCH_SPACES = {
    "OSC": {
        "lambda1": PARAM_RANGE,
        "lambda2": PARAM_RANGE,
    },
    "SSC-TV": {
        "lambda_e": PARAM_RANGE,
        "gamma": PARAM_RANGE,
    },
    "SSC-TV-L21-P": {
        "lambda_e": PARAM_RANGE,
        "gamma": PARAM_RANGE,
    },
    "SSC-TV-L21-PQ": {
        "lambda_e": PARAM_RANGE,
        "gamma": PARAM_RANGE,
    },
    "SSC-TV-E1E21": {
        "lambda_e1": PARAM_RANGE,
        "lambda_e21": PARAM_RANGE,
        "gamma": PARAM_RANGE,
    },
    "SSC-TV-E1E21-L21-P": {
        "lambda_e1": PARAM_RANGE,
        "lambda_e21": PARAM_RANGE,
        "gamma": PARAM_RANGE,
    },
    "SSC-TV-E1E21-L21-PQ": {
        "lambda_e1": PARAM_RANGE,
        "lambda_e21": PARAM_RANGE,
        "gamma": PARAM_RANGE,
    },
    "TKSS": {
        "lam": PARAM_RANGE,
        "s": (1, 8),
        "d": (1, 6),
    },
}
INT_SEARCH_KEYS = {"s", "d"}


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


def _eval_config(name, extra, graphs, max_iter):
    spec = spec_by_name(name)
    kw = dict(spec["defaults"])
    if spec["kind"] != "tkss":
        kw["max_iter"] = max_iter
    kw.update(extra)
    if spec["kind"] == "ssc":
        kw["lambda_z"] = bench.LAMBDA_Z
    solver = bench.make_solver(spec, kw)
    return eval_solver(solver, name, graphs)


def tune_method_optuna(
    name, graphs, max_iter, n_trials, n_startup, seed, storage_path, resume,
):
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
        ari, f1, score = _eval_config(name, extra, graphs, max_iter)
        trial.set_user_attr("val_ari", ari)
        trial.set_user_attr("val_f1", f1)
        if not np.isfinite(score):
            return -1.0
        return score

    def callback(study, trial):
        done = n_finished(study)
        elapsed = time.perf_counter() - t0
        left = max(target - done, 0)
        eta = (elapsed / max(done - n_done_start, 1)) * left
        value = trial.value
        score_s = f"{value:.3f}" if value is not None and np.isfinite(value) else "nan"
        try:
            best_s = f"{study.best_value:.3f}"
        except ValueError:
            best_s = "n/a"
        ari = trial.user_attrs.get("val_ari")
        f1 = trial.user_attrs.get("val_f1")
        ari_s = f"{ari:.3f}" if ari is not None and np.isfinite(ari) else "nan"
        f1_s = f"{f1:.3f}" if f1 is not None and np.isfinite(f1) else "nan"
        print(
            f"  [{done}/{target}] {name}  {trial.params}  "
            f"ARI={ari_s}  F1={f1_s}  score={score_s}  best={best_s}  "
            f"ETA {eta:.0f}s",
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
        "val_ari": float(best_trial.user_attrs.get("val_ari", best_trial.value)),
        "val_f1": float(best_trial.user_attrs.get("val_f1", 0.0)),
        "score": float(best_trial.value),
        "n_trials": len(study.trials),
    }
    rows = []
    for t in study.trials:
        rec = {
            "method": name,
            "trial": t.number,
            "val_ari": t.user_attrs.get("val_ari", t.value),
            "val_f1": t.user_attrs.get("val_f1"),
            "score": t.value,
        }
        rec.update(t.params)
        rows.append(rec)
    return best, rows


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir", type=str, default=str(ROOT / "results"))
    p.add_argument("--seed", type=int, default=VAL_SEED)
    p.add_argument("--trials", type=int, default=2)
    p.add_argument("--max-iter", type=int, default=TUNE_MAX_ITER)
    p.add_argument(
        "--lambdas", type=float, nargs="+", default=list(TUNE_LAMBDAS),
        help="Poisson rates on the tune graphs (default: 0, 0.10, 0.50, 1.00).",
    )
    p.add_argument(
        "--cases", nargs="+", default=list(TUNE_CASES),
        choices=list(bench.CASES),
    )
    p.add_argument(
        "--methods", nargs="+",
        default=[s["name"] for s in bench.METHOD_SPECS],
        choices=[s["name"] for s in bench.METHOD_SPECS],
    )
    p.add_argument(
        "--retune", nargs="+", default=None,
        choices=[s["name"] for s in bench.METHOD_SPECS],
        metavar="METHOD",
        help="Only search these methods and merge the winners into existing "
             "best_hyperparams.json / tune_grid.csv. Example: --retune OSC",
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
        "--quick", action="store_true",
        help="1 graph trial, cases 1 and 4 only — sanity-check the search loop.",
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
    if args.retune:
        args.methods = list(args.retune)
        args.merge = True
        print(f"Retune: {', '.join(args.methods)}  (merging into existing files)",
              flush=True)

    cases = args.cases
    n_graph_trials = args.trials
    n_optuna_trials = args.n_trials
    if args.quick:
        cases = ["1_three_block", "4_three_block_outliers"]
        n_graph_trials = 1
        if args.n_trials == N_TRIALS:
            n_optuna_trials = 3
        print(
            f"Quick mode: cases 1+4, 1 graph trial, {n_optuna_trials} Optuna trials.",
            flush=True,
        )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"Building validation graphs  "
        f"({len(cases)} cases × {len(args.lambdas)} λ × {n_graph_trials} trials, "
        f"seed={args.seed}) …",
        flush=True,
    )
    graphs = build_val_graphs(cases, args.lambdas, n_graph_trials, args.seed)
    print(f"  {len(graphs)} graphs, Y shape {graphs[0]['Y'].shape}", flush=True)

    n_jobs = n_optuna_trials * len(args.methods) * len(graphs)
    print(
        f"Search: Optuna TPE, {n_optuna_trials} trials × {len(args.methods)} methods "
        f"× {len(graphs)} graphs = {n_jobs} solver runs  "
        f"(max_iter={args.max_iter})",
        flush=True,
    )

    all_rows = []
    selected = {}
    t_start = time.perf_counter()
    storage_path = out_dir / "optuna.db"
    for name in args.methods:
        print(
            f"\n=== {name}  (Optuna TPE, {n_optuna_trials} trials × "
            f"{len(graphs)} graphs, max_iter={args.max_iter}) ===",
            flush=True,
        )
        best, rows = tune_method_optuna(
            name, graphs, args.max_iter,
            n_trials=n_optuna_trials,
            n_startup=args.n_startup_trials,
            seed=args.tune_seed,
            storage_path=storage_path,
            resume=args.resume,
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
    skip = {"method", "val_ari", "val_f1", "score"}
    param_keys = sorted({
        k for r in combined for k in r
        if k not in skip and r.get(k) not in ("", None)
    })
    with csv_path.open("w", newline="") as fh:
        fields = ["method", *param_keys, "val_ari", "val_f1", "score"]
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for r in combined:
            writer.writerow(r)
    print(f"\nWrote {csv_path}", flush=True)

    json_path = out_dir / "best_hyperparams.json"
    elapsed = time.perf_counter() - t_start
    protocol_extra = {
        "search": "optuna",
        "sampler": "TPESampler(multivariate=True)",
        "n_optuna_trials": n_optuna_trials,
        "n_startup_trials": args.n_startup_trials,
        "search_spaces": {m: SEARCH_SPACES[m] for m in args.methods},
        "y_normalization": "frobenius",
        "lambda_z": bench.LAMBDA_Z,
        "lambda_e21_scale": "sqrt(N)",
    }
    if args.merge and json_path.exists():
        payload = json.loads(json_path.read_text())
        payload.setdefault("methods", {}).update(selected)
        payload.setdefault("protocol", {}).update({
            "elapsed_seconds": elapsed,
            "retune": {
                "methods": list(args.methods),
                **protocol_extra,
            },
        })
    else:
        payload = {
            "protocol": {
                "val_seed": args.seed,
                "test_seed": 0,
                "cases": cases,
                "lambdas": list(args.lambdas),
                "trials": n_graph_trials,
                "max_iter_tune": args.max_iter,
                "f1_weight": F1_WEIGHT,
                "objective": "mean_ARI + f1_weight * mean_case4_F1",
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
