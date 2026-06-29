"""Re-test previously generated solvers on additional TSPLIB instances.

Loads every solver saved under ``genai_experiments/generated_code/`` from the
original experiment matrix, runs each one on a configurable set of instances
(default: pr439 and pr1002), and appends per-instance rows to
``experiment_log.csv``. Nothing about the original generations is changed;
this only extends the evaluation grid with new test instances.

Useful when Stefan asks for a harder differentiation than the original
51 / 52 / 130 / 198-city ladder.

Run from the project root:

    python -m genai_experiments.run_new_instances --instances pr439 pr1002 --timeout 300
"""

import argparse
from pathlib import Path

import pandas as pd

from genai_experiments.executor import run_generated_solver
from genai_experiments.run_experiments import (
    GENERATED_CODE_DIR,
    LOG_PATH,
    STRATEGY_DIR,
    TSPLIB_OPTIMA,
)
from src.utils.distance import distance_matrix
from src.utils.parser import parse_tsp

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "tsplib"

# Folder name on disk -> canonical strategy name used in the log
DIR_TO_STRATEGY = {v: k for k, v in STRATEGY_DIR.items()}
GENERATION_METADATA = [
    "refinement_rounds",
    "input_tokens",
    "output_tokens",
    "generation_error",
]


def discover_solvers() -> list[dict]:
    """Walk generated_code/ and yield one entry per saved solver."""
    entries = []
    for path in sorted(GENERATED_CODE_DIR.rglob("*.py")):
        # Expected layout: generated_code/{model}/{strategy_dir}/run{N}/{algorithm}.py
        model_key = path.parents[2].name
        strategy_dir = path.parents[1].name
        run_dir = path.parents[0].name
        algorithm = path.stem
        if not run_dir.startswith("run"):
            continue
        run_idx = int(run_dir.removeprefix("run"))
        strategy = DIR_TO_STRATEGY.get(strategy_dir, strategy_dir)
        entries.append({
            "model": model_key,
            "strategy": strategy,
            "algorithm": algorithm,
            "run": run_idx,
            "path": path,
        })
    return entries


def evaluate(entries: list[dict], instances: list[str], timeout: float) -> pd.DataFrame:
    """Run every saved solver against every requested new instance."""
    matrices = {}
    for name in instances:
        coords = parse_tsp(DATA_DIR / f"{name}.tsp")
        matrices[name] = (len(coords), distance_matrix(coords))

    rows = []
    total = len(entries) * len(instances)
    for i, entry in enumerate(entries, start=1):
        code = entry["path"].read_text()
        for instance in instances:
            n, matrix = matrices[instance]
            optimum = TSPLIB_OPTIMA[instance]
            result = run_generated_solver(code, matrix, timeout=timeout)
            length = int(result.length) if result.length is not None else None
            gap = (round((result.length - optimum) / optimum * 100, 2)
                   if result.length is not None else None)
            rows.append({
                "model": entry["model"],
                "strategy": entry["strategy"],
                "algorithm": entry["algorithm"],
                "run": entry["run"],
                "instance": instance,
                "cities": n,
                "status": result.status,
                "ran": result.status == "ok",
                "length": length,
                "optimum": optimum,
                "gap_pct": gap,
                "runtime_s": round(result.runtime_s, 3) if result.runtime_s is not None else None,
                "reported_length": result.reported_length,
                "refinement_rounds": None,
                "input_tokens": None,
                "output_tokens": None,
                "generation_error": None,
            })
            gap_str = f"{gap:.2f}%" if gap is not None else "-"
            done = (i - 1) * len(instances) + instances.index(instance) + 1
            print(f"[{done:>3}/{total}] {entry['model']:<6} {entry['strategy']:<22} "
                  f"{entry['algorithm']:<20} run{entry['run']} {instance:<8} "
                  f"{result.status:<13} gap={gap_str}")

    return pd.DataFrame(rows)


def carry_forward_generation_metadata(new_rows: pd.DataFrame, existing: pd.DataFrame) -> pd.DataFrame:
    """Copy per-generation metadata from the current log onto re-test rows.

    Token counts and refinement rounds describe the generation event, not the
    instance re-test. If the original log is present, keep that metadata so the
    extended rows remain comparable to the original smaller-instance matrix.
    """
    key_cols = ["model", "strategy", "algorithm", "run"]
    available = [c for c in GENERATION_METADATA if c in existing.columns]
    if not available or not all(c in existing.columns for c in key_cols):
        return new_rows

    metadata = (
        existing[key_cols + available]
        .dropna(how="all", subset=available)
        .drop_duplicates(subset=key_cols, keep="first")
    )
    if metadata.empty:
        return new_rows

    without_metadata = new_rows.drop(columns=available, errors="ignore")
    return without_metadata.merge(metadata, on=key_cols, how="left")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Re-test saved LLM-generated solvers on additional TSPLIB instances."
    )
    parser.add_argument("--instances", nargs="+", default=["pr439", "pr1002"],
                        help="Instances to evaluate (must exist in data/tsplib and TSPLIB_OPTIMA)")
    parser.add_argument("--timeout", type=float, default=300.0,
                        help="per-run wall-clock limit (s); default 300 for larger instances")
    args = parser.parse_args()

    unknown = [name for name in args.instances if name not in TSPLIB_OPTIMA]
    if unknown:
        parser.error(f"Unknown instances (not in TSPLIB_OPTIMA): {unknown}")

    entries = discover_solvers()
    print(f"Found {len(entries)} saved solvers under {GENERATED_CODE_DIR.relative_to(PROJECT_ROOT)}")
    print(f"Evaluating on instances: {args.instances} (timeout={args.timeout}s)\n")

    new_rows = evaluate(entries, args.instances, args.timeout)

    # Merge into the existing log: drop any pre-existing rows for the same
    # (model, strategy, algorithm, run, instance) and append the new ones.
    key_cols = ["model", "strategy", "algorithm", "run", "instance"]
    if LOG_PATH.exists():
        existing = pd.read_csv(LOG_PATH)
        new_rows = carry_forward_generation_metadata(new_rows, existing)
        if not existing.empty and all(c in existing.columns for c in key_cols):
            new_keys = set(zip(*(new_rows[c] for c in key_cols)))
            keep = [k not in new_keys for k in zip(*(existing[c] for c in key_cols))]
            merged = pd.concat([existing[keep], new_rows], ignore_index=True)
        else:
            merged = new_rows
    else:
        merged = new_rows
    merged.to_csv(LOG_PATH, index=False)

    ran = int(new_rows["ran"].sum())
    print(f"\nAppended {len(new_rows)} new rows ({ran} ran successfully). "
          f"Log now has {len(merged)} total rows.")


if __name__ == "__main__":
    main()
