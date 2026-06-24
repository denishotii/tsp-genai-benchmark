"""Run all reference algorithms on all TSPLIB instances and record results.

Executes greedy, simulated annealing, and the genetic algorithm on each of
the four benchmark instances, measures solution quality (gap-to-optimal) and
wall-clock runtime, and writes the results to ``results/benchmark_results.csv``.

Run from the project root:
    python -m src.benchmark.run_benchmark
"""

import time
from pathlib import Path

import pandas as pd

from src.algorithms.genetic_algorithm import genetic_algorithm
from src.algorithms.greedy import greedy_tour
from src.algorithms.simulated_annealing import simulated_annealing
from src.utils.distance import distance_matrix
from src.utils.parser import parse_tsp
from src.utils.tour import is_valid_tour, tour_length

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "tsplib"
RESULTS_DIR = PROJECT_ROOT / "results"

# Known optimal tour lengths (TSPLIB, EUC_2D nint distances) — CLAUDE.md §11.
TSPLIB_OPTIMA = {
    "eil51": 426,
    "berlin52": 7542,
    "ch130": 6110,
    "d198": 15780,
    "pr439": 107217,
    "pr1002": 259045,
}

# Reference solvers. Each takes a distance matrix and returns (tour, length).
ALGORITHMS = {
    "greedy": greedy_tour,
    "simulated_annealing": simulated_annealing,
    "genetic_algorithm": genetic_algorithm,
}


def run_benchmark() -> pd.DataFrame:
    """Run every algorithm on every instance and return a results DataFrame."""
    rows = []
    for instance, optimum in TSPLIB_OPTIMA.items():
        coords = parse_tsp(DATA_DIR / f"{instance}.tsp")
        matrix = distance_matrix(coords)
        n = len(coords)

        for algorithm, solver in ALGORITHMS.items():
            start = time.perf_counter()
            tour, length = solver(matrix)
            elapsed = time.perf_counter() - start

            assert is_valid_tour(tour, n), f"{algorithm} on {instance}: invalid tour"
            assert length == tour_length(tour, matrix), \
                f"{algorithm} on {instance}: reported length disagrees with tour"

            gap = (length - optimum) / optimum * 100
            rows.append({
                "instance": instance,
                "cities": n,
                "algorithm": algorithm,
                "length": int(length),
                "optimum": optimum,
                "gap_pct": round(gap, 2),
                "time_s": round(elapsed, 3),
            })
            print(f"{instance:<10} {algorithm:<20} "
                  f"length={int(length):>7d}  gap={gap:>6.2f}%  time={elapsed:>6.2f}s")

    return pd.DataFrame(rows)


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    results = run_benchmark()
    out_path = RESULTS_DIR / "benchmark_results.csv"
    results.to_csv(out_path, index=False)
    print(f"\nWrote {len(results)} rows to {out_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
