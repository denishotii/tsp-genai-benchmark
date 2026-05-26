"""Isolated runner for LLM-generated TSP solvers.

Invoked as a subprocess by ``executor.py``:

    python _subprocess_runner.py <workdir>

``<workdir>`` must contain:
    - ``candidate_solution.py`` : the generated code defining ``solve``
    - ``matrix.npy``            : the (n x n) distance matrix

The runner loads the candidate in isolation, calls ``solve(matrix)``,
validates the returned tour, recomputes the canonical closed-tour length
from the matrix (never trusting the model's reported length), and writes the
outcome to ``<workdir>/result.json``. Results go to a file rather than stdout
so that stray ``print`` calls in generated code cannot corrupt them.
"""

import importlib.util
import json
import sys
import traceback
from pathlib import Path

import numpy as np


def _is_number(value: object) -> bool:
    try:
        float(value)  # type: ignore[arg-type]
        return True
    except (TypeError, ValueError):
        return False


def _load_candidate(code_path: Path):
    spec = importlib.util.spec_from_file_location("candidate_solution", code_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # raises on syntax/import/runtime errors
    return module


def run(workdir: Path) -> dict:
    matrix = np.load(workdir / "matrix.npy")
    n = matrix.shape[0]

    try:
        module = _load_candidate(workdir / "candidate_solution.py")
    except Exception:
        return {"status": "error", "error": traceback.format_exc()}

    if not hasattr(module, "solve"):
        return {"status": "no_solve", "error": "no solve() function defined"}

    try:
        output = module.solve(matrix)
    except Exception:
        return {"status": "error", "error": traceback.format_exc()}

    try:
        tour, reported_length = output
        tour = [int(city) for city in tour]
    except Exception:
        return {
            "status": "bad_output",
            "error": f"solve did not return (tour, length); got {type(output).__name__}",
        }

    if sorted(tour) != list(range(n)):
        return {
            "status": "invalid_tour",
            "error": f"tour is not a permutation of 0..{n - 1} (length {len(tour)})",
        }

    recomputed = sum(
        float(matrix[tour[k], tour[(k + 1) % n]]) for k in range(n)
    )
    return {
        "status": "ok",
        "tour": tour,
        "reported_length": float(reported_length) if _is_number(reported_length) else None,
        "recomputed_length": recomputed,
    }


def main() -> None:
    workdir = Path(sys.argv[1])
    try:
        result = run(workdir)
    except Exception:
        result = {"status": "error", "error": traceback.format_exc()}
    (workdir / "result.json").write_text(json.dumps(result))


if __name__ == "__main__":
    main()
