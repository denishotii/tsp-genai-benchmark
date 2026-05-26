"""Safely run and benchmark LLM-generated TSP solvers.

Generated code is untrusted: it may not parse, may crash, may loop forever,
or may return an invalid tour. Each candidate is therefore executed in an
isolated subprocess (see ``_subprocess_runner.py``) with a wall-clock
timeout, and its result is validated and re-scored against our own
nint-rounded distance matrix.

Note: subprocess isolation + timeout contains hangs and crashes; it is NOT a
security sandbox. Only run code from trusted model sources on a machine where
that is acceptable.
"""

import json
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_RUNNER = Path(__file__).resolve().parent / "_subprocess_runner.py"

# status values:
#   ok           - ran and returned a valid tour
#   error        - raised an exception (syntax, import, or runtime)
#   timeout      - exceeded the wall-clock limit
#   no_solve     - no solve() entry point defined
#   bad_output   - solve() did not return (tour, length)
#   invalid_tour - returned tour is not a permutation of 0..n-1
#   crashed      - subprocess died without writing a result


@dataclass
class ExecutionResult:
    """Outcome of running one generated solver on one distance matrix."""

    status: str
    tour: list[int] | None = None
    length: float | None = None           # canonical length recomputed by us
    reported_length: float | None = None  # length the model claimed (may differ)
    error: str | None = None
    runtime_s: float | None = None

    @property
    def ok(self) -> bool:
        return self.status == "ok"


def run_generated_solver(
    code: str,
    dist_matrix: np.ndarray,
    timeout: float = 60.0,
) -> ExecutionResult:
    """Run generated ``code`` (which must define ``solve``) on ``dist_matrix``.

    Returns an :class:`ExecutionResult`; ``length`` is the canonical
    closed-tour length recomputed from ``dist_matrix`` (independent of what
    the model reported).
    """
    with tempfile.TemporaryDirectory() as tmp:
        workdir = Path(tmp)
        (workdir / "candidate_solution.py").write_text(code)
        np.save(workdir / "matrix.npy", dist_matrix)

        start = time.perf_counter()
        try:
            completed = subprocess.run(
                [sys.executable, str(_RUNNER), str(workdir)],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=workdir,
            )
        except subprocess.TimeoutExpired:
            return ExecutionResult(
                status="timeout",
                runtime_s=timeout,
                error=f"exceeded {timeout:.0f}s wall-clock limit",
            )
        runtime = time.perf_counter() - start

        result_path = workdir / "result.json"
        if not result_path.exists():
            stderr = (completed.stderr or "").strip()
            return ExecutionResult(
                status="crashed",
                runtime_s=runtime,
                error=stderr or "subprocess produced no result",
            )

        data = json.loads(result_path.read_text())

    return ExecutionResult(
        status=data["status"],
        tour=data.get("tour"),
        length=data.get("recomputed_length"),
        reported_length=data.get("reported_length"),
        error=data.get("error"),
        runtime_s=runtime,
    )
