"""Unit tests for the experiment orchestrator (no API calls, no file writes)."""

import numpy as np

from genai_experiments.code_extraction import extract_code
from genai_experiments.executor import ExecutionResult, run_generated_solver
from genai_experiments.run_experiments import _DRY_RUN_RESPONSE, _format_feedback

_SQUARE = np.array(
    [[0, 10, 14, 10], [10, 0, 10, 14], [14, 10, 0, 10], [10, 14, 10, 0]],
    dtype=float,
)


def test_dry_run_stub_extracts_to_a_runnable_solver():
    # The canned --dry-run response must survive extraction and execute, so the
    # offline pipeline exercises the real executor path.
    code = extract_code(_DRY_RUN_RESPONSE)
    result = run_generated_solver(code, _SQUARE)
    assert result.ok
    assert result.length == 40.0


def test_format_feedback_describes_each_failure_mode():
    assert "time limit" in _format_feedback(ExecutionResult("timeout", error="exceeded 60s"))
    assert "invalid tour" in _format_feedback(ExecutionResult("invalid_tour", error="dup"))
    assert "solve" in _format_feedback(ExecutionResult("no_solve"))
    assert "wrong type" in _format_feedback(ExecutionResult("bad_output", error="int"))
    assert "error" in _format_feedback(ExecutionResult("error", error="boom"))
