"""Tests for the GenAI code executor and code extraction."""

import numpy as np

from genai_experiments.code_extraction import extract_code
from genai_experiments.executor import run_generated_solver

# A small, fixed 4-city square: edges along the side = 10, diagonals = 14.
SQUARE = np.array(
    [
        [0, 10, 14, 10],
        [10, 0, 10, 14],
        [14, 10, 0, 10],
        [10, 14, 10, 0],
    ],
    dtype=float,
)


# --- code extraction ---

def test_extract_code_from_fenced_block():
    response = "Sure:\n\n```python\ndef solve(d):\n    return [0], 0.0\n```\nDone."
    assert extract_code(response) == "def solve(d):\n    return [0], 0.0"


def test_extract_code_returns_last_block_for_cot():
    # CoT puts a reasoning snippet first and the real solution last.
    response = (
        "First a sketch:\n```python\n# pseudocode\n```\n"
        "Final version:\n```python\ndef solve(d):\n    return [0], 0.0\n```"
    )
    assert extract_code(response) == "def solve(d):\n    return [0], 0.0"


def test_extract_code_without_fence_returns_stripped_text():
    assert extract_code("  def solve(d):\n    return [0], 0.0  ") == \
        "def solve(d):\n    return [0], 0.0"


# --- executor: happy path ---

def test_executor_runs_valid_solver():
    code = (
        "def solve(dist_matrix):\n"
        "    n = dist_matrix.shape[0]\n"
        "    tour = list(range(n))\n"
        "    length = sum(dist_matrix[tour[k]][tour[(k+1) % n]] for k in range(n))\n"
        "    return tour, length\n"
    )
    result = run_generated_solver(code, SQUARE)
    assert result.ok
    assert result.tour == [0, 1, 2, 3]
    assert result.length == 40.0          # perimeter of the square
    assert result.reported_length == 40.0


def test_executor_recomputes_length_and_ignores_model_claim():
    # The model returns a valid tour but lies about its length; we must
    # report the true recomputed length, not the claim.
    code = (
        "def solve(dist_matrix):\n"
        "    return [0, 1, 2, 3], 0.0\n"
    )
    result = run_generated_solver(code, SQUARE)
    assert result.ok
    assert result.length == 40.0
    assert result.reported_length == 0.0


# --- executor: failure modes ---

def test_executor_detects_syntax_error():
    result = run_generated_solver("def solve(dist_matrix)\n    return [], 0\n", SQUARE)
    assert result.status == "error"


def test_executor_detects_missing_solve():
    result = run_generated_solver("def helper():\n    pass\n", SQUARE)
    assert result.status == "no_solve"


def test_executor_detects_bad_output():
    result = run_generated_solver("def solve(dist_matrix):\n    return 42\n", SQUARE)
    assert result.status == "bad_output"


def test_executor_detects_invalid_tour():
    code = (
        "def solve(dist_matrix):\n"
        "    n = dist_matrix.shape[0]\n"
        "    return [0] * n, 0.0\n"  # duplicates, not a permutation
    )
    result = run_generated_solver(code, SQUARE)
    assert result.status == "invalid_tour"


def test_executor_detects_runtime_error():
    result = run_generated_solver("def solve(dist_matrix):\n    raise ValueError('x')\n", SQUARE)
    assert result.status == "error"


def test_executor_times_out_on_infinite_loop():
    result = run_generated_solver(
        "def solve(dist_matrix):\n    while True:\n        pass\n",
        SQUARE,
        timeout=2.0,
    )
    assert result.status == "timeout"
