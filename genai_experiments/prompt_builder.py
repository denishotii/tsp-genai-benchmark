"""Build LLM prompts for each (strategy, algorithm) combination.

Strategy templates live in ``genai_experiments/prompts/*.txt`` and contain
``{task}``, ``{interface}``, and (few-shot only) ``{example}`` placeholders.
This module injects the algorithm-specific task description and the shared
entry-point contract that every generated solver must satisfy.

The task descriptions are deliberately spec-level (matching CLAUDE.md §4) and
reveal no implementation details — the point of the experiment is to test the
model's own design, not its ability to copy a reference.
"""

from pathlib import Path

PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"

STRATEGIES = ("zero_shot", "few_shot", "chain_of_thought", "iterative_refinement")
ALGORITHMS = ("greedy", "simulated_annealing", "genetic_algorithm")

TASKS = {
    "greedy": (
        "Implement the nearest-neighbor greedy heuristic for the Traveling "
        "Salesman Problem (TSP). Starting from a fixed city, repeatedly travel "
        "to the nearest not-yet-visited city until all cities have been "
        "visited, then return to the start."
    ),
    "simulated_annealing": (
        "Implement a Simulated Annealing solver for the Traveling Salesman "
        "Problem (TSP). Use a 2-opt neighborhood (reverse a segment of the "
        "tour) and a geometric cooling schedule."
    ),
    "genetic_algorithm": (
        "Implement a Genetic Algorithm for the Traveling Salesman Problem "
        "(TSP). Use a permutation encoding, tournament selection, Order "
        "Crossover (OX), swap mutation, and elitism."
    ),
}

INTERFACE = """\
Your solution must define exactly this entry point:

    def solve(dist_matrix):
        ...
        return tour, length

where:
  - dist_matrix is an (n x n) symmetric NumPy array of pairwise distances
    (n = number of cities); dist_matrix[i][j] is the distance from city i to
    city j, and the diagonal is zero.
  - tour is a list containing each city index 0..n-1 exactly once, giving the
    visiting order of a single closed tour.
  - length is that closed tour's total length as a float: the sum of distances
    between consecutive cities plus the edge from the last city back to the
    first.

Provide your solution as a single Python code block. Use only NumPy and the
Python standard library (no external optimization libraries)."""

FEW_SHOT_EXAMPLE = """\
import numpy as np

def minimize(f, x0, step=0.1, iterations=1000):
    \"\"\"Minimize a 1-D function f by simple hill climbing from x0.\"\"\"
    x, fx = x0, f(x0)
    for _ in range(iterations):
        for candidate in (x - step, x + step):
            fc = f(candidate)
            if fc < fx:
                x, fx = candidate, fc
    return x, fx"""

FEEDBACK_TEMPLATE = """\
When we ran your previous implementation, we observed:

{feedback}

Please correct the problem and return the complete, fixed implementation.

{interface}"""


def build_prompt(strategy: str, algorithm: str) -> str:
    """Render the initial prompt for a (strategy, algorithm) pair."""
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown strategy: {strategy!r}")
    if algorithm not in ALGORITHMS:
        raise ValueError(f"unknown algorithm: {algorithm!r}")
    template = (PROMPTS_DIR / f"{strategy}.txt").read_text()
    return (
        template
        .replace("{task}", TASKS[algorithm])
        .replace("{example}", FEW_SHOT_EXAMPLE)
        .replace("{interface}", INTERFACE)
    )


def build_feedback_prompt(feedback: str) -> str:
    """Render a follow-up prompt for the iterative-refinement strategy."""
    return (
        FEEDBACK_TEMPLATE
        .replace("{feedback}", feedback)
        .replace("{interface}", INTERFACE)
    )
