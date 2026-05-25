"""Simulated Annealing for the Traveling Salesman Problem.

Reference: Kirkpatrick, Gelatt & Vecchi (1983), "Optimization by Simulated
Annealing", *Science* 220(4598), 671-680.

SA models the cost function as the energy of a physical system being slowly
cooled. At high temperature, uphill (worsening) moves are likely, letting
the search escape local optima; at low temperature only downhill moves
survive, converging to a near-optimum.

Design choices for this implementation:
    - Neighborhood : 2-opt (reverse a random sub-segment of the tour)
    - Cooling      : geometric, T <- T * alpha after each *temperature level*
    - Markov chain : L = chain_factor * n moves per temperature level — at
                     each T we explore the neighborhood proportionally to
                     the problem size, the textbook SA structure
    - T_initial    : auto-calibrated from the *median* uphill 2-opt delta,
                     so the typical uphill move is accepted with ~80%
                     probability while long-tail catastrophic moves are
                     suppressed
    - Starting tour: greedy nearest-neighbor (warm start)
    - Stopping     : temperature cools below ``t_final``
"""

import math
import random

import numpy as np

from src.algorithms.greedy import greedy_tour


def simulated_annealing(
    dist_matrix: np.ndarray,
    *,
    alpha: float = 0.995,
    t_final: float = 1e-3,
    chain_factor: float = 2.0,
    seed: int = 42,
) -> tuple[list[int], float]:
    """Run simulated annealing on a TSP distance matrix.

    Args:
        dist_matrix: square symmetric distance matrix of shape (n, n).
        alpha: geometric cooling factor in (0, 1); larger = slower cooling.
        t_final: stop once temperature drops below this threshold.
        chain_factor: Markov chain length per temperature level is
            ``max(1, int(chain_factor * n))``. ``1.0`` matches the
            classical "L = n" recipe.
        seed: RNG seed; seeds both ``random`` and ``numpy.random``.

    Returns:
        ``(best_tour, best_length)`` — the best tour seen across the run.
    """
    rng = random.Random(seed)
    np.random.seed(seed)

    n = dist_matrix.shape[0]
    current, current_length = greedy_tour(dist_matrix)
    best, best_length = current[:], current_length

    temperature = _calibrate_initial_temperature(current, dist_matrix, rng)
    chain_length = max(1, int(chain_factor * n))

    while temperature > t_final:
        for _ in range(chain_length):
            i, j = _random_segment(n, rng)
            delta = _two_opt_delta(current, dist_matrix, i, j, n)
            if delta < 0 or rng.random() < math.exp(-delta / temperature):
                current[i:j + 1] = current[i:j + 1][::-1]
                current_length += delta
                if current_length < best_length:
                    best = current[:]
                    best_length = current_length
        temperature *= alpha

    return best, best_length


def _random_segment(n: int, rng: random.Random) -> tuple[int, int]:
    """Pick ``1 <= i < j <= n-1`` for a 2-opt reversal.

    Restricting i >= 1 avoids the degenerate "reverse entire tour" move,
    which on a closed cycle is a topological no-op but our O(1) delta
    formula would mis-score.
    """
    i = rng.randint(1, n - 2)
    j = rng.randint(i + 1, n - 1)
    return i, j


def _two_opt_delta(
    tour: list[int],
    dist_matrix: np.ndarray,
    i: int,
    j: int,
    n: int,
) -> float:
    """O(1) change in closed-tour length from reversing ``tour[i:j+1]``.

    Reversing the segment swaps the two boundary edges
    ``(tour[i-1], tour[i])`` and ``(tour[j], tour[j+1])`` for
    ``(tour[i-1], tour[j])`` and ``(tour[i], tour[j+1])``; interior
    edges are unchanged (symmetric distance assumed).
    """
    a, b = tour[i - 1], tour[i]
    c, d = tour[j], tour[(j + 1) % n]
    return float(dist_matrix[a, c] + dist_matrix[b, d]
                 - dist_matrix[a, b] - dist_matrix[c, d])


def _calibrate_initial_temperature(
    tour: list[int],
    dist_matrix: np.ndarray,
    rng: random.Random,
    n_samples: int = 200,
    target_accept: float = 0.8,
) -> float:
    """Estimate T such that the *typical* uphill 2-opt move is accepted
    with probability ``target_accept``.

    Samples ``n_samples`` random 2-opt moves and uses the **median** of
    positive (uphill) deltas — robust to long-tail extreme moves
    (e.g., reversing nearly the entire tour, which can cost 100x more
    than a typical local perturbation). Solves
    ``exp(-median_delta / T) = target_accept`` for T.

    Falls back to T = 1.0 if no uphill moves are sampled (degenerate
    only on very small instances).
    """
    n = dist_matrix.shape[0]
    uphill: list[float] = []
    for _ in range(n_samples):
        i, j = _random_segment(n, rng)
        delta = _two_opt_delta(tour, dist_matrix, i, j, n)
        if delta > 0:
            uphill.append(delta)
    if not uphill:
        return 1.0
    median_delta = float(np.median(uphill))
    return -median_delta / math.log(target_accept)
