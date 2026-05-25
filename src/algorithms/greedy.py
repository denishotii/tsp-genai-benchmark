"""Nearest-neighbor greedy heuristic for the Traveling Salesman Problem.

Classical constructive heuristic: starting from a fixed city, repeatedly
move to the closest unvisited city until every city is visited, then close
the tour by returning to the start.

Fast (O(n^2)) and simple, but suboptimal — typical EUC_2D solution
quality is 20-25% above the true optimum. Used in this project as the
quality floor: any serious solver (SA, GA, LLM-generated code) must beat
greedy to be worth comparing.
"""

import numpy as np

from src.utils.tour import tour_length


def greedy_tour(dist_matrix: np.ndarray, start: int = 0) -> tuple[list[int], float]:
    """Build a TSP tour by always moving to the nearest unvisited city.

    Args:
        dist_matrix: square symmetric distance matrix of shape (n, n).
        start: index of the starting city (default 0).

    Returns:
        ``(tour, length)`` where ``tour`` is a permutation of
        ``range(n)`` beginning with ``start`` and ``length`` is the
        closed-tour length under ``dist_matrix``.
    """
    n = dist_matrix.shape[0]
    visited = np.zeros(n, dtype=bool)
    tour = [start]
    visited[start] = True
    current = start

    for _ in range(n - 1):
        candidates = dist_matrix[current].copy()
        candidates[visited] = np.inf
        nxt = int(np.argmin(candidates))
        tour.append(nxt)
        visited[nxt] = True
        current = nxt

    return tour, tour_length(tour, dist_matrix)
