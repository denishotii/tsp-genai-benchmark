"""Tour validation and length computation."""

import numpy as np


def tour_length(tour: list[int], dist_matrix: np.ndarray) -> float:
    """Length of the closed tour, including the edge back to the start city."""
    arr = np.asarray(tour)
    return float(dist_matrix[arr, np.roll(arr, -1)].sum())


def is_valid_tour(tour: list[int], n: int) -> bool:
    """True iff ``tour`` is a permutation of ``{0, 1, ..., n-1}``."""
    if len(tour) != n:
        return False
    return set(tour) == set(range(n))
