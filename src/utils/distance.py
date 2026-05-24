"""Euclidean distance matrix for TSPLIB EUC_2D instances."""

import numpy as np


def distance_matrix(coords: list[tuple[float, float]]) -> np.ndarray:
    """Return the symmetric distance matrix for EUC_2D coordinates.

    Distances follow the TSPLIB95 EUC_2D convention: nearest-integer
    rounding with half-up tie-breaking (``floor(sqrt(dx^2 + dy^2) + 0.5)``).
    This matches the rounding used to compute the published optimal tour
    lengths in TSPLIB, so gap-to-optimal comparisons are exact.

    Returned dtype is float for downstream numpy compatibility, but all
    entries are whole numbers.
    """
    pts = np.asarray(coords, dtype=float)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    raw = np.sqrt(np.sum(diff ** 2, axis=-1))
    return np.floor(raw + 0.5)
