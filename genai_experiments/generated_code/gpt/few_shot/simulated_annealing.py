import math
import numpy as np


def _tour_length(tour, dist):
    """Return the total closed-tour length."""
    if len(tour) == 0:
        return 0.0
    return float(np.sum(dist[tour, np.roll(tour, -1)]))


def _nearest_neighbor_tour(dist):
    """Construct a simple deterministic nearest-neighbor initial tour."""
    n = dist.shape[0]
    tour = np.empty(n, dtype=int)
    unvisited = np.ones(n, dtype=bool)

    current = 0
    tour[0] = current
    unvisited[current] = False

    for k in range(1, n):
        candidates = np.flatnonzero(unvisited)
        current = int(candidates[np.argmin(dist[current, candidates])])
        tour[k] = current
        unvisited[current] = False

    return tour


def _two_opt_delta(tour, dist, i, j):
    """
    Change in tour length from reversing tour[i:j+1].

    Removes edges:
        tour[i-1] -> tour[i]
        tour[j]   -> tour[j+1]
    Adds edges:
        tour[i-1] -> tour[j]
        tour[i]   -> tour[j+1]
    """
    n = len(tour)
    a = tour[i - 1]
    b = tour[i]
    c = tour[j]
    d = tour[(j + 1) % n]

    return float(dist[a, c] + dist[b, d] - dist[a, b] - dist[c, d])


def _two_opt_local_search(tour, dist, length, max_passes=30):
    """Small first-improvement 2-opt polish."""
    n = len(tour)

    for _ in range(max_passes):
        improved = False

        for i in range(n - 1):
            for j in range(i + 1, n):
                if i == 0 and j == n - 1:
                    continue

                delta = _two_opt_delta(tour, dist, i, j)

                if delta < -1e-12:
                    tour[i:j + 1] = tour[i:j + 1][::-1].copy()
                    length += delta
                    improved = True
                    break

            if improved:
                break

        if not improved:
            break

    return tour, float(length)


def solve(dist_matrix):
    dist = np.asarray(dist_matrix, dtype=float)
    n = dist.shape[0]

    if n == 0:
        return [], 0.0
    if n == 1:
        return [0], 0.0
    if n == 2:
        return [0, 1], float(dist[0, 1] + dist[1, 0])

    rng = np.random.default_rng(12345)

    tour = _nearest_neighbor_tour(dist)
    current_length = _tour_length(tour, dist)

    best_tour = tour.copy()
    best_length = current_length

    # Estimate an initial temperature from sampled uphill 2-opt moves.
    positive_deltas = []
    samples = min(2000, max(100, 20 * n))

    for _ in range(samples):
        i = int(rng.integers(0, n))
        j = int(rng.integers(0, n))

        if i > j:
            i, j = j, i
        if i == j or (i == 0 and j == n - 1):
            continue

        delta = _two_opt_delta(tour, dist, i, j)
        if delta > 1e-12:
            positive_deltas.append(delta)

    if positive_deltas:
        # Choose T so that an average uphill move is accepted with probability ~0.8.
        temperature = -float(np.mean(positive_deltas)) / math.log(0.8)
    else:
        temperature = max(current_length / n, 1e-12)

    temperature = max(temperature, 1e-12)
    final_temperature = temperature * 1e-4

    max_moves = int(max(5000, min(300000, 2000 * n)))
    moves_per_temperature = max(10, min(100, n))
    cooling_steps = max(1, max_moves // moves_per_temperature)

    # Geometric cooling factor.
    alpha = (final_temperature / temperature) ** (1.0 / cooling_steps)

    moves_done = 0

    while moves_done < max_moves and temperature > final_temperature:
        for _ in range(moves_per_temperature):
            if moves_done >= max_moves:
                break

            i = int(rng.integers(0, n))
            j = int(rng.integers(0, n))

            if i > j:
                i, j = j, i

            if i == j or (i == 0 and j == n - 1):
                continue

            delta = _two_opt_delta(tour, dist, i, j)

            if delta <= 0.0 or rng.random() < math.exp(-delta / temperature):
                tour[i:j + 1] = tour[i:j + 1][::-1].copy()
                current_length += delta

                if current_length < best_length:
                    best_length = current_length
                    best_tour = tour.copy()

            moves_done += 1

        temperature *= alpha

    # Optional deterministic 2-opt polish for modest instance sizes.
    if n <= 200:
        best_tour, best_length = _two_opt_local_search(
            best_tour.copy(),
            dist,
            best_length,
        )

    best_length = _tour_length(best_tour, dist)

    return best_tour.tolist(), float(best_length)