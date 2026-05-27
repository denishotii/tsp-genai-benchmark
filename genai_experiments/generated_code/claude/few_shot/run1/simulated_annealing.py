import numpy as np
import random
import math


def _tour_length(tour, dist_matrix):
    return float(dist_matrix[tour, np.roll(tour, -1)].sum())


def _nearest_neighbor_tour(dist_matrix):
    n = dist_matrix.shape[0]
    start = 0
    unvisited = set(range(n))
    unvisited.remove(start)
    tour = [start]
    current = start
    while unvisited:
        nxt = min(unvisited, key=lambda j: dist_matrix[current, j])
        tour.append(nxt)
        unvisited.remove(nxt)
        current = nxt
    return tour


def solve(dist_matrix):
    dist_matrix = np.asarray(dist_matrix, dtype=float)
    n = dist_matrix.shape[0]

    if n <= 1:
        return list(range(n)), 0.0
    if n == 2:
        return [0, 1], float(dist_matrix[0, 1] + dist_matrix[1, 0])

    # Initial tour via nearest neighbor heuristic
    tour = np.array(_nearest_neighbor_tour(dist_matrix), dtype=int)
    current_length = _tour_length(tour, dist_matrix)

    best_tour = tour.copy()
    best_length = current_length

    # Estimate initial temperature from average edge lengths
    iu = np.triu_indices(n, k=1)
    mean_edge = float(dist_matrix[iu].mean()) if len(iu[0]) > 0 else 1.0
    T0 = mean_edge
    T_min = T0 * 1e-6 if T0 > 0 else 1e-9
    alpha = 0.9995

    # Number of iterations scales with problem size
    iterations = max(20000, 200 * n * n)
    iterations = min(iterations, 300000)

    T = T0
    rng = random.Random(0xC0FFEE)

    for _ in range(iterations):
        # Pick two distinct indices i < j defining a segment to reverse
        i = rng.randint(0, n - 1)
        j = rng.randint(0, n - 1)
        if i == j:
            continue
        if i > j:
            i, j = j, i
        # Skip trivial reversals (full tour reversal yields same closed tour)
        if i == 0 and j == n - 1:
            continue

        a = tour[i - 1] if i > 0 else tour[n - 1]
        b = tour[i]
        c = tour[j]
        d = tour[j + 1] if j < n - 1 else tour[0]

        # Change in length when reversing segment tour[i..j]
        old = dist_matrix[a, b] + dist_matrix[c, d]
        new = dist_matrix[a, c] + dist_matrix[b, d]
        delta = new - old

        if delta < 0 or (T > 0 and rng.random() < math.exp(-delta / T)):
            tour[i:j + 1] = tour[i:j + 1][::-1]
            current_length += delta
            if current_length < best_length:
                best_length = current_length
                best_tour = tour.copy()

        T *= alpha
        if T < T_min:
            T = T_min

    # Final cleanup: deterministic 2-opt improvement pass
    improved = True
    while improved:
        improved = False
        for i in range(n - 1):
            a = best_tour[i - 1] if i > 0 else best_tour[n - 1]
            b = best_tour[i]
            for j in range(i + 1, n):
                if i == 0 and j == n - 1:
                    continue
                c = best_tour[j]
                d = best_tour[j + 1] if j < n - 1 else best_tour[0]
                delta = (dist_matrix[a, c] + dist_matrix[b, d]
                         - dist_matrix[a, b] - dist_matrix[c, d])
                if delta < -1e-12:
                    best_tour[i:j + 1] = best_tour[i:j + 1][::-1]
                    best_length += delta
                    improved = True
                    b = best_tour[i]
        # Recompute to avoid drift
        best_length = _tour_length(best_tour, dist_matrix)

    return best_tour.tolist(), float(best_length)