import numpy as np
import random
import math


def tour_length(tour, dist_matrix):
    idx = np.asarray(tour)
    return float(dist_matrix[idx, np.roll(idx, -1)].sum())


def solve(dist_matrix):
    n = dist_matrix.shape[0]
    if n <= 1:
        return list(range(n)), 0.0
    if n == 2:
        return [0, 1], float(dist_matrix[0, 1] + dist_matrix[1, 0])

    # Nearest neighbor initial tour
    def nearest_neighbor(start):
        unvisited = set(range(n))
        unvisited.remove(start)
        tour = [start]
        cur = start
        while unvisited:
            nxt = min(unvisited, key=lambda x: dist_matrix[cur, x])
            tour.append(nxt)
            unvisited.remove(nxt)
            cur = nxt
        return tour

    best_tour = nearest_neighbor(0)
    best_len = tour_length(best_tour, dist_matrix)

    # Estimate initial temperature from average edge cost
    sample = dist_matrix[dist_matrix > 0]
    avg_edge = float(sample.mean()) if sample.size > 0 else 1.0
    T0 = avg_edge
    T_min = max(1e-6, avg_edge * 1e-6)
    alpha = 0.9995

    # Iterations scale with problem size
    max_iter = max(10000, 200 * n * n)
    max_iter = min(max_iter, 500000)

    current = list(best_tour)
    current_len = best_len
    T = T0

    rng = random.Random(42)

    no_improve = 0
    reheat_limit = max(2000, 20 * n)

    for it in range(max_iter):
        # Pick two indices for 2-opt segment reversal
        i = rng.randint(0, n - 1)
        j = rng.randint(0, n - 1)
        if i == j:
            continue
        if i > j:
            i, j = j, i
        # Avoid trivial full-rotation
        if i == 0 and j == n - 1:
            continue

        # Compute delta efficiently
        a = current[i - 1] if i > 0 else current[n - 1]
        b = current[i]
        c = current[j]
        d = current[j + 1] if j + 1 < n else current[0]

        if a == c or b == d:
            delta = 0.0
        else:
            delta = (dist_matrix[a, c] + dist_matrix[b, d]
                     - dist_matrix[a, b] - dist_matrix[c, d])

        accept = False
        if delta < 0:
            accept = True
        else:
            if T > 1e-12:
                try:
                    if rng.random() < math.exp(-delta / T):
                        accept = True
                except OverflowError:
                    accept = False

        if accept:
            current[i:j + 1] = current[i:j + 1][::-1]
            current_len += delta
            if current_len < best_len - 1e-12:
                best_len = current_len
                best_tour = list(current)
                no_improve = 0
            else:
                no_improve += 1
        else:
            no_improve += 1

        T *= alpha
        if T < T_min:
            T = T_min

        if no_improve > reheat_limit:
            T = max(T, T0 * 0.3)
            no_improve = 0

    # Final exact recompute
    best_len = tour_length(best_tour, dist_matrix)
    return list(best_tour), float(best_len)