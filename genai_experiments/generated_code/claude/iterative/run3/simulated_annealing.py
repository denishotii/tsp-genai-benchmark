import numpy as np
import random
import math


def tour_length(tour, dist_matrix):
    idx = np.asarray(tour)
    return float(dist_matrix[idx, np.roll(idx, -1)].sum())


def nearest_neighbor_tour(dist_matrix, start=0):
    n = dist_matrix.shape[0]
    visited = [False] * n
    tour = [start]
    visited[start] = True
    current = start
    for _ in range(n - 1):
        dists = dist_matrix[current].copy()
        dists[visited] = np.inf
        nxt = int(np.argmin(dists))
        tour.append(nxt)
        visited[nxt] = True
        current = nxt
    return tour


def solve(dist_matrix):
    dist_matrix = np.asarray(dist_matrix, dtype=float)
    n = dist_matrix.shape[0]

    if n == 0:
        return [], 0.0
    if n == 1:
        return [0], 0.0
    if n == 2:
        return [0, 1], float(dist_matrix[0, 1] + dist_matrix[1, 0])

    # Initial tour from nearest neighbor
    current = nearest_neighbor_tour(dist_matrix, start=0)
    current_len = tour_length(current, dist_matrix)

    best = current[:]
    best_len = current_len

    # Initial temperature: based on average edge cost
    # Sample some random differences to set T0
    sample_diffs = []
    for _ in range(min(100, n * n)):
        i, j = random.sample(range(n), 2)
        sample_diffs.append(abs(dist_matrix[i, j]))
    if sample_diffs:
        T0 = max(np.mean(sample_diffs), 1e-9)
    else:
        T0 = 1.0

    T = T0
    T_min = T0 * 1e-6
    alpha = 0.9995

    # Number of iterations scales with problem size
    max_iter = max(10000, 200 * n * n)
    max_iter = min(max_iter, 500000)

    iters_no_improve = 0
    restart_limit = max_iter // 4

    for it in range(max_iter):
        if T < T_min:
            T = T0 * 0.5  # reheat
            T0 = T

        # 2-opt move: pick two indices i < j, reverse segment current[i..j]
        i = random.randint(0, n - 2)
        j = random.randint(i + 1, n - 1)
        if i == 0 and j == n - 1:
            continue  # full reverse = same tour

        # Compute delta efficiently
        a = current[i - 1] if i > 0 else current[n - 1]
        b = current[i]
        c = current[j]
        d = current[(j + 1) % n]

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
                    if random.random() < math.exp(-delta / T):
                        accept = True
                except OverflowError:
                    accept = False

        if accept:
            current[i:j + 1] = current[i:j + 1][::-1]
            current_len += delta
            if current_len < best_len - 1e-12:
                best = current[:]
                best_len = current_len
                iters_no_improve = 0
            else:
                iters_no_improve += 1
        else:
            iters_no_improve += 1

        T *= alpha

        if iters_no_improve > restart_limit:
            current = best[:]
            current_len = best_len
            T = T0
            iters_no_improve = 0

    # Final cleanup: greedy 2-opt
    improved = True
    while improved:
        improved = False
        for i in range(n - 1):
            for j in range(i + 1, n):
                if i == 0 and j == n - 1:
                    continue
                a = best[i - 1] if i > 0 else best[n - 1]
                b = best[i]
                c = best[j]
                d = best[(j + 1) % n]
                if a == c or b == d:
                    continue
                delta = (dist_matrix[a, c] + dist_matrix[b, d]
                         - dist_matrix[a, b] - dist_matrix[c, d])
                if delta < -1e-12:
                    best[i:j + 1] = best[i:j + 1][::-1]
                    best_len += delta
                    improved = True

    best_len = tour_length(best, dist_matrix)
    return best, float(best_len)