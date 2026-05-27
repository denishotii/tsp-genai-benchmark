import numpy as np
import random
import math


def tour_length(tour, dist_matrix):
    return float(dist_matrix[tour, np.roll(tour, -1)].sum())


def solve(dist_matrix):
    n = len(dist_matrix)
    if n <= 1:
        return list(range(n)), 0.0
    if n == 2:
        return [0, 1], float(dist_matrix[0, 1] + dist_matrix[1, 0])

    # Initial tour via nearest neighbor heuristic
    def nearest_neighbor(start):
        unvisited = set(range(n))
        unvisited.remove(start)
        tour = [start]
        cur = start
        while unvisited:
            nxt = min(unvisited, key=lambda j: dist_matrix[cur, j])
            tour.append(nxt)
            unvisited.remove(nxt)
            cur = nxt
        return tour

    best_tour = nearest_neighbor(0)
    best_len = tour_length(best_tour, dist_matrix)

    current = list(best_tour)
    current_len = best_len

    # Estimate initial temperature from average edge length
    avg_edge = float(dist_matrix[dist_matrix > 0].mean()) if np.any(dist_matrix > 0) else 1.0
    T0 = avg_edge
    T_min = avg_edge * 1e-4
    alpha = 0.9995

    # Iteration budget scales with n
    max_iter = max(20000, 200 * n * n)
    max_iter = min(max_iter, 500000)

    T = T0
    rng = random.Random(42)

    for it in range(max_iter):
        if T < T_min:
            break

        # 2-opt move: pick i < j, reverse segment current[i:j+1]
        i = rng.randint(0, n - 1)
        j = rng.randint(0, n - 1)
        if i == j:
            T *= alpha
            continue
        if i > j:
            i, j = j, i
        if i == 0 and j == n - 1:
            T *= alpha
            continue

        # Delta computation for 2-opt on a cyclic tour
        a = current[i - 1]
        b = current[i]
        c = current[j]
        d = current[(j + 1) % n]

        d_old = dist_matrix[a, b] + dist_matrix[c, d]
        d_new = dist_matrix[a, c] + dist_matrix[b, d]
        delta = d_new - d_old

        if delta < 0 or rng.random() < math.exp(-delta / T):
            current[i:j + 1] = current[i:j + 1][::-1]
            current_len += delta
            if current_len < best_len:
                best_len = current_len
                best_tour = list(current)

        T *= alpha

    # Final cleanup: full 2-opt local search
    improved = True
    tour = list(best_tour)
    while improved:
        improved = False
        for i in range(1, n - 1):
            a = tour[i - 1]
            b = tour[i]
            for j in range(i + 1, n):
                c = tour[j]
                d = tour[(j + 1) % n]
                if a == d:
                    continue
                d_old = dist_matrix[a, b] + dist_matrix[c, d]
                d_new = dist_matrix[a, c] + dist_matrix[b, d]
                if d_new + 1e-12 < d_old:
                    tour[i:j + 1] = tour[i:j + 1][::-1]
                    improved = True
                    b = tour[i]
        # one pass; loop continues if improved

    final_len = tour_length(tour, dist_matrix)
    if final_len < best_len:
        best_len = final_len
        best_tour = tour

    return list(best_tour), float(best_len)