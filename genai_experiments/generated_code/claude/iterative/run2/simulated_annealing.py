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

    # Nearest neighbor initial tour
    start = 0
    unvisited = set(range(n))
    unvisited.remove(start)
    tour = [start]
    cur = start
    while unvisited:
        nxt = min(unvisited, key=lambda x: dist_matrix[cur, x])
        tour.append(nxt)
        unvisited.remove(nxt)
        cur = nxt
    tour = np.array(tour, dtype=np.int64)
    cur_len = tour_length(tour, dist_matrix)

    best_tour = tour.copy()
    best_len = cur_len

    # Estimate initial temperature from average edge
    avg_edge = dist_matrix[dist_matrix > 0].mean() if np.any(dist_matrix > 0) else 1.0
    T = avg_edge
    T_min = avg_edge * 1e-4
    alpha = 0.9995

    # Iterations scale with n
    max_iter = max(10000, 200 * n * n)
    max_iter = min(max_iter, 300000)

    rng = random.Random(42)

    iters = 0
    while T > T_min and iters < max_iter:
        # 2-opt move: pick i<j, reverse tour[i:j+1]
        i = rng.randint(0, n - 1)
        j = rng.randint(0, n - 1)
        if i == j:
            iters += 1
            T *= alpha
            continue
        if i > j:
            i, j = j, i
        # Skip degenerate full-reverse
        if i == 0 and j == n - 1:
            iters += 1
            T *= alpha
            continue

        a = tour[i - 1] if i > 0 else tour[n - 1]
        b = tour[i]
        c = tour[j]
        d = tour[(j + 1) % n]

        # Change in length: remove edges (a,b) and (c,d), add (a,c) and (b,d)
        delta = (dist_matrix[a, c] + dist_matrix[b, d]
                 - dist_matrix[a, b] - dist_matrix[c, d])

        if delta < 0 or rng.random() < math.exp(-delta / T):
            tour[i:j + 1] = tour[i:j + 1][::-1]
            cur_len += delta
            if cur_len < best_len:
                best_len = cur_len
                best_tour = tour.copy()

        T *= alpha
        iters += 1

    # Final cleanup: recompute length to avoid float drift
    best_len = tour_length(best_tour, dist_matrix)
    return best_tour.tolist(), float(best_len)