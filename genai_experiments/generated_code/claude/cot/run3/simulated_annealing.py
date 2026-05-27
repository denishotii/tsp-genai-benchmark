import numpy as np
import random
import math


def solve(dist_matrix):
    D = np.asarray(dist_matrix, dtype=float)
    n = D.shape[0]

    if n == 0:
        return [], 0.0
    if n == 1:
        return [0], 0.0
    if n == 2:
        return [0, 1], float(D[0, 1] + D[1, 0])

    def tour_length(t):
        idx = np.asarray(t)
        return float(D[idx, np.roll(idx, -1)].sum())

    # Nearest neighbor initial tour
    def nearest_neighbor(start=0):
        visited = [False] * n
        t = [start]
        visited[start] = True
        cur = start
        for _ in range(n - 1):
            best_d = float('inf')
            best_j = -1
            row = D[cur]
            for j in range(n):
                if not visited[j] and row[j] < best_d:
                    best_d = row[j]
                    best_j = j
            t.append(best_j)
            visited[best_j] = True
            cur = best_j
        return t

    # Try a few starts and take the best
    best_init = None
    best_init_len = float('inf')
    for s in range(min(n, 5)):
        t = nearest_neighbor(s)
        L = tour_length(t)
        if L < best_init_len:
            best_init_len = L
            best_init = t

    tour = list(best_init)
    cur_len = best_init_len
    best_tour = list(tour)
    best_len = cur_len

    # Estimate initial temperature from average edge length
    # Sample some edges to set T0 so that small upward moves are often accepted
    sample_size = min(1000, n * n)
    flat = D[np.triu_indices(n, k=1)]
    if flat.size > 0:
        avg_edge = float(flat.mean())
    else:
        avg_edge = 1.0

    T0 = max(avg_edge * 0.5, 1e-6)
    T_end = max(avg_edge * 1e-4, 1e-9)

    # Iteration budget
    iters = max(20000, 200 * n * n)
    iters = min(iters, 500000)

    # geometric cooling: T = T0 * alpha^k, alpha = (T_end/T0)^(1/iters)
    alpha = (T_end / T0) ** (1.0 / iters)

    T = T0
    rng = random.Random(12345)

    for k in range(iters):
        # pick two indices i < j, with 1 <= i, j <= n-1 ideally avoid full reversal
        i = rng.randint(1, n - 2)
        j = rng.randint(i + 1, n - 1)
        # treat tour as circular but with fixed start; reversing [i..j]
        a = tour[i - 1]
        b = tour[i]
        c = tour[j]
        d = tour[(j + 1) % n]

        if a == c or b == d:
            T *= alpha
            continue

        before = D[a, b] + D[c, d]
        after = D[a, c] + D[b, d]
        delta = after - before

        if delta < 0 or rng.random() < math.exp(-delta / T):
            tour[i:j + 1] = tour[i:j + 1][::-1]
            cur_len += delta
            if cur_len < best_len:
                best_len = cur_len
                best_tour = list(tour)

        T *= alpha

    # Final 2-opt local search polish
    improved = True
    while improved:
        improved = False
        for i in range(1, n - 1):
            a = best_tour[i - 1]
            b = best_tour[i]
            Dab = D[a, b]
            for j in range(i + 1, n):
                c = best_tour[j]
                d = best_tour[(j + 1) % n]
                if a == c or b == d:
                    continue
                delta = (D[a, c] + D[b, d]) - (Dab + D[c, d])
                if delta < -1e-12:
                    best_tour[i:j + 1] = best_tour[i:j + 1][::-1]
                    best_len += delta
                    improved = True
                    break
            if improved:
                break

    # Recompute length for numerical safety
    best_len = tour_length(best_tour)
    return best_tour, float(best_len)