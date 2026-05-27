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
    current = start
    while unvisited:
        nxt = min(unvisited, key=lambda x: dist_matrix[current, x])
        tour.append(nxt)
        unvisited.remove(nxt)
        current = nxt
    tour = np.array(tour, dtype=np.int64)

    current_length = tour_length(tour, dist_matrix)
    best_tour = tour.copy()
    best_length = current_length

    # Initial temperature: based on average edge cost
    avg_edge = float(dist_matrix.sum()) / (n * (n - 1)) if n > 1 else 1.0
    T0 = avg_edge
    T_min = avg_edge * 1e-6 if avg_edge > 0 else 1e-9
    alpha = 0.9995

    # Iteration budget scales with n
    max_iters = max(20000, 200 * n * n)
    max_iters = min(max_iters, 500000)

    T = T0
    rng = random.Random(42)

    for it in range(max_iters):
        if T < T_min:
            break

        i = rng.randint(0, n - 1)
        j = rng.randint(0, n - 1)
        if i == j:
            continue
        if i > j:
            i, j = j, i

        # 2-opt delta calculation
        a = tour[i - 1]
        b = tour[i]
        c = tour[j]
        d = tour[(j + 1) % n]
        if a == c or b == d:
            continue

        delta = (dist_matrix[a, c] + dist_matrix[b, d]
                 - dist_matrix[a, b] - dist_matrix[c, d])

        if delta < 0 or rng.random() < math.exp(-delta / T):
            tour[i:j + 1] = tour[i:j + 1][::-1]
            current_length += delta
            if current_length < best_length:
                best_length = current_length
                best_tour = tour.copy()

        T *= alpha

    # Final verification
    best_length = tour_length(best_tour, dist_matrix)
    return best_tour.tolist(), float(best_length)