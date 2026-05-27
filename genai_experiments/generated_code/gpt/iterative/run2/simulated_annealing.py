import math
import numpy as np


def solve(dist_matrix):
    d = np.asarray(dist_matrix, dtype=float)
    n = d.shape[0]

    if n == 0:
        return [], 0.0
    if n == 1:
        return [0], 0.0
    if n == 2:
        return [0, 1], float(d[0, 1] + d[1, 0])

    rng = np.random.default_rng(123456789)

    def tour_length(tour):
        return float(np.sum(d[tour, np.roll(tour, -1)]))

    def nearest_neighbor(start):
        tour = np.empty(n, dtype=np.int64)
        unvisited = np.ones(n, dtype=bool)

        current = int(start)
        for k in range(n):
            tour[k] = current
            unvisited[current] = False

            if k == n - 1:
                break

            candidates = np.nonzero(unvisited)[0]
            current = int(candidates[np.argmin(d[current, candidates])])

        return tour

    # Build a good initial solution using a few nearest-neighbor starts.
    starts = [0]
    extra = min(n - 1, 7 if n <= 200 else 3)
    if extra > 0:
        starts.extend(map(int, rng.choice(np.arange(1, n), size=extra, replace=False)))

    current_tour = None
    current_len = float("inf")

    for s in starts:
        t = nearest_neighbor(s)
        L = tour_length(t)
        if L < current_len:
            current_tour = t
            current_len = L

    best_tour = current_tour.copy()
    best_len = float(current_len)

    # Estimate an initial temperature from sampled uphill 2-opt moves.
    positive_deltas = []
    samples = min(3000, max(200, 30 * n))

    for _ in range(samples):
        i = int(rng.integers(0, n - 1))
        j = int(rng.integers(i + 1, n))

        if i == 0 and j == n - 1:
            continue

        a = current_tour[i - 1]
        b = current_tour[i]
        c = current_tour[j]
        e = current_tour[(j + 1) % n]

        delta = d[a, c] + d[b, e] - d[a, b] - d[c, e]
        if delta > 1e-12:
            positive_deltas.append(float(delta))

    if positive_deltas:
        temp = float(np.mean(positive_deltas)) / (-math.log(0.8))
    else:
        temp = max(current_len / n, 1e-9)

    # Simulated Annealing with geometric cooling.
    max_iter = int(max(5000, min(300000, 1200 * n)))
    alpha = (1e-4) ** (1.0 / max_iter)

    for _ in range(max_iter):
        i = int(rng.integers(0, n - 1))
        j = int(rng.integers(i + 1, n))

        if i == 0 and j == n - 1:
            temp *= alpha
            continue

        a = current_tour[i - 1]
        b = current_tour[i]
        c = current_tour[j]
        e = current_tour[(j + 1) % n]

        delta = float(d[a, c] + d[b, e] - d[a, b] - d[c, e])

        if delta <= 0.0 or rng.random() < math.exp(-delta / temp):
            current_tour[i:j + 1] = current_tour[i:j + 1][::-1].copy()
            current_len += delta

            if current_len < best_len:
                best_len = float(current_len)
                best_tour = current_tour.copy()

        temp *= alpha

    # Final deterministic 2-opt descent polish.
    tour = best_tour.copy()
    length = float(best_len)

    if n <= 600:
        max_passes = 25 if n <= 150 else 10 if n <= 300 else 4
        eps = 1e-12

        for _ in range(max_passes):
            improved = False

            for i in range(n - 1):
                for j in range(i + 1, n):
                    if i == 0 and j == n - 1:
                        continue

                    a = tour[i - 1]
                    b = tour[i]
                    c = tour[j]
                    e = tour[(j + 1) % n]

                    delta = float(d[a, c] + d[b, e] - d[a, b] - d[c, e])

                    if delta < -eps:
                        tour[i:j + 1] = tour[i:j + 1][::-1].copy()
                        length += delta
                        improved = True
                        break

            if not improved:
                break

    length = tour_length(tour)
    return [int(x) for x in tour.tolist()], float(length)