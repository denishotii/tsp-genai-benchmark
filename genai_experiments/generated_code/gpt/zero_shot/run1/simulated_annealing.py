import math
import itertools
import numpy as np


def solve(dist_matrix):
    D = np.asarray(dist_matrix, dtype=float)
    n = D.shape[0]

    def closed_length(tour):
        if len(tour) <= 1:
            return 0.0
        idx = np.asarray(tour, dtype=int)
        return float(D[idx, np.roll(idx, -1)].sum())

    if n == 0:
        return [], 0.0
    if n == 1:
        return [0], 0.0
    if n == 2:
        tour = [0, 1]
        return tour, closed_length(tour)

    # Exact solution for very small instances; the general solver below is SA.
    if n <= 9:
        best_tour = None
        best_len = float("inf")
        for perm in itertools.permutations(range(1, n)):
            tour = [0] + list(perm)
            length = closed_length(tour)
            if length < best_len:
                best_len = length
                best_tour = tour
        return best_tour, float(best_len)

    rng = np.random.default_rng(123456789)

    def nearest_neighbor_start(start=0):
        unvisited = np.ones(n, dtype=bool)
        unvisited[start] = False
        tour = [int(start)]
        current = int(start)

        for _ in range(n - 1):
            candidates = np.flatnonzero(unvisited)
            nxt = int(candidates[np.argmin(D[current, candidates])])
            tour.append(nxt)
            unvisited[nxt] = False
            current = nxt

        return tour

    def random_2opt_indices():
        while True:
            i = int(rng.integers(0, n))
            j = int(rng.integers(0, n))
            if i == j:
                continue
            if i > j:
                i, j = j, i
            if i == 0 and j == n - 1:
                continue
            return i, j

    def two_opt_delta(tour, i, j):
        a = tour[i - 1]
        b = tour[i]
        c = tour[j]
        d = tour[(j + 1) % n]
        return float(D[a, c] + D[b, d] - D[a, b] - D[c, d])

    def two_opt_descent(tour, length, eval_budget):
        evals = 0
        improved = True
        eps = 1e-12

        while improved and evals < eval_budget:
            improved = False

            for i in range(n - 1):
                if evals >= eval_budget:
                    break

                for j in range(i + 1, n):
                    if i == 0 and j == n - 1:
                        continue

                    delta = two_opt_delta(tour, i, j)
                    evals += 1

                    if delta < -eps:
                        tour[i:j + 1] = reversed(tour[i:j + 1])
                        length += delta
                        improved = True
                        break

                    if evals >= eval_budget:
                        break

                if improved:
                    break

        return tour, length

    # Build an initial tour using nearest neighbor.
    starts = [0]
    if n <= 300:
        extra = min(4, n - 1)
        starts.extend(rng.choice(np.arange(1, n), size=extra, replace=False).tolist())

    tour = None
    current_len = float("inf")
    for s in starts:
        candidate = nearest_neighbor_start(s)
        candidate_len = closed_length(candidate)
        if candidate_len < current_len:
            tour = candidate
            current_len = candidate_len

    # Light initial 2-opt polishing.
    if n <= 250:
        init_budget = min(300_000, 20 * n * n)
    elif n <= 1000:
        init_budget = 100_000
    else:
        init_budget = 50_000

    tour, current_len = two_opt_descent(tour, current_len, init_budget)

    best_tour = tour.copy()
    best_len = current_len

    # Estimate an initial temperature from positive 2-opt move costs.
    positive_deltas = []
    samples = max(100, min(5000, 20 * n))

    for _ in range(samples):
        i, j = random_2opt_indices()
        delta = two_opt_delta(tour, i, j)
        if delta > 1e-12:
            positive_deltas.append(delta)

    if positive_deltas:
        avg_positive = float(np.mean(positive_deltas))
        temperature = -avg_positive / math.log(0.8)
    else:
        avg_edge = current_len / n if n > 0 else 1.0
        temperature = max(avg_edge, 1.0)

    temperature = max(temperature, 1e-12)
    final_temperature = temperature * 1e-4

    # Geometric cooling schedule.
    if n <= 100:
        levels = 350
        trials_per_level = max(50, 10 * n)
    elif n <= 500:
        levels = 300
        trials_per_level = max(100, 5 * n)
    else:
        levels = 250
        trials_per_level = min(max(1000, 2 * n), 5000)

    alpha = (final_temperature / temperature) ** (1.0 / max(1, levels - 1))

    for _ in range(levels):
        for _ in range(trials_per_level):
            i, j = random_2opt_indices()
            delta = two_opt_delta(tour, i, j)

            if delta <= 0.0 or rng.random() < math.exp(-delta / temperature):
                tour[i:j + 1] = reversed(tour[i:j + 1])
                current_len += delta

                if current_len < best_len:
                    best_len = current_len
                    best_tour = tour.copy()

        temperature *= alpha
        temperature = max(temperature, 1e-300)

    # Final 2-opt polish from the best SA tour.
    tour = best_tour.copy()
    current_len = best_len

    if n <= 250:
        final_budget = min(500_000, 40 * n * n)
    elif n <= 1000:
        final_budget = 150_000
    else:
        final_budget = 75_000

    tour, current_len = two_opt_descent(tour, current_len, final_budget)

    length = closed_length(tour)
    return [int(x) for x in tour], float(length)