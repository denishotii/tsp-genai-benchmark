import math
import numpy as np


def solve(dist_matrix):
    D = np.asarray(dist_matrix, dtype=float)
    n = D.shape[0]

    if n == 0:
        return [], 0.0
    if n == 1:
        return [0], 0.0
    if n == 2:
        return [0, 1], float(2.0 * D[0, 1])

    rng = np.random.default_rng(12345)

    def tour_length(tour):
        idx = np.asarray(tour, dtype=int)
        return float(D[idx, np.roll(idx, -1)].sum())

    def nearest_neighbor(start):
        unvisited = np.ones(n, dtype=bool)
        unvisited[start] = False
        tour = [int(start)]
        current = int(start)

        for _ in range(n - 1):
            candidates = np.where(unvisited, D[current], np.inf)
            nxt = int(np.argmin(candidates))
            tour.append(nxt)
            unvisited[nxt] = False
            current = nxt

        return tour

    def two_opt_delta(tour, i, k):
        a = tour[i - 1]
        b = tour[i]
        c = tour[k]
        d = tour[(k + 1) % n]
        return D[a, c] + D[b, d] - D[a, b] - D[c, d]

    # Build a good initial solution with a few nearest-neighbor starts.
    if n <= 500:
        num_starts = min(n, 8)
    elif n <= 2000:
        num_starts = min(n, 4)
    else:
        num_starts = 1

    starts = [0]
    if num_starts > 1:
        extra = rng.choice(n, size=num_starts - 1, replace=False)
        starts.extend(int(x) for x in extra if int(x) != 0)
        starts = list(dict.fromkeys(starts))[:num_starts]

    best_tour = None
    best_len = float("inf")

    for s in starts:
        t = nearest_neighbor(s)
        L = tour_length(t)
        if L < best_len:
            best_tour = t
            best_len = L

    tour = best_tour[:]
    current_len = best_len

    # Estimate an initial temperature from sampled worsening 2-opt moves.
    positive_deltas = []
    sample_count = min(2000, max(100, 20 * n))

    for _ in range(sample_count):
        i = int(rng.integers(0, n - 1))
        k = int(rng.integers(i + 1, n))
        if i == 0 and k == n - 1:
            continue

        delta = two_opt_delta(tour, i, k)
        if delta > 1e-12:
            positive_deltas.append(delta)

    if positive_deltas:
        avg_worsening = float(np.mean(positive_deltas))
        temperature = -avg_worsening / math.log(0.8)
    else:
        temperature = max(1.0, best_len / max(1, n))

    # Geometric cooling schedule.
    max_moves = min(600_000, max(20_000, 400 * n))
    steps_per_temp = max(100, min(1000, 5 * n))
    num_temps = max(1, max_moves // steps_per_temp)
    final_temperature_ratio = 1e-3
    alpha = final_temperature_ratio ** (1.0 / num_temps)

    for _ in range(num_temps):
        for _ in range(steps_per_temp):
            i = int(rng.integers(0, n - 1))
            k = int(rng.integers(i + 1, n))

            # Reversing the whole tour gives the same closed tour.
            if i == 0 and k == n - 1:
                continue

            delta = two_opt_delta(tour, i, k)

            if delta <= 0.0 or rng.random() < math.exp(-delta / temperature):
                tour[i:k + 1] = reversed(tour[i:k + 1])
                current_len += float(delta)

                if current_len < best_len:
                    best_len = current_len
                    best_tour = tour[:]

        temperature *= alpha
        if temperature <= 1e-15:
            break

    # Deterministic 2-opt polishing pass for moderate-size instances.
    tour = best_tour[:]
    current_len = tour_length(tour)

    if n <= 800:
        max_passes = 10 if n <= 300 else 4

        for _ in range(max_passes):
            improved = False

            for i in range(n - 1):
                for k in range(i + 1, n):
                    if i == 0 and k == n - 1:
                        continue

                    delta = two_opt_delta(tour, i, k)

                    if delta < -1e-12:
                        tour[i:k + 1] = reversed(tour[i:k + 1])
                        current_len += float(delta)
                        improved = True

            if not improved:
                break

    length = tour_length(tour)
    return [int(x) for x in tour], float(length)