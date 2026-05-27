import math
import numpy as np


def solve(dist_matrix):
    """Solve a symmetric TSP instance using simulated annealing with 2-opt moves."""
    dist = np.asarray(dist_matrix, dtype=float)
    n = dist.shape[0]

    if n == 0:
        return [], 0.0
    if n == 1:
        return [0], 0.0

    rng = np.random.default_rng()

    def tour_length(tour):
        return float(sum(dist[tour[i], tour[(i + 1) % n]] for i in range(n)))

    def nearest_neighbor_tour():
        unvisited = np.ones(n, dtype=bool)
        tour = [0]
        unvisited[0] = False

        for _ in range(n - 1):
            last = tour[-1]
            candidates = np.where(unvisited)[0]
            nxt = candidates[np.argmin(dist[last, candidates])]
            tour.append(int(nxt))
            unvisited[nxt] = False

        return tour

    def random_2opt_indices():
        i, j = sorted(rng.choice(n, size=2, replace=False))
        return int(i), int(j)

    def two_opt_delta(tour, i, j):
        """Cost change from reversing tour[i:j+1]."""
        if i == 0 and j == n - 1:
            return 0.0

        a = tour[i - 1]
        b = tour[i]
        c = tour[j]
        d = tour[(j + 1) % n]

        old_cost = dist[a, b] + dist[c, d]
        new_cost = dist[a, c] + dist[b, d]
        return float(new_cost - old_cost)

    tour = nearest_neighbor_tour()
    current_length = tour_length(tour)

    best_tour = tour[:]
    best_length = current_length

    # Estimate a reasonable initial temperature from sampled uphill 2-opt moves.
    uphill_deltas = []
    for _ in range(max(50, 5 * n)):
        i, j = random_2opt_indices()
        if i == 0 and j == n - 1:
            continue
        delta = two_opt_delta(tour, i, j)
        if delta > 0:
            uphill_deltas.append(delta)

    if uphill_deltas:
        initial_temp = float(np.mean(uphill_deltas) / -math.log(0.8))
    else:
        initial_temp = max(1.0, current_length / max(n, 1))

    temperature = initial_temp
    final_temperature = initial_temp * 1e-4
    cooling_rate = 0.995
    moves_per_temperature = max(100, 10 * n)

    while temperature > final_temperature:
        for _ in range(moves_per_temperature):
            i, j = random_2opt_indices()

            # Reversing the whole cyclic tour is equivalent to the same tour.
            if i == 0 and j == n - 1:
                continue

            delta = two_opt_delta(tour, i, j)

            if delta <= 0.0 or rng.random() < math.exp(-delta / temperature):
                tour[i:j + 1] = reversed(tour[i:j + 1])
                current_length += delta

                if current_length < best_length:
                    best_length = current_length
                    best_tour = tour[:]

        temperature *= cooling_rate

    # Recompute to remove any small accumulated floating-point drift.
    best_length = tour_length(best_tour)

    return best_tour, float(best_length)