import math
import numpy as np


def solve(dist_matrix):
    """
    Step-by-step design notes:

    1. Problem and algorithm:
       - We need a closed TSP tour visiting every city exactly once.
       - We use Simulated Annealing:
         * Start from an initial feasible tour.
         * Repeatedly propose a neighboring tour using a 2-opt move.
         * Accept improving moves always.
         * Accept worsening moves with probability exp(-delta / temperature).
         * Decrease temperature geometrically.

    2. Data structures and main loop:
       - The tour is stored as a Python list of city indices.
       - The distance matrix is converted to a NumPy float array.
       - A 2-opt move chooses indices i < j and reverses tour[i:j+1].
       - The change in closed-tour length is computed in O(1):
             remove edges (a,b), (c,d)
             add edges    (a,c), (b,d)
         where:
             a = tour[i-1], b = tour[i],
             c = tour[j],   d = tour[(j+1) % n]
       - The main annealing loop uses a geometric cooling schedule.

    3. Edge cases:
       - n == 0: return empty tour with length 0.
       - n == 1: single-city closed tour has length 0.
       - n == 2: closed tour length is d[0,1] + d[1,0].
       - Ties or zero-delta moves are accepted.
    """

    dist = np.asarray(dist_matrix, dtype=float)

    if dist.ndim != 2 or dist.shape[0] != dist.shape[1]:
        raise ValueError("dist_matrix must be a square NumPy array")

    n = dist.shape[0]

    def tour_length(t):
        if len(t) <= 1:
            return 0.0
        total = 0.0
        for k in range(len(t)):
            total += dist[t[k], t[(k + 1) % len(t)]]
        return float(total)

    if n == 0:
        return [], 0.0

    if n == 1:
        return [0], 0.0

    if n == 2:
        tour = [0, 1]
        return tour, float(dist[0, 1] + dist[1, 0])

    # Initial tour: nearest-neighbor heuristic from city 0.
    unvisited = np.ones(n, dtype=bool)
    unvisited[0] = False
    tour = [0]
    current = 0

    for _ in range(n - 1):
        candidates = np.flatnonzero(unvisited)
        next_city = int(candidates[np.argmin(dist[current, candidates])])
        tour.append(next_city)
        unvisited[next_city] = False
        current = next_city

    current_len = tour_length(tour)
    best_tour = tour.copy()
    best_len = current_len

    rng = np.random.default_rng()

    def two_opt_delta(t, i, j):
        a = t[i - 1]
        b = t[i]
        c = t[j]
        d = t[(j + 1) % n]
        return float(dist[a, c] + dist[b, d] - dist[a, b] - dist[c, d])

    def random_two_opt_indices():
        # Avoid reversing the entire tour, which only changes orientation.
        while True:
            i = int(rng.integers(0, n - 1))
            j = int(rng.integers(i + 1, n))
            if not (i == 0 and j == n - 1):
                return i, j

    # Estimate a reasonable starting temperature from sampled uphill moves.
    sampled_positive_deltas = []
    sample_count = min(2000, max(100, 20 * n))

    for _ in range(sample_count):
        i, j = random_two_opt_indices()
        delta = two_opt_delta(tour, i, j)
        if delta > 0.0:
            sampled_positive_deltas.append(delta)

    if sampled_positive_deltas:
        avg_uphill = float(np.mean(sampled_positive_deltas))
    else:
        avg_edge = current_len / n if n > 0 else 1.0
        avg_uphill = max(avg_edge * 0.01, 1e-12)

    # Choose T0 so an average uphill move is initially accepted with ~80% probability.
    initial_temperature = max(avg_uphill / (-math.log(0.80)), 1e-12)

    # Geometric cooling parameters.
    chain_length = max(100, 10 * n)
    max_moves = min(1_500_000, max(20_000, 1500 * n))
    outer_steps = max(1, max_moves // chain_length)

    final_temperature_ratio = 1e-4
    cooling_rate = final_temperature_ratio ** (1.0 / outer_steps)

    temperature = initial_temperature
    eps = 1e-12

    for _ in range(outer_steps):
        for _ in range(chain_length):
            i, j = random_two_opt_indices()
            delta = two_opt_delta(tour, i, j)

            if delta <= eps:
                accept = True
            else:
                accept = rng.random() < math.exp(-delta / temperature)

            if accept:
                tour[i:j + 1] = reversed(tour[i:j + 1])
                current_len += delta

                if current_len < best_len - eps:
                    best_len = current_len
                    best_tour = tour.copy()

        temperature *= cooling_rate

    # Recompute exactly to avoid small floating-point drift from accumulated deltas.
    final_tour = best_tour
    final_length = tour_length(final_tour)

    return final_tour, float(final_length)