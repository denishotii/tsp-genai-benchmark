import math
import numpy as np


def solve(dist_matrix):
    """
    Simulated Annealing TSP solver using a 2-opt neighborhood and geometric cooling.

    Step-by-step design:
      1. Problem / algorithm components:
         - We need a closed tour visiting every city exactly once.
         - The neighborhood move is 2-opt: choose indices i < j and reverse tour[i:j+1].
         - The length change is computed in O(1) by considering only the two broken
           edges and the two added edges.
         - Simulated annealing accepts improving/equal moves always, and worsening
           moves with probability exp(-delta / temperature).
         - Temperature follows a geometric schedule: T <- alpha * T.

      2. Data structures / main loop:
         - The current tour is stored as a NumPy integer array for efficient slicing.
         - The best tour found so far is stored as a separate copy.
         - City 0 is kept fixed at position 0; this removes equivalent rotations of
           the same cycle while still allowing all meaningful tours.
         - Each iteration samples a random 2-opt move, computes its delta, accepts or
           rejects it, and periodically cools the temperature.

      3. Edge cases:
         - n == 0: empty tour, length 0.
         - n == 1: single-city closed tour, length 0.
         - n <= 3: for symmetric TSP, all closed tours over 2 or 3 cities have the
           same set of undirected edges, so a simple ordered tour is sufficient.
         - Ties / zero-delta moves are accepted to allow neutral exploration.
    """

    dist = np.asarray(dist_matrix, dtype=float)

    if dist.ndim != 2 or dist.shape[0] != dist.shape[1]:
        raise ValueError("dist_matrix must be a square 2D NumPy array")

    n = dist.shape[0]

    def tour_length(t):
        if len(t) <= 1:
            return 0.0
        t = np.asarray(t, dtype=int)
        return float(np.sum(dist[t, np.roll(t, -1)]))

    if n == 0:
        return [], 0.0

    if n <= 3:
        tour = list(range(n))
        return tour, tour_length(tour)

    rng = np.random.default_rng()

    # Build a deterministic nearest-neighbor initial tour starting from city 0.
    # This gives simulated annealing a reasonably good starting point.
    tour = np.empty(n, dtype=int)
    tour[0] = 0

    unvisited = np.ones(n, dtype=bool)
    unvisited[0] = False

    current_city = 0
    for k in range(1, n):
        row = dist[current_city].copy()
        row[~unvisited] = np.inf
        next_city = int(np.argmin(row))
        tour[k] = next_city
        unvisited[next_city] = False
        current_city = next_city

    current_len = tour_length(tour)
    best_tour = tour.copy()
    best_len = current_len

    def two_opt_delta(t, i, j):
        """
        Delta for reversing t[i:j+1], with city 0 fixed outside the segment.

        Removed edges:
            t[i-1] -- t[i]
            t[j]   -- t[j+1]
        Added edges:
            t[i-1] -- t[j]
            t[i]   -- t[j+1]
        """
        a = t[i - 1]
        b = t[i]
        c = t[j]
        d = t[(j + 1) % n]
        return float((dist[a, c] + dist[b, d]) - (dist[a, b] + dist[c, d]))

    # Estimate an initial temperature from sampled uphill 2-opt moves.
    positive_deltas = []
    sample_count = min(1000, max(50, 10 * n))

    for _ in range(sample_count):
        i = int(rng.integers(1, n - 1))
        j = int(rng.integers(i + 1, n))
        delta = two_opt_delta(tour, i, j)
        if math.isfinite(delta) and delta > 1e-12:
            positive_deltas.append(delta)

    if positive_deltas:
        # Choose T so that an average uphill move has high initial acceptance.
        initial_temperature = float(np.mean(positive_deltas)) / -math.log(0.8)
    else:
        upper = dist[np.triu_indices(n, 1)]
        finite_positive = upper[np.isfinite(upper) & (upper > 0)]
        initial_temperature = float(np.mean(finite_positive)) if finite_positive.size else 1.0

    if not math.isfinite(initial_temperature) or initial_temperature <= 0.0:
        initial_temperature = 1.0

    temperature = initial_temperature
    min_temperature = initial_temperature * 1e-6

    # Iteration budget: scales with n but is capped for practicality.
    max_iterations = int(min(1_000_000, max(20_000, 2000 * n)))
    moves_per_temperature = max(10, n)

    # Compute a geometric cooling factor that reaches min_temperature near the
    # end of the iteration budget.
    cooling_steps = max(1, math.ceil(max_iterations / moves_per_temperature))
    alpha = math.exp(math.log(min_temperature / initial_temperature) / cooling_steps)

    iterations = 0

    while iterations < max_iterations and temperature > min_temperature:
        # Sample a valid 2-opt segment while keeping city 0 fixed.
        i = int(rng.integers(1, n - 1))
        j = int(rng.integers(i + 1, n))

        delta = two_opt_delta(tour, i, j)

        accept = False
        if delta <= 0.0:
            accept = True
        else:
            # Metropolis criterion.
            if rng.random() < math.exp(-delta / temperature):
                accept = True

        if accept:
            tour[i:j + 1] = tour[i:j + 1][::-1].copy()
            current_len += delta

            if current_len < best_len - 1e-12:
                best_len = current_len
                best_tour = tour.copy()

        iterations += 1

        if iterations % moves_per_temperature == 0:
            temperature *= alpha

    # Recompute exactly to avoid accumulated floating-point drift from deltas.
    best_len = tour_length(best_tour)

    return best_tour.astype(int).tolist(), float(best_len)