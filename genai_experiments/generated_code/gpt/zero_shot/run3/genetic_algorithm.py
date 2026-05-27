import numpy as np


def _tour_lengths(population, dist_matrix):
    """Vectorized closed-tour lengths for a population of permutations."""
    return dist_matrix[population, np.roll(population, -1, axis=1)].sum(axis=1)


def _nearest_neighbor_tour(dist_matrix, start):
    """Construct one permutation using the nearest-neighbor heuristic."""
    n = dist_matrix.shape[0]
    tour = np.empty(n, dtype=np.int64)
    visited = np.zeros(n, dtype=bool)

    current = int(start)
    tour[0] = current
    visited[current] = True

    for k in range(1, n):
        row = dist_matrix[current].copy()
        row[visited] = np.inf
        current = int(np.argmin(row))
        tour[k] = current
        visited[current] = True

    return tour


def _make_initial_population(dist_matrix, pop_size, rng):
    """Permutation-encoded initial population."""
    n = dist_matrix.shape[0]
    population = np.empty((pop_size, n), dtype=np.int64)

    idx = 0

    # Add a few nearest-neighbor tours to seed the population.
    greedy_count = min(pop_size // 5, n, 20)
    if greedy_count > 0:
        starts = rng.permutation(n)[:greedy_count]
        for s in starts:
            population[idx] = _nearest_neighbor_tour(dist_matrix, int(s))
            idx += 1

    # Fill the rest with random permutations.
    while idx < pop_size:
        population[idx] = rng.permutation(n)
        idx += 1

    return population


def _tournament_select(lengths, tournament_size, rng):
    """Tournament selection: lower tour length is better."""
    pop_size = len(lengths)
    candidates = rng.integers(0, pop_size, size=tournament_size)
    return int(candidates[np.argmin(lengths[candidates])])


def _order_crossover(parent1, parent2, rng):
    """Order Crossover OX for permutation encodings."""
    n = len(parent1)

    if n < 2:
        return parent1.copy()

    c1, c2 = sorted(rng.choice(n, size=2, replace=False))
    c2 += 1

    child = np.empty(n, dtype=np.int64)
    child.fill(-1)

    child[c1:c2] = parent1[c1:c2]

    used = np.zeros(n, dtype=bool)
    used[child[c1:c2]] = True

    pos = c2 % n

    for gene in np.concatenate((parent2[c2:], parent2[:c2])):
        if not used[gene]:
            child[pos] = gene
            pos = (pos + 1) % n

    return child


def _swap_mutation(tour, rng):
    """Swap mutation for a permutation."""
    n = len(tour)
    if n < 2:
        return

    i, j = rng.choice(n, size=2, replace=False)
    tour[i], tour[j] = tour[j], tour[i]


def _two_opt(tour, dist_matrix, max_passes):
    """Small final 2-opt polish for the best GA solution."""
    n = len(tour)
    if n < 4:
        return tour

    for _ in range(max_passes):
        improved = False

        for i in range(n - 1):
            a = tour[i - 1]
            b = tour[i]

            for j in range(i + 1, n):
                if i == 0 and j == n - 1:
                    continue

                c = tour[j]
                d = tour[(j + 1) % n]

                old = dist_matrix[a, b] + dist_matrix[c, d]
                new = dist_matrix[a, c] + dist_matrix[b, d]

                if new + 1e-12 < old:
                    tour[i:j + 1] = tour[i:j + 1][::-1]
                    improved = True

        if not improved:
            break

    return tour


def solve(dist_matrix):
    dist_matrix = np.asarray(dist_matrix, dtype=float)
    n = dist_matrix.shape[0]

    if n == 0:
        return [], 0.0

    if n == 1:
        return [0], 0.0

    rng = np.random.default_rng(123456789)

    # GA parameters.
    if n <= 30:
        pop_size = max(60, 10 * n)
        generations = 600
    elif n <= 100:
        pop_size = max(120, min(300, 6 * n))
        generations = 500
    elif n <= 300:
        pop_size = 250
        generations = 300
    else:
        pop_size = 200
        generations = 180

    elite_count = max(1, pop_size // 20)
    tournament_size = 4
    crossover_rate = 0.90
    mutation_rate = 0.20

    population = _make_initial_population(dist_matrix, pop_size, rng)
    lengths = _tour_lengths(population, dist_matrix)

    best_idx = int(np.argmin(lengths))
    best_tour = population[best_idx].copy()
    best_length = float(lengths[best_idx])

    for _ in range(generations):
        next_population = np.empty_like(population)

        # Elitism: copy the best individuals unchanged.
        elite_idx = np.argpartition(lengths, elite_count - 1)[:elite_count]
        elite_idx = elite_idx[np.argsort(lengths[elite_idx])]
        next_population[:elite_count] = population[elite_idx]

        fill = elite_count

        while fill < pop_size:
            p1_idx = _tournament_select(lengths, tournament_size, rng)
            p2_idx = _tournament_select(lengths, tournament_size, rng)

            parent1 = population[p1_idx]
            parent2 = population[p2_idx]

            if rng.random() < crossover_rate:
                child1 = _order_crossover(parent1, parent2, rng)
                child2 = _order_crossover(parent2, parent1, rng)
            else:
                child1 = parent1.copy()
                child2 = parent2.copy()

            if rng.random() < mutation_rate:
                _swap_mutation(child1, rng)

            if rng.random() < mutation_rate:
                _swap_mutation(child2, rng)

            next_population[fill] = child1
            fill += 1

            if fill < pop_size:
                next_population[fill] = child2
                fill += 1

        population = next_population
        lengths = _tour_lengths(population, dist_matrix)

        gen_best_idx = int(np.argmin(lengths))
        gen_best_length = float(lengths[gen_best_idx])

        if gen_best_length < best_length:
            best_length = gen_best_length
            best_tour = population[gen_best_idx].copy()

    # Final local polish of the best GA tour.
    if n <= 150:
        max_2opt_passes = 4
    elif n <= 500:
        max_2opt_passes = 2
    else:
        max_2opt_passes = 1

    best_tour = _two_opt(best_tour, dist_matrix, max_2opt_passes)
    best_length = float(dist_matrix[best_tour, np.roll(best_tour, -1)].sum())

    return best_tour.tolist(), best_length