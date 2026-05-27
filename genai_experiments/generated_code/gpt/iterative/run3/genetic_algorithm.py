import numpy as np


def _tour_length(dist_matrix, tour):
    tour = np.asarray(tour, dtype=np.int64)
    if len(tour) <= 1:
        return 0.0
    return float(dist_matrix[tour, np.roll(tour, -1)].sum())


def _population_lengths(dist_matrix, population):
    return dist_matrix[population, np.roll(population, -1, axis=1)].sum(axis=1)


def _nearest_neighbor_tour(dist_matrix, start):
    n = dist_matrix.shape[0]
    tour = np.empty(n, dtype=np.int64)
    unvisited = np.ones(n, dtype=bool)

    city = int(start)
    for k in range(n):
        tour[k] = city
        unvisited[city] = False

        if k < n - 1:
            row = dist_matrix[city]
            city = int(np.argmin(np.where(unvisited, row, np.inf)))

    return tour


def _order_crossover(parent1, parent2, rng):
    n = len(parent1)
    child = np.full(n, -1, dtype=np.int64)

    a, b = rng.integers(0, n, size=2)
    if a > b:
        a, b = b, a

    child[a:b + 1] = parent1[a:b + 1]

    used = np.zeros(n, dtype=bool)
    used[child[a:b + 1]] = True

    pos = (b + 1) % n
    for offset in range(n):
        city = parent2[(b + 1 + offset) % n]
        if not used[city]:
            child[pos] = city
            used[city] = True
            pos = (pos + 1) % n

    return child


def _swap_mutation(tour, rng, mutation_rate):
    n = len(tour)
    if n > 1 and rng.random() < mutation_rate:
        i, j = rng.choice(n, size=2, replace=False)
        tour[i], tour[j] = tour[j], tour[i]


def _tournament_select(lengths, rng, tournament_size):
    candidates = rng.integers(0, len(lengths), size=tournament_size)
    return int(candidates[np.argmin(lengths[candidates])])


def _two_opt(dist_matrix, tour, max_passes):
    n = len(tour)
    if n < 4:
        return tour

    tour = tour.copy()

    for _ in range(max_passes):
        improved = False

        for i in range(n - 1):
            a = tour[i]
            b = tour[(i + 1) % n]

            for j in range(i + 2, n):
                if i == 0 and j == n - 1:
                    continue

                c = tour[j]
                d = tour[(j + 1) % n]

                delta = (
                    dist_matrix[a, c]
                    + dist_matrix[b, d]
                    - dist_matrix[a, b]
                    - dist_matrix[c, d]
                )

                if delta < -1e-12:
                    tour[i + 1:j + 1] = tour[i + 1:j + 1][::-1]
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
    if n == 2:
        tour = [0, 1]
        return tour, _tour_length(dist_matrix, tour)

    rng = np.random.default_rng(12345)

    if n <= 20:
        population_size = 120
        generations = 500
        mutation_rate = 0.20
    elif n <= 60:
        population_size = 160
        generations = 600
        mutation_rate = 0.14
    elif n <= 150:
        population_size = 200
        generations = 500
        mutation_rate = 0.10
    elif n <= 400:
        population_size = 220
        generations = 350
        mutation_rate = 0.08
    else:
        population_size = 160
        generations = 250
        mutation_rate = 0.06

    crossover_rate = 0.90
    tournament_size = 3
    elite_count = max(2, population_size // 20)
    patience = max(100, generations // 3)

    population = np.empty((population_size, n), dtype=np.int64)

    idx = 0

    nn_count = min(n, max(2, population_size // 10), 25)
    starts = rng.permutation(n)[:nn_count]
    for start in starts:
        population[idx] = _nearest_neighbor_tour(dist_matrix, start)
        idx += 1

    while idx < population_size:
        population[idx] = rng.permutation(n)
        idx += 1

    lengths = _population_lengths(dist_matrix, population)

    best_idx = int(np.argmin(lengths))
    best_tour = population[best_idx].copy()
    best_length = float(lengths[best_idx])

    no_improvement = 0

    for generation in range(generations):
        elite_indices = np.argsort(lengths)[:elite_count]

        new_population = np.empty_like(population)
        new_population[:elite_count] = population[elite_indices]

        current_mutation_rate = mutation_rate
        if no_improvement > 50:
            current_mutation_rate = min(0.50, mutation_rate * 2.0)

        fill = elite_count

        while fill < population_size:
            p1_idx = _tournament_select(lengths, rng, tournament_size)
            p2_idx = _tournament_select(lengths, rng, tournament_size)

            parent1 = population[p1_idx]
            parent2 = population[p2_idx]

            if rng.random() < crossover_rate:
                child1 = _order_crossover(parent1, parent2, rng)
                child2 = _order_crossover(parent2, parent1, rng)
            else:
                child1 = parent1.copy()
                child2 = parent2.copy()

            _swap_mutation(child1, rng, current_mutation_rate)
            new_population[fill] = child1
            fill += 1

            if fill < population_size:
                _swap_mutation(child2, rng, current_mutation_rate)
                new_population[fill] = child2
                fill += 1

        population = new_population
        lengths = _population_lengths(dist_matrix, population)

        gen_best_idx = int(np.argmin(lengths))
        gen_best_length = float(lengths[gen_best_idx])

        if gen_best_length < best_length - 1e-12:
            best_length = gen_best_length
            best_tour = population[gen_best_idx].copy()
            no_improvement = 0
        else:
            no_improvement += 1

        if no_improvement > 0 and no_improvement % 50 == 0:
            immigrant_count = max(1, population_size // 10)
            worst_indices = np.argsort(lengths)[-immigrant_count:]
            for wi in worst_indices:
                population[wi] = rng.permutation(n)
            lengths[worst_indices] = _population_lengths(dist_matrix, population[worst_indices])

        if no_improvement >= patience:
            break

    if n <= 80:
        two_opt_passes = 25
    elif n <= 200:
        two_opt_passes = 10
    elif n <= 500:
        two_opt_passes = 3
    else:
        two_opt_passes = 1

    best_tour = _two_opt(dist_matrix, best_tour, two_opt_passes)
    best_length = _tour_length(dist_matrix, best_tour)

    return [int(x) for x in best_tour], float(best_length)