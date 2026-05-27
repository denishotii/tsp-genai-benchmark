import numpy as np


def _tour_length(tour, dist_matrix):
    tour = np.asarray(tour, dtype=int)
    if len(tour) <= 1:
        return 0.0
    return float(dist_matrix[tour, np.roll(tour, -1)].sum())


def _population_lengths(population, dist_matrix):
    return dist_matrix[population, np.roll(population, -1, axis=1)].sum(axis=1)


def _nearest_neighbor_tour(start, dist_matrix):
    n = dist_matrix.shape[0]
    tour = [start]
    unvisited = np.ones(n, dtype=bool)
    unvisited[start] = False
    current = start

    for _ in range(n - 1):
        candidates = np.flatnonzero(unvisited)
        nxt = candidates[np.argmin(dist_matrix[current, candidates])]
        tour.append(int(nxt))
        unvisited[nxt] = False
        current = int(nxt)

    return np.array(tour, dtype=int)


def _tournament_select(population, fitness, rng, tournament_size=3):
    idx = rng.integers(0, len(population), size=tournament_size)
    winner = idx[np.argmin(fitness[idx])]
    return population[winner]


def _order_crossover(parent1, parent2, rng):
    n = len(parent1)
    child = np.full(n, -1, dtype=int)

    a, b = sorted(rng.choice(n, size=2, replace=False))
    child[a:b + 1] = parent1[a:b + 1]

    used = np.zeros(n, dtype=bool)
    used[child[a:b + 1]] = True

    pos = (b + 1) % n
    for offset in range(n):
        gene = parent2[(b + 1 + offset) % n]
        if not used[gene]:
            child[pos] = gene
            used[gene] = True
            pos = (pos + 1) % n

    return child


def _swap_mutation(tour, rng):
    n = len(tour)
    if n > 1:
        i, j = rng.choice(n, size=2, replace=False)
        tour[i], tour[j] = tour[j], tour[i]


def _two_opt(tour, dist_matrix, max_passes):
    n = len(tour)
    if n < 4:
        return tour

    tour = list(map(int, tour))
    eps = 1e-12

    for _ in range(max_passes):
        best_delta = 0.0
        best_i = None
        best_k = None

        for i in range(n):
            a = tour[i - 1]
            b = tour[i]

            for k in range(i + 1, n):
                if i == 0 and k == n - 1:
                    continue

                c = tour[k]
                d = tour[(k + 1) % n]

                delta = (
                    dist_matrix[a, c]
                    + dist_matrix[b, d]
                    - dist_matrix[a, b]
                    - dist_matrix[c, d]
                )

                if delta < best_delta - eps:
                    best_delta = delta
                    best_i = i
                    best_k = k

        if best_i is None:
            break

        tour[best_i:best_k + 1] = reversed(tour[best_i:best_k + 1])

    return np.array(tour, dtype=int)


def solve(dist_matrix):
    dist_matrix = np.asarray(dist_matrix, dtype=float)
    n = dist_matrix.shape[0]

    if n == 0:
        return [], 0.0
    if n == 1:
        return [0], 0.0
    if n == 2:
        tour = [0, 1]
        return tour, float(dist_matrix[0, 1] + dist_matrix[1, 0])

    rng = np.random.default_rng(12345)

    pop_size = min(300, max(60, 6 * n))
    generations = max(200, min(1200, 60000 // pop_size))

    elite_count = max(1, pop_size // 20)
    tournament_size = 3
    crossover_rate = 0.9
    base_mutation_rate = 0.20

    population = np.empty((pop_size, n), dtype=int)

    heuristic_count = min(n, max(1, pop_size // 10))
    starts = np.linspace(0, n - 1, heuristic_count, dtype=int)

    fill = 0
    for s in starts:
        population[fill] = _nearest_neighbor_tour(int(s), dist_matrix)
        fill += 1
        if fill >= pop_size:
            break

    while fill < pop_size:
        population[fill] = rng.permutation(n)
        fill += 1

    fitness = _population_lengths(population, dist_matrix)
    best_idx = int(np.argmin(fitness))
    best_tour = population[best_idx].copy()
    best_length = float(fitness[best_idx])
    no_improvement = 0

    for _ in range(generations):
        elite_indices = np.argsort(fitness)[:elite_count]

        new_population = np.empty_like(population)
        new_population[:elite_count] = population[elite_indices]

        mutation_rate = base_mutation_rate
        if no_improvement > 50:
            mutation_rate = min(0.75, base_mutation_rate * 2.5)

        pos = elite_count

        while pos < pop_size:
            parent1 = _tournament_select(population, fitness, rng, tournament_size)
            parent2 = _tournament_select(population, fitness, rng, tournament_size)

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

            new_population[pos] = child1
            pos += 1

            if pos < pop_size:
                new_population[pos] = child2
                pos += 1

        population = new_population
        fitness = _population_lengths(population, dist_matrix)

        current_idx = int(np.argmin(fitness))
        current_length = float(fitness[current_idx])

        if current_length < best_length:
            best_length = current_length
            best_tour = population[current_idx].copy()
            no_improvement = 0
        else:
            no_improvement += 1

    if n <= 100:
        two_opt_passes = 30
    elif n <= 300:
        two_opt_passes = 10
    else:
        two_opt_passes = 4

    best_tour = _two_opt(best_tour, dist_matrix, two_opt_passes)
    best_length = _tour_length(best_tour, dist_matrix)

    return best_tour.tolist(), float(best_length)