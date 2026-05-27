import numpy as np


def _tour_length(tour, dist):
    tour = np.asarray(tour, dtype=np.int64)
    if tour.size == 0:
        return 0.0
    return float(np.sum(dist[tour, np.roll(tour, -1)], dtype=np.float64))


def _population_lengths(pop, dist):
    nxt = np.concatenate((pop[:, 1:], pop[:, :1]), axis=1)
    return np.sum(dist[pop, nxt], axis=1, dtype=np.float64)


def _nearest_neighbor_tour(dist, start):
    n = dist.shape[0]
    tour = np.empty(n, dtype=np.int64)
    unvisited = np.ones(n, dtype=bool)

    current = int(start)
    for pos in range(n):
        tour[pos] = current
        unvisited[current] = False

        if pos < n - 1:
            candidates = np.flatnonzero(unvisited)
            current = int(candidates[np.argmin(dist[current, candidates])])

    return tour


def _tournament_select(pop, lengths, tournament_size, rng):
    candidates = rng.integers(0, pop.shape[0], size=tournament_size)
    winner = candidates[np.argmin(lengths[candidates])]
    return pop[winner]


def _order_crossover(parent1, parent2, rng):
    n = parent1.size

    if n < 2:
        return parent1.copy()

    a, b = sorted(rng.choice(n, size=2, replace=False))

    child = np.full(n, -1, dtype=np.int64)
    child[a:b + 1] = parent1[a:b + 1]

    used = np.zeros(n, dtype=bool)
    used[parent1[a:b + 1]] = True

    pos = (b + 1) % n
    for k in range(n):
        city = parent2[(b + 1 + k) % n]
        if not used[city]:
            child[pos] = city
            used[city] = True
            pos = (pos + 1) % n

    return child


def _swap_mutation(tour, mutation_rate, rng):
    n = tour.size
    if n >= 2 and rng.random() < mutation_rate:
        i, j = rng.choice(n, size=2, replace=False)
        tour[i], tour[j] = tour[j], tour[i]


def _two_opt(tour, dist, max_passes=2):
    n = tour.size
    if n < 4:
        return tour

    tour = tour.copy()

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

                old = dist[a, b] + dist[c, d]
                new = dist[a, c] + dist[b, d]

                if new + 1e-12 < old:
                    tour[i:j + 1] = tour[i:j + 1][::-1]
                    improved = True

        if not improved:
            break

    return tour


def solve(dist_matrix):
    dist = np.asarray(dist_matrix, dtype=np.float64)
    n = dist.shape[0]

    if n == 0:
        return [], 0.0

    if n == 1:
        return [0], 0.0

    if n == 2:
        tour = [0, 1]
        return tour, _tour_length(tour, dist)

    rng = np.random.default_rng(123456789)

    if n <= 50:
        pop_size = max(80, 8 * n)
    elif n <= 200:
        pop_size = min(500, 6 * n)
    elif n <= 500:
        pop_size = 300
    else:
        pop_size = 200

    pop_size = int(pop_size)

    if n <= 30:
        generations = 600
    elif n <= 100:
        generations = 300
    elif n <= 300:
        generations = 180
    else:
        generations = 120

    elite_count = max(1, pop_size // 20)
    tournament_size = 4
    crossover_rate = 0.95
    mutation_rate = 0.20 if n <= 100 else 0.15
    immigrant_count = max(1, pop_size // 100)

    pop = np.empty((pop_size, n), dtype=np.int64)

    fill = 0

    nn_count = min(n, max(1, pop_size // 10))
    starts = rng.choice(n, size=nn_count, replace=False)
    for s in starts:
        pop[fill] = _nearest_neighbor_tour(dist, int(s))
        fill += 1

    while fill < pop_size:
        pop[fill] = rng.permutation(n)
        fill += 1

    lengths = _population_lengths(pop, dist)

    for _ in range(generations):
        order = np.argsort(lengths)

        new_pop = np.empty_like(pop)
        new_pop[:elite_count] = pop[order[:elite_count]]

        fill = elite_count

        while fill < pop_size:
            p1 = _tournament_select(pop, lengths, tournament_size, rng)
            p2 = _tournament_select(pop, lengths, tournament_size, rng)

            if rng.random() < crossover_rate:
                c1 = _order_crossover(p1, p2, rng)
                c2 = _order_crossover(p2, p1, rng)
            else:
                c1 = p1.copy()
                c2 = p2.copy()

            _swap_mutation(c1, mutation_rate, rng)
            _swap_mutation(c2, mutation_rate, rng)

            new_pop[fill] = c1
            fill += 1

            if fill < pop_size:
                new_pop[fill] = c2
                fill += 1

        for k in range(immigrant_count):
            idx = pop_size - 1 - k
            if idx >= elite_count:
                new_pop[idx] = rng.permutation(n)

        pop = new_pop
        lengths = _population_lengths(pop, dist)

    best_idx = int(np.argmin(lengths))
    best_tour = pop[best_idx].copy()

    if n <= 150:
        best_tour = _two_opt(best_tour, dist, max_passes=5)
    elif n <= 500:
        best_tour = _two_opt(best_tour, dist, max_passes=2)
    else:
        best_tour = _two_opt(best_tour, dist, max_passes=1)

    best_length = _tour_length(best_tour, dist)

    return best_tour.tolist(), float(best_length)