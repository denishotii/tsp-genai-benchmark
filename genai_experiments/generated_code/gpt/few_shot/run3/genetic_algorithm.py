import numpy as np

def solve(dist_matrix):
    """Solve a symmetric TSP instance using a Genetic Algorithm."""
    dist = np.asarray(dist_matrix, dtype=float)
    n = dist.shape[0]

    if n == 0:
        return [], 0.0
    if n == 1:
        return [0], 0.0
    if n == 2:
        return [0, 1], float(dist[0, 1] + dist[1, 0])

    rng = np.random.default_rng(12345)

    pop_size = int(min(180, max(40, 6 * n)))
    generations = int(max(80, min(800, 4_000_000 // (pop_size * n))))
    tournament_size = 4
    elite_count = max(1, pop_size // 20)
    crossover_rate = 0.90
    mutation_rate = min(0.30, max(0.03, 2.0 / n))

    def tour_length(tour):
        return float(dist[tour, np.roll(tour, -1)].sum())

    def population_lengths(pop):
        return dist[pop, np.roll(pop, -1, axis=1)].sum(axis=1)

    def nearest_neighbor(start):
        tour = np.empty(n, dtype=int)
        unvisited = np.ones(n, dtype=bool)
        city = start

        for i in range(n):
            tour[i] = city
            unvisited[city] = False
            if i + 1 < n:
                candidates = np.flatnonzero(unvisited)
                city = candidates[np.argmin(dist[city, candidates])]

        return tour

    def tournament_select(lengths):
        candidates = rng.integers(0, len(lengths), size=tournament_size)
        return candidates[np.argmin(lengths[candidates])]

    def order_crossover(parent1, parent2):
        child = np.full(n, -1, dtype=int)

        a, b = sorted(rng.choice(n, size=2, replace=False))
        b += 1

        child[a:b] = parent1[a:b]

        used = np.zeros(n, dtype=bool)
        used[parent1[a:b]] = True

        pos = b % n
        for city in np.concatenate((parent2[b:], parent2[:b])):
            if not used[city]:
                child[pos] = city
                pos = (pos + 1) % n

        return child

    def swap_mutation(tour):
        i, j = rng.choice(n, size=2, replace=False)
        tour[i], tour[j] = tour[j], tour[i]

    def two_opt(tour, max_passes):
        tour = np.array(tour, dtype=int, copy=True)

        for _ in range(max_passes):
            improved = False

            for i in range(1, n - 1):
                a = tour[i - 1]
                b = tour[i]

                for j in range(i + 2, n + 1):
                    c = tour[j - 1]
                    d = tour[j % n]

                    old = dist[a, b] + dist[c, d]
                    new = dist[a, c] + dist[b, d]

                    if new < old - 1e-12:
                        tour[i:j] = tour[i:j][::-1]
                        improved = True
                        break

                if improved:
                    break

            if not improved:
                break

        return tour

    population = np.empty((pop_size, n), dtype=int)

    seed_count = min(pop_size, min(n, 8))
    starts = np.linspace(0, n - 1, seed_count, dtype=int)

    for i, start in enumerate(starts):
        population[i] = nearest_neighbor(int(start))

    for i in range(seed_count, pop_size):
        population[i] = rng.permutation(n)

    lengths = population_lengths(population)
    best_idx = int(np.argmin(lengths))
    best_tour = population[best_idx].copy()
    best_length = float(lengths[best_idx])

    for _ in range(generations):
        elite_idx = np.argsort(lengths)[:elite_count]

        new_population = np.empty_like(population)
        new_population[:elite_count] = population[elite_idx]

        pos = elite_count
        while pos < pop_size:
            p1 = population[tournament_select(lengths)]
            p2 = population[tournament_select(lengths)]

            if rng.random() < crossover_rate:
                child = order_crossover(p1, p2)
            else:
                child = p1.copy()

            if rng.random() < mutation_rate:
                swap_mutation(child)

            new_population[pos] = child
            pos += 1

        population = new_population
        lengths = population_lengths(population)

        gen_best_idx = int(np.argmin(lengths))
        gen_best_length = float(lengths[gen_best_idx])

        if gen_best_length < best_length:
            best_length = gen_best_length
            best_tour = population[gen_best_idx].copy()

    if n <= 200:
        best_tour = two_opt(best_tour, max_passes=40)
    elif n <= 600:
        best_tour = two_opt(best_tour, max_passes=10)
    else:
        best_tour = two_opt(best_tour, max_passes=3)

    best_length = tour_length(best_tour)

    return best_tour.tolist(), float(best_length)