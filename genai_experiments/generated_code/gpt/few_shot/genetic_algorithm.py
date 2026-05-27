import numpy as np

def solve(dist_matrix):
    """Solve a symmetric TSP instance using a Genetic Algorithm."""
    dist_matrix = np.asarray(dist_matrix, dtype=float)
    n = dist_matrix.shape[0]

    if n == 0:
        return [], 0.0
    if n == 1:
        return [0], 0.0

    rng = np.random.default_rng(12345)

    pop_size = min(max(60, 6 * n), 250)
    generations = min(max(250, 20 * n), 1000)
    elite_size = max(1, pop_size // 20)
    tournament_size = 3
    crossover_rate = 0.9
    mutation_rate = 0.2

    def tour_length(tour):
        return float(dist_matrix[tour, np.roll(tour, -1)].sum())

    def population_lengths(pop):
        return dist_matrix[pop, np.roll(pop, -1, axis=1)].sum(axis=1)

    def nearest_neighbor(start):
        tour = np.empty(n, dtype=int)
        unvisited = np.ones(n, dtype=bool)
        current = start

        for i in range(n):
            tour[i] = current
            unvisited[current] = False

            if i < n - 1:
                distances = dist_matrix[current].copy()
                distances[~unvisited] = np.inf
                current = int(np.argmin(distances))

        return tour

    def make_initial_population():
        pop = np.empty((pop_size, n), dtype=int)
        count = 0

        greedy_count = min(n, max(1, pop_size // 5))
        starts = np.linspace(0, n - 1, greedy_count, dtype=int)

        for start in starts:
            pop[count] = nearest_neighbor(int(start))
            count += 1

        while count < pop_size:
            pop[count] = rng.permutation(n)
            count += 1

        return pop

    def tournament_select(pop, lengths):
        candidates = rng.integers(0, pop_size, size=tournament_size)
        best = candidates[np.argmin(lengths[candidates])]
        return pop[best]

    def order_crossover(parent1, parent2):
        child = np.full(n, -1, dtype=int)

        a, b = sorted(rng.choice(n, size=2, replace=False))
        b += 1

        child[a:b] = parent1[a:b]

        used = np.zeros(n, dtype=bool)
        used[child[a:b]] = True

        pos = b % n
        for city in np.concatenate((parent2[b:], parent2[:b])):
            if not used[city]:
                child[pos] = city
                used[city] = True
                pos = (pos + 1) % n

        return child

    def swap_mutation(tour):
        if rng.random() < mutation_rate:
            i, j = rng.choice(n, size=2, replace=False)
            tour[i], tour[j] = tour[j], tour[i]

    population = make_initial_population()
    lengths = population_lengths(population)

    best_idx = int(np.argmin(lengths))
    best_tour = population[best_idx].copy()
    best_length = float(lengths[best_idx])

    for _ in range(generations):
        order = np.argsort(lengths)

        new_population = np.empty_like(population)
        new_population[:elite_size] = population[order[:elite_size]]

        fill = elite_size
        while fill < pop_size:
            parent1 = tournament_select(population, lengths)
            parent2 = tournament_select(population, lengths)

            if rng.random() < crossover_rate:
                child1 = order_crossover(parent1, parent2)
                child2 = order_crossover(parent2, parent1)
            else:
                child1 = parent1.copy()
                child2 = parent2.copy()

            swap_mutation(child1)
            swap_mutation(child2)

            new_population[fill] = child1
            fill += 1

            if fill < pop_size:
                new_population[fill] = child2
                fill += 1

        population = new_population
        lengths = population_lengths(population)

        current_idx = int(np.argmin(lengths))
        current_length = float(lengths[current_idx])

        if current_length < best_length:
            best_length = current_length
            best_tour = population[current_idx].copy()

    return best_tour.tolist(), tour_length(best_tour)