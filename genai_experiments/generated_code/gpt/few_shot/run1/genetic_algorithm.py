import numpy as np

def solve(dist_matrix):
    """Solve a symmetric TSP approximately using a Genetic Algorithm."""
    D = np.asarray(dist_matrix, dtype=float)
    n = D.shape[0]

    if n == 0:
        return [], 0.0
    if n == 1:
        return [0], 0.0

    rng = np.random.default_rng()

    def tour_length(tour):
        return float(D[tour, np.roll(tour, -1)].sum())

    def population_lengths(pop):
        return D[pop, np.roll(pop, -1, axis=1)].sum(axis=1)

    def nearest_neighbor(start):
        tour = np.empty(n, dtype=int)
        unvisited = np.ones(n, dtype=bool)
        current = start

        for k in range(n):
            tour[k] = current
            unvisited[current] = False
            if k < n - 1:
                remaining = np.flatnonzero(unvisited)
                current = remaining[np.argmin(D[current, remaining])]

        return tour

    def tournament_select(pop, lengths, tournament_size=3):
        candidates = rng.integers(0, len(pop), size=tournament_size)
        return pop[candidates[np.argmin(lengths[candidates])]]

    def order_crossover(parent1, parent2):
        """Order Crossover, OX."""
        a, b = np.sort(rng.choice(n, size=2, replace=False))

        child = np.full(n, -1, dtype=int)
        child[a:b + 1] = parent1[a:b + 1]

        used = np.zeros(n, dtype=bool)
        used[child[a:b + 1]] = True

        pos = (b + 1) % n
        for k in range(n):
            city = parent2[(b + 1 + k) % n]
            if not used[city]:
                child[pos] = city
                pos = (pos + 1) % n

        return child

    def swap_mutation(tour):
        i, j = rng.choice(n, size=2, replace=False)
        tour[i], tour[j] = tour[j], tour[i]

    def two_opt_polish(tour):
        """Small final 2-opt improvement pass."""
        if n < 4 or n > 250:
            return tour

        tour = tour.copy()
        max_moves = 200
        moves = 0

        while moves < max_moves:
            improved = False

            for i in range(1, n - 1):
                a, b = tour[i - 1], tour[i]

                for j in range(i + 1, n):
                    c, d = tour[j], tour[(j + 1) % n]

                    old = D[a, b] + D[c, d]
                    new = D[a, c] + D[b, d]

                    if new + 1e-12 < old:
                        tour[i:j + 1] = tour[i:j + 1][::-1]
                        moves += 1
                        improved = True
                        break

                if improved:
                    break

            if not improved:
                break

        return tour

    pop_size = int(min(300, max(60, 6 * n)))
    generations = int(min(1000, max(80, 20000 // max(n, 1))))
    elite_count = max(1, pop_size // 20)

    crossover_rate = 0.90
    mutation_rate = 0.20

    population = np.empty((pop_size, n), dtype=int)

    # Seed part of the population with nearest-neighbor tours.
    count = 0
    seed_count = min(n, pop_size // 5, 30)
    starts = rng.choice(n, size=seed_count, replace=False)

    for s in starts:
        population[count] = nearest_neighbor(int(s))
        count += 1

    # Fill the rest randomly.
    while count < pop_size:
        population[count] = rng.permutation(n)
        count += 1

    best_tour = None
    best_length = float("inf")

    for _ in range(generations):
        lengths = population_lengths(population)
        order = np.argsort(lengths)

        if lengths[order[0]] < best_length:
            best_length = float(lengths[order[0]])
            best_tour = population[order[0]].copy()

        next_population = np.empty_like(population)

        # Elitism: copy the best tours directly.
        elite_indices = order[:elite_count]
        next_population[:elite_count] = population[elite_indices]

        # Create the rest of the next generation.
        for i in range(elite_count, pop_size):
            parent1 = tournament_select(population, lengths)
            parent2 = tournament_select(population, lengths)

            if rng.random() < crossover_rate:
                child = order_crossover(parent1, parent2)
            else:
                child = parent1.copy()

            if rng.random() < mutation_rate:
                swap_mutation(child)

            next_population[i] = child

        population = next_population

    # Evaluate final population.
    lengths = population_lengths(population)
    idx = int(np.argmin(lengths))
    if lengths[idx] < best_length:
        best_length = float(lengths[idx])
        best_tour = population[idx].copy()

    best_tour = two_opt_polish(best_tour)
    best_length = tour_length(best_tour)

    return best_tour.tolist(), float(best_length)