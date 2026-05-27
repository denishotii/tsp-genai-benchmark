import numpy as np


def solve(dist_matrix):
    """
    Genetic Algorithm for the symmetric Traveling Salesman Problem.

    Step-by-step design notes:
    1. Problem/components:
       - A candidate solution is a permutation of city indices 0..n-1.
       - Fitness is based on closed-tour length; shorter is better.
       - The GA uses tournament selection, Order Crossover (OX), swap mutation,
         and elitism to preserve the best individuals.

    2. Data structures/main loop:
       - Population: a NumPy array of shape (pop_size, n), where each row is a tour.
       - Lengths: a NumPy array of shape (pop_size,) containing each tour's cost.
       - Each generation:
         a. Copy the best elite_count tours directly into the next population.
         b. Fill the rest using tournament-selected parents.
         c. Apply OX with high probability.
         d. Apply swap mutation with fixed probability.
         e. Recompute lengths and update the global best tour.

    3. Edge cases:
       - n == 0: return an empty tour with length 0.
       - n == 1: the only tour has length 0.
       - n == 2: the closed tour length is d(0,1) + d(1,0).
       - Ties are resolved naturally by NumPy's argmin, which picks one minimum.
    """

    D = np.asarray(dist_matrix, dtype=float)

    if D.ndim != 2 or D.shape[0] != D.shape[1]:
        raise ValueError("dist_matrix must be a square NumPy array")

    n = D.shape[0]

    if n == 0:
        return [], 0.0
    if n == 1:
        return [0], 0.0
    if n == 2:
        return [0, 1], float(D[0, 1] + D[1, 0])

    # Fixed seed for reproducibility.
    rng = np.random.default_rng(12345)

    def tour_length(tour):
        return float(D[tour, np.roll(tour, -1)].sum())

    def population_lengths(population):
        return D[population, np.roll(population, -1, axis=1)].sum(axis=1)

    def nearest_neighbor_tour(start):
        tour = np.empty(n, dtype=int)
        unvisited = np.ones(n, dtype=bool)

        current = start
        for k in range(n):
            tour[k] = current
            unvisited[current] = False

            if k < n - 1:
                candidates = np.flatnonzero(unvisited)
                current = candidates[np.argmin(D[current, candidates])]

        return tour

    def tournament_select(population, lengths, tournament_size):
        candidates = rng.integers(0, len(population), size=tournament_size)
        winner = candidates[np.argmin(lengths[candidates])]
        return population[winner]

    def order_crossover(parent1, parent2):
        child = np.full(n, -1, dtype=int)

        a = int(rng.integers(0, n))
        b = int(rng.integers(0, n - 1))
        if b >= a:
            b += 1
        if a > b:
            a, b = b, a

        child[a:b + 1] = parent1[a:b + 1]

        used = np.zeros(n, dtype=bool)
        used[child[a:b + 1]] = True

        fill_pos = (b + 1) % n

        for offset in range(n):
            gene = parent2[(b + 1 + offset) % n]
            if not used[gene]:
                child[fill_pos] = gene
                used[gene] = True
                fill_pos = (fill_pos + 1) % n

        return child

    def swap_mutate(tour):
        i = int(rng.integers(0, n))
        j = int(rng.integers(0, n - 1))
        if j >= i:
            j += 1
        tour[i], tour[j] = tour[j], tour[i]

    # GA hyperparameters chosen adaptively but kept bounded.
    pop_size = int(min(250, max(50, 5 * n)))
    generations = int(min(1000, max(200, 25 * n)))
    elite_count = max(1, pop_size // 20)
    tournament_size = min(5, pop_size)
    crossover_rate = 0.90
    mutation_rate = 0.20
    patience = max(80, generations // 3)

    # Initial population: a few heuristic tours plus random permutations.
    population = np.empty((pop_size, n), dtype=int)

    pos = 0
    population[pos] = np.arange(n)
    pos += 1

    greedy_count = min(n, 10, max(1, pop_size // 10))
    starts = rng.choice(n, size=greedy_count, replace=False)

    for s in starts:
        if pos >= pop_size:
            break
        population[pos] = nearest_neighbor_tour(int(s))
        pos += 1

    while pos < pop_size:
        population[pos] = rng.permutation(n)
        pos += 1

    lengths = population_lengths(population)

    best_idx = int(np.argmin(lengths))
    best_tour = population[best_idx].copy()
    best_length = float(lengths[best_idx])

    stagnant = 0

    for _ in range(generations):
        order = np.argsort(lengths)

        new_population = np.empty_like(population)

        # Elitism: preserve the best tours unchanged.
        new_population[:elite_count] = population[order[:elite_count]]

        fill = elite_count

        while fill < pop_size:
            p1 = tournament_select(population, lengths, tournament_size)
            p2 = tournament_select(population, lengths, tournament_size)

            if rng.random() < crossover_rate:
                child1 = order_crossover(p1, p2)
            else:
                child1 = p1.copy()

            if rng.random() < mutation_rate:
                swap_mutate(child1)

            new_population[fill] = child1
            fill += 1

            if fill >= pop_size:
                break

            if rng.random() < crossover_rate:
                child2 = order_crossover(p2, p1)
            else:
                child2 = p2.copy()

            if rng.random() < mutation_rate:
                swap_mutate(child2)

            new_population[fill] = child2
            fill += 1

        population = new_population
        lengths = population_lengths(population)

        gen_best_idx = int(np.argmin(lengths))
        gen_best_length = float(lengths[gen_best_idx])

        if gen_best_length < best_length - 1e-12:
            best_length = gen_best_length
            best_tour = population[gen_best_idx].copy()
            stagnant = 0
        else:
            stagnant += 1

        if stagnant >= patience:
            break

    # Recompute the returned length directly from the returned tour.
    final_length = tour_length(best_tour)

    return best_tour.tolist(), final_length