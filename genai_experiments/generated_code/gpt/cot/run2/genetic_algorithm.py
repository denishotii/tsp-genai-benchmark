import numpy as np
import math
import random


"""
Step-by-step reasoning:

1. Problem restatement and key components
   We need to solve the symmetric Traveling Salesman Problem: given an n x n
   distance matrix, find a closed tour visiting every city exactly once and
   returning to the start, minimizing total length.

   This implementation uses a Genetic Algorithm with:
   - Permutation encoding: each individual is an ordering of city indices.
   - Tournament selection: choose the best individual among a random subset.
   - Order Crossover (OX): preserves a slice from one parent and fills the rest
     in the order they appear in the other parent.
   - Swap mutation: randomly swap two cities in a tour.
   - Elitism: copy the best individuals unchanged into the next generation.

2. Data structures and main loop
   - Population: NumPy array of shape (population_size, n), where each row is a
     permutation of cities.
   - Fitness/objective: tour length, minimized.
   - Main loop:
       a. Evaluate population.
       b. Keep elite individuals.
       c. Fill remaining population using tournament selection, OX crossover,
          and swap mutation.
       d. Track the best tour seen.
   - The final returned tour is a Python list and length is a float.

3. Edge cases
   - n == 0: return empty tour and length 0.0.
   - n == 1: return [0] and length 0.0.
   - n == 2: the only closed tour has length 2 * dist[0][1].
   - Ties are handled naturally by taking the first minimum encountered.
"""


def solve(dist_matrix):
    dist_matrix = np.asarray(dist_matrix, dtype=float)

    if dist_matrix.ndim != 2 or dist_matrix.shape[0] != dist_matrix.shape[1]:
        raise ValueError("dist_matrix must be a square 2D NumPy array")

    n = dist_matrix.shape[0]

    if n == 0:
        return [], 0.0

    if n == 1:
        return [0], 0.0

    if n == 2:
        return [0, 1], float(dist_matrix[0, 1] + dist_matrix[1, 0])

    # Fixed seed for reproducibility.
    rng = np.random.default_rng(12345)
    py_rng = random.Random(12345)

    # GA parameters chosen adaptively.
    population_size = min(max(80, 8 * n), 300)
    generations = min(max(300, 25 * n), 1500)
    tournament_size = 4
    crossover_rate = 0.90
    mutation_rate = 0.20
    elite_count = max(1, population_size // 20)

    def tour_length(tour):
        """Length of one closed tour."""
        return float(dist_matrix[tour, np.roll(tour, -1)].sum())

    def population_lengths(pop):
        """Vectorized closed-tour lengths for the whole population."""
        return dist_matrix[pop, np.roll(pop, -1, axis=1)].sum(axis=1)

    def nearest_neighbor_tour(start):
        """Construct a simple greedy tour to seed the population."""
        unvisited = set(range(n))
        tour = [start]
        unvisited.remove(start)

        current = start
        while unvisited:
            nxt = min(unvisited, key=lambda city: dist_matrix[current, city])
            tour.append(nxt)
            unvisited.remove(nxt)
            current = nxt

        return np.array(tour, dtype=int)

    def tournament_select(pop, lengths):
        """Return one selected parent using tournament selection."""
        indices = rng.integers(0, len(pop), size=tournament_size)
        best_idx = indices[np.argmin(lengths[indices])]
        return pop[best_idx]

    def order_crossover(parent1, parent2):
        """
        Order Crossover, OX.

        A segment from parent1 is copied into the child. Remaining positions are
        filled from parent2 in cyclic order, skipping already-used cities.
        """
        child = np.full(n, -1, dtype=int)

        a, b = sorted(rng.choice(n, size=2, replace=False))
        child[a:b + 1] = parent1[a:b + 1]

        used = np.zeros(n, dtype=bool)
        used[child[a:b + 1]] = True

        fill_pos = (b + 1) % n
        scan_pos = (b + 1) % n

        while np.any(child == -1):
            city = parent2[scan_pos]
            if not used[city]:
                child[fill_pos] = city
                used[city] = True
                fill_pos = (fill_pos + 1) % n
            scan_pos = (scan_pos + 1) % n

        return child

    def swap_mutation(tour):
        """Swap two positions in the permutation."""
        if rng.random() < mutation_rate:
            i, j = rng.choice(n, size=2, replace=False)
            tour[i], tour[j] = tour[j], tour[i]

    def canonicalize(tour):
        """Rotate tour so city 0 appears first; length is unchanged."""
        idx = tour.index(0)
        return tour[idx:] + tour[:idx]

    # Initialize population.
    population = np.empty((population_size, n), dtype=int)

    # Add a few nearest-neighbor heuristic seeds.
    seed_count = min(n, max(1, population_size // 10))
    starts = rng.choice(n, size=seed_count, replace=False)
    for i, start in enumerate(starts):
        population[i] = nearest_neighbor_tour(int(start))

    # Fill the rest with random permutations.
    base = np.arange(n)
    for i in range(seed_count, population_size):
        population[i] = rng.permutation(base)

    lengths = population_lengths(population)
    best_idx = int(np.argmin(lengths))
    best_tour = population[best_idx].copy()
    best_length = float(lengths[best_idx])

    # Main GA loop.
    for _ in range(generations):
        lengths = population_lengths(population)

        current_best_idx = int(np.argmin(lengths))
        current_best_length = float(lengths[current_best_idx])

        if current_best_length < best_length:
            best_length = current_best_length
            best_tour = population[current_best_idx].copy()

        new_population = np.empty_like(population)

        # Elitism: copy the best individuals unchanged.
        elite_indices = np.argsort(lengths)[:elite_count]
        new_population[:elite_count] = population[elite_indices]

        # Generate the rest of the population.
        pos = elite_count
        while pos < population_size:
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

            new_population[pos] = child1
            pos += 1

            if pos < population_size:
                new_population[pos] = child2
                pos += 1

        population = new_population

    # Final evaluation after the last generation.
    lengths = population_lengths(population)
    final_best_idx = int(np.argmin(lengths))
    final_best_length = float(lengths[final_best_idx])

    if final_best_length < best_length:
        best_length = final_best_length
        best_tour = population[final_best_idx].copy()

    tour = best_tour.tolist()
    tour = canonicalize(tour)
    length = tour_length(np.array(tour, dtype=int))

    return tour, float(length)