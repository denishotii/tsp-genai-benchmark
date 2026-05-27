import numpy as np
import random


def solve(dist_matrix):
    dist_matrix = np.asarray(dist_matrix, dtype=float)
    n = dist_matrix.shape[0]

    if n <= 1:
        return list(range(n)), 0.0
    if n == 2:
        tour = [0, 1]
        length = float(dist_matrix[0, 1] + dist_matrix[1, 0])
        return tour, length

    # GA parameters
    pop_size = max(50, min(200, 10 * n))
    n_generations = max(200, min(2000, 50 * n))
    tournament_size = 5
    mutation_rate = 0.2
    elite_size = max(2, pop_size // 20)

    def tour_length(tour):
        idx = np.asarray(tour)
        return float(dist_matrix[idx, np.roll(idx, -1)].sum())

    def nearest_neighbor_tour(start):
        unvisited = set(range(n))
        unvisited.remove(start)
        tour = [start]
        current = start
        while unvisited:
            nxt = min(unvisited, key=lambda c: dist_matrix[current, c])
            tour.append(nxt)
            unvisited.remove(nxt)
            current = nxt
        return tour

    # Initialize population
    population = []
    # Seed with a few nearest neighbor tours
    n_nn = min(n, max(1, pop_size // 10))
    for i in range(n_nn):
        population.append(nearest_neighbor_tour(i % n))
    while len(population) < pop_size:
        perm = list(range(n))
        random.shuffle(perm)
        population.append(perm)

    fitnesses = [tour_length(t) for t in population]

    def tournament_select():
        contenders = random.sample(range(pop_size), tournament_size)
        best = min(contenders, key=lambda i: fitnesses[i])
        return population[best]

    def order_crossover(p1, p2):
        a, b = sorted(random.sample(range(n), 2))
        child = [-1] * n
        child[a:b+1] = p1[a:b+1]
        in_child = set(p1[a:b+1])
        pos = (b + 1) % n
        for i in range(n):
            gene = p2[(b + 1 + i) % n]
            if gene not in in_child:
                child[pos] = gene
                pos = (pos + 1) % n
        return child

    def swap_mutate(tour):
        t = tour[:]
        i, j = random.sample(range(n), 2)
        t[i], t[j] = t[j], t[i]
        return t

    best_idx = min(range(pop_size), key=lambda i: fitnesses[i])
    best_tour = population[best_idx][:]
    best_len = fitnesses[best_idx]

    for gen in range(n_generations):
        # Elitism
        sorted_idx = sorted(range(pop_size), key=lambda i: fitnesses[i])
        new_pop = [population[i][:] for i in sorted_idx[:elite_size]]

        while len(new_pop) < pop_size:
            p1 = tournament_select()
            p2 = tournament_select()
            child = order_crossover(p1, p2)
            if random.random() < mutation_rate:
                child = swap_mutate(child)
            new_pop.append(child)

        population = new_pop
        fitnesses = [tour_length(t) for t in population]

        cur_best_idx = min(range(pop_size), key=lambda i: fitnesses[i])
        if fitnesses[cur_best_idx] < best_len:
            best_len = fitnesses[cur_best_idx]
            best_tour = population[cur_best_idx][:]

    return best_tour, best_len