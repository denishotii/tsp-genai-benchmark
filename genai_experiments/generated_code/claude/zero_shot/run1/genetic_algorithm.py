import numpy as np
import random


def solve(dist_matrix):
    dist_matrix = np.asarray(dist_matrix, dtype=float)
    n = dist_matrix.shape[0]

    if n <= 1:
        return list(range(n)), 0.0
    if n == 2:
        return [0, 1], float(dist_matrix[0, 1] + dist_matrix[1, 0])

    # GA parameters
    pop_size = max(50, min(200, 10 * n))
    generations = max(100, min(1000, 30 * n))
    tournament_size = 5
    mutation_rate = 0.2
    elitism_count = max(2, pop_size // 20)

    def tour_length(tour):
        idx = np.asarray(tour)
        nxt = np.roll(idx, -1)
        return float(dist_matrix[idx, nxt].sum())

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
    # Seed with some nearest-neighbor tours
    seeds = min(n, max(1, pop_size // 10))
    for i in range(seeds):
        population.append(nearest_neighbor_tour(i % n))
    while len(population) < pop_size:
        t = list(range(n))
        random.shuffle(t)
        population.append(t)

    fitness = [tour_length(t) for t in population]

    def tournament_select():
        contenders = random.sample(range(pop_size), tournament_size)
        best = min(contenders, key=lambda i: fitness[i])
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
        if random.random() < mutation_rate:
            i, j = random.sample(range(n), 2)
            t[i], t[j] = t[j], t[i]
        return t

    best_idx = int(np.argmin(fitness))
    best_tour = population[best_idx][:]
    best_len = fitness[best_idx]

    for gen in range(generations):
        # Sort by fitness for elitism
        order = sorted(range(pop_size), key=lambda i: fitness[i])
        new_pop = [population[order[i]][:] for i in range(elitism_count)]

        while len(new_pop) < pop_size:
            p1 = tournament_select()
            p2 = tournament_select()
            child = order_crossover(p1, p2)
            child = swap_mutate(child)
            new_pop.append(child)

        population = new_pop
        fitness = [tour_length(t) for t in population]

        cur_best = int(np.argmin(fitness))
        if fitness[cur_best] < best_len:
            best_len = fitness[cur_best]
            best_tour = population[cur_best][:]

    return best_tour, best_len