import numpy as np
import random


def solve(dist_matrix):
    dist = np.asarray(dist_matrix, dtype=float)
    n = dist.shape[0]

    if n <= 1:
        return list(range(n)), 0.0
    if n == 2:
        return [0, 1], float(dist[0, 1] + dist[1, 0])

    def tour_length(tour):
        idx = np.asarray(tour)
        return float(dist[idx, np.roll(idx, -1)].sum())

    # GA parameters
    pop_size = max(50, min(200, 10 * n))
    generations = max(100, min(1000, 50 * n))
    tournament_k = 5
    mutation_rate = 0.2
    elite_size = max(2, pop_size // 20)
    crossover_rate = 0.9

    # Initialize population
    def nearest_neighbor_tour(start):
        unvisited = set(range(n))
        unvisited.remove(start)
        tour = [start]
        current = start
        while unvisited:
            nxt = min(unvisited, key=lambda x: dist[current, x])
            tour.append(nxt)
            unvisited.remove(nxt)
            current = nxt
        return tour

    population = []
    # Seed with a few nearest-neighbor tours
    nn_count = min(n, pop_size // 10)
    for i in range(nn_count):
        population.append(nearest_neighbor_tour(i % n))
    while len(population) < pop_size:
        t = list(range(n))
        random.shuffle(t)
        population.append(t)

    fitness = [tour_length(t) for t in population]

    def tournament_select():
        best_idx = random.randrange(pop_size)
        best_fit = fitness[best_idx]
        for _ in range(tournament_k - 1):
            i = random.randrange(pop_size)
            if fitness[i] < best_fit:
                best_fit = fitness[i]
                best_idx = i
        return population[best_idx]

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
            num_swaps = random.randint(1, max(1, n // 20))
            for _ in range(num_swaps):
                i, j = random.sample(range(n), 2)
                t[i], t[j] = t[j], t[i]
        return t

    best_idx = int(np.argmin(fitness))
    best_tour = population[best_idx][:]
    best_length = fitness[best_idx]

    for gen in range(generations):
        # Elitism: keep best individuals
        sorted_idx = np.argsort(fitness)
        new_pop = [population[i][:] for i in sorted_idx[:elite_size]]

        while len(new_pop) < pop_size:
            p1 = tournament_select()
            p2 = tournament_select()
            if random.random() < crossover_rate:
                child = order_crossover(p1, p2)
            else:
                child = p1[:]
            child = swap_mutate(child)
            new_pop.append(child)

        population = new_pop
        fitness = [tour_length(t) for t in population]

        cur_best = int(np.argmin(fitness))
        if fitness[cur_best] < best_length:
            best_length = fitness[cur_best]
            best_tour = population[cur_best][:]

    return best_tour, best_length