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
        nxt = np.roll(idx, -1)
        return float(dist[idx, nxt].sum())

    # GA parameters
    pop_size = 100
    n_generations = 500
    tournament_k = 5
    crossover_rate = 0.9
    mutation_rate = 0.2
    elite_size = 2

    # Initialize population
    population = []
    for _ in range(pop_size):
        ind = list(range(n))
        random.shuffle(ind)
        population.append(ind)

    # Add a nearest-neighbor seed
    def nearest_neighbor(start):
        unvisited = set(range(n))
        unvisited.remove(start)
        tour = [start]
        cur = start
        while unvisited:
            nxt = min(unvisited, key=lambda c: dist[cur, c])
            tour.append(nxt)
            unvisited.remove(nxt)
            cur = nxt
        return tour

    population[0] = nearest_neighbor(0)

    fitness = [tour_length(ind) for ind in population]

    def tournament_select():
        best_i = random.randrange(pop_size)
        best_f = fitness[best_i]
        for _ in range(tournament_k - 1):
            i = random.randrange(pop_size)
            if fitness[i] < best_f:
                best_i = i
                best_f = fitness[i]
        return population[best_i]

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

    def swap_mutation(ind):
        i, j = random.sample(range(n), 2)
        ind[i], ind[j] = ind[j], ind[i]

    best_idx = int(np.argmin(fitness))
    best_tour = population[best_idx][:]
    best_len = fitness[best_idx]

    for gen in range(n_generations):
        # Elitism
        order = sorted(range(pop_size), key=lambda i: fitness[i])
        new_pop = [population[order[i]][:] for i in range(elite_size)]

        while len(new_pop) < pop_size:
            p1 = tournament_select()
            p2 = tournament_select()
            if random.random() < crossover_rate:
                child = order_crossover(p1, p2)
            else:
                child = p1[:]
            if random.random() < mutation_rate:
                swap_mutation(child)
            new_pop.append(child)

        population = new_pop
        fitness = [tour_length(ind) for ind in population]

        cur_best = int(np.argmin(fitness))
        if fitness[cur_best] < best_len:
            best_len = fitness[cur_best]
            best_tour = population[cur_best][:]

    return best_tour, best_len