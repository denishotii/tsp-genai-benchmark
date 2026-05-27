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
    pop_size = min(200, max(50, 10 * n))
    generations = min(1000, max(200, 30 * n))
    tournament_k = 5
    mutation_rate = 0.2
    elite_size = max(2, pop_size // 20)
    crossover_rate = 0.9

    def tour_length(tour):
        idx = np.asarray(tour)
        return float(dist_matrix[idx, np.roll(idx, -1)].sum())

    def nearest_neighbor_tour(start):
        unvisited = set(range(n))
        unvisited.remove(start)
        tour = [start]
        current = start
        while unvisited:
            nxt = min(unvisited, key=lambda x: dist_matrix[current, x])
            tour.append(nxt)
            unvisited.remove(nxt)
            current = nxt
        return tour

    # Initialize population
    population = []
    # seed a few with nearest neighbor
    for _ in range(min(5, pop_size)):
        s = random.randrange(n)
        population.append(nearest_neighbor_tour(s))
    while len(population) < pop_size:
        perm = list(range(n))
        random.shuffle(perm)
        population.append(perm)

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
        in_child = set(child[a:b+1])
        pos = (b + 1) % n
        for i in range(n):
            gene = p2[(b + 1 + i) % n]
            if gene not in in_child:
                child[pos] = gene
                pos = (pos + 1) % n
        return child

    def swap_mutation(tour):
        i, j = random.sample(range(n), 2)
        tour[i], tour[j] = tour[j], tour[i]
        return tour

    best_idx = int(np.argmin(fitness))
    best_tour = population[best_idx][:]
    best_length = fitness[best_idx]

    for _ in range(generations):
        # Sort by fitness
        order = sorted(range(pop_size), key=lambda i: fitness[i])
        new_pop = [population[order[i]][:] for i in range(elite_size)]

        while len(new_pop) < pop_size:
            p1 = tournament_select()
            if random.random() < crossover_rate:
                p2 = tournament_select()
                child = order_crossover(p1, p2)
            else:
                child = p1[:]
            if random.random() < mutation_rate:
                child = swap_mutation(child)
            new_pop.append(child)

        population = new_pop
        fitness = [tour_length(t) for t in population]

        gen_best_idx = int(np.argmin(fitness))
        if fitness[gen_best_idx] < best_length:
            best_length = fitness[gen_best_idx]
            best_tour = population[gen_best_idx][:]

    return best_tour, best_length