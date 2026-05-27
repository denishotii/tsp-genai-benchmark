import numpy as np
import random


def solve(dist_matrix):
    dist_matrix = np.asarray(dist_matrix, dtype=float)
    n = dist_matrix.shape[0]

    if n == 0:
        return [], 0.0
    if n == 1:
        return [0], 0.0
    if n == 2:
        return [0, 1], float(dist_matrix[0, 1] + dist_matrix[1, 0])

    # GA parameters
    pop_size = max(50, min(200, 10 * n))
    n_generations = max(200, min(1000, 50 * n))
    tournament_k = 5
    mutation_rate = 0.02
    elitism = max(2, pop_size // 20)
    crossover_rate = 0.9

    def tour_length(tour):
        idx = np.asarray(tour)
        nxt = np.roll(idx, -1)
        return float(dist_matrix[idx, nxt].sum())

    # Initialize population
    population = []
    base = list(range(n))
    for _ in range(pop_size):
        perm = base[:]
        random.shuffle(perm)
        population.append(perm)

    # Add a greedy nearest-neighbor seed
    def greedy(start):
        unvisited = set(range(n))
        unvisited.remove(start)
        tour = [start]
        cur = start
        while unvisited:
            nxt = min(unvisited, key=lambda x: dist_matrix[cur, x])
            tour.append(nxt)
            unvisited.remove(nxt)
            cur = nxt
        return tour

    population[0] = greedy(0)

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
        for i in range(n):
            if random.random() < mutation_rate:
                j = random.randrange(n)
                t[i], t[j] = t[j], t[i]
        return t

    best_idx = int(np.argmin(fitness))
    best_tour = population[best_idx][:]
    best_len = fitness[best_idx]

    for gen in range(n_generations):
        # Sort by fitness
        order = sorted(range(pop_size), key=lambda i: fitness[i])
        new_pop = [population[order[i]][:] for i in range(elitism)]
        new_fit = [fitness[order[i]] for i in range(elitism)]

        while len(new_pop) < pop_size:
            p1 = tournament_select()
            p2 = tournament_select()
            if random.random() < crossover_rate:
                child = order_crossover(p1, p2)
            else:
                child = p1[:]
            child = swap_mutate(child)
            new_pop.append(child)
            new_fit.append(tour_length(child))

        population = new_pop
        fitness = new_fit

        cur_best_idx = int(np.argmin(fitness))
        if fitness[cur_best_idx] < best_len:
            best_len = fitness[cur_best_idx]
            best_tour = population[cur_best_idx][:]

    return best_tour, best_len