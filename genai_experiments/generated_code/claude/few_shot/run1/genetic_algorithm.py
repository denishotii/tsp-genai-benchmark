import numpy as np
import random

def solve(dist_matrix):
    dist_matrix = np.asarray(dist_matrix, dtype=float)
    n = dist_matrix.shape[0]

    if n <= 1:
        return list(range(n)), 0.0
    if n == 2:
        return [0, 1], float(dist_matrix[0, 1] + dist_matrix[1, 0])

    def tour_length(tour):
        idx = np.asarray(tour)
        return float(dist_matrix[idx, np.roll(idx, -1)].sum())

    # GA parameters
    pop_size = min(200, max(50, 10 * n))
    generations = min(1000, max(200, 30 * n))
    tournament_k = 5
    mutation_rate = 0.2
    elite_count = max(2, pop_size // 20)

    # Initialize population: mostly random, some greedy
    def greedy_tour(start):
        unvisited = set(range(n))
        unvisited.remove(start)
        tour = [start]
        cur = start
        while unvisited:
            nxt = min(unvisited, key=lambda j: dist_matrix[cur, j])
            tour.append(nxt)
            unvisited.remove(nxt)
            cur = nxt
        return tour

    population = []
    num_greedy = min(n, max(1, pop_size // 10))
    for i in range(num_greedy):
        population.append(greedy_tour(i % n))
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
                best_idx = i
                best_fit = fitness[i]
        return population[best_idx]

    def order_crossover(p1, p2):
        a, b = sorted(random.sample(range(n), 2))
        child = [-1] * n
        segment = p1[a:b+1]
        child[a:b+1] = segment
        seg_set = set(segment)
        fill = [c for c in p2 if c not in seg_set]
        pos = 0
        for i in range(n):
            if child[i] == -1:
                child[i] = fill[pos]
                pos += 1
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
    best_len = fitness[best_idx]

    stagnation = 0
    last_best = best_len

    for gen in range(generations):
        # Sort by fitness for elitism
        order = sorted(range(pop_size), key=lambda i: fitness[i])
        new_pop = [population[order[i]][:] for i in range(elite_count)]

        while len(new_pop) < pop_size:
            p1 = tournament_select()
            p2 = tournament_select()
            child = order_crossover(p1, p2)
            child = swap_mutate(child)
            new_pop.append(child)

        population = new_pop
        fitness = [tour_length(t) for t in population]

        cur_best_idx = int(np.argmin(fitness))
        if fitness[cur_best_idx] < best_len:
            best_len = fitness[cur_best_idx]
            best_tour = population[cur_best_idx][:]

        if best_len < last_best - 1e-9:
            last_best = best_len
            stagnation = 0
        else:
            stagnation += 1
            if stagnation > 50:
                break

    # 2-opt local refinement on best tour
    def two_opt(tour):
        improved = True
        t = tour[:]
        while improved:
            improved = False
            for i in range(n - 1):
                for j in range(i + 2, n):
                    if i == 0 and j == n - 1:
                        continue
                    a, b = t[i], t[i+1]
                    c, d = t[j], t[(j+1) % n]
                    delta = (dist_matrix[a, c] + dist_matrix[b, d]
                             - dist_matrix[a, b] - dist_matrix[c, d])
                    if delta < -1e-12:
                        t[i+1:j+1] = t[i+1:j+1][::-1]
                        improved = True
        return t

    if n <= 200:
        best_tour = two_opt(best_tour)
        best_len = tour_length(best_tour)

    return list(best_tour), float(best_len)