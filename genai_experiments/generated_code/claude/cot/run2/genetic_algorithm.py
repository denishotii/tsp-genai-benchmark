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

    def tour_length(tour):
        idx = np.asarray(tour)
        nxt = np.roll(idx, -1)
        return float(dist_matrix[idx, nxt].sum())

    # GA parameters (scaled lightly with n)
    pop_size = max(50, min(200, 10 * n))
    generations = max(100, min(1000, 40 * n))
    tournament_k = 5
    mutation_rate = 0.2
    elite_count = max(2, pop_size // 20)
    crossover_rate = 0.9

    # --- Initialization: random perms + a nearest-neighbor seed ---
    population = []
    for _ in range(pop_size - 1):
        perm = list(range(n))
        random.shuffle(perm)
        population.append(perm)

    # Nearest neighbor heuristic seed
    start = random.randrange(n)
    visited = [False] * n
    nn_tour = [start]
    visited[start] = True
    current = start
    for _ in range(n - 1):
        dists = dist_matrix[current].copy()
        for j in range(n):
            if visited[j]:
                dists[j] = np.inf
        nxt = int(np.argmin(dists))
        nn_tour.append(nxt)
        visited[nxt] = True
        current = nxt
    population.append(nn_tour)

    fitness = np.array([tour_length(ind) for ind in population])

    def tournament_select():
        contenders = random.sample(range(pop_size), tournament_k)
        best = contenders[0]
        best_fit = fitness[best]
        for c in contenders[1:]:
            if fitness[c] < best_fit:
                best = c
                best_fit = fitness[c]
        return population[best]

    def order_crossover(p1, p2):
        a, b = sorted(random.sample(range(n), 2))
        child = [-1] * n
        # copy slice from p1
        slice_set = set()
        for i in range(a, b + 1):
            child[i] = p1[i]
            slice_set.add(p1[i])
        # fill remaining from p2 in order, starting after b
        fill_pos = (b + 1) % n
        p2_pos = (b + 1) % n
        while -1 in child:
            gene = p2[p2_pos]
            if gene not in slice_set:
                child[fill_pos] = gene
                fill_pos = (fill_pos + 1) % n
            p2_pos = (p2_pos + 1) % n
        return child

    def swap_mutate(ind):
        i, j = random.sample(range(n), 2)
        ind[i], ind[j] = ind[j], ind[i]
        return ind

    best_idx = int(np.argmin(fitness))
    best_tour = list(population[best_idx])
    best_len = float(fitness[best_idx])

    for gen in range(generations):
        # Elitism: keep top elite_count
        elite_indices = np.argsort(fitness)[:elite_count]
        new_population = [list(population[i]) for i in elite_indices]

        while len(new_population) < pop_size:
            p1 = tournament_select()
            p2 = tournament_select()
            if random.random() < crossover_rate:
                child = order_crossover(p1, p2)
            else:
                child = list(p1)
            if random.random() < mutation_rate:
                child = swap_mutate(child)
            new_population.append(child)

        population = new_population
        fitness = np.array([tour_length(ind) for ind in population])

        cur_best_idx = int(np.argmin(fitness))
        if fitness[cur_best_idx] < best_len:
            best_len = float(fitness[cur_best_idx])
            best_tour = list(population[cur_best_idx])

    return best_tour, best_len