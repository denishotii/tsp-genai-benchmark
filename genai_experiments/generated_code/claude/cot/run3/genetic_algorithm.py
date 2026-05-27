import numpy as np
import random

def solve(dist_matrix):
    dist = np.asarray(dist_matrix, dtype=float)
    n = dist.shape[0]
    
    if n == 0:
        return [], 0.0
    if n == 1:
        return [0], 0.0
    if n == 2:
        return [0, 1], float(dist[0, 1] + dist[1, 0])
    
    def tour_length(tour):
        return float(dist[tour, np.roll(tour, -1)].sum())
    
    def nearest_neighbor(start):
        unvisited = set(range(n))
        unvisited.remove(start)
        tour = [start]
        current = start
        while unvisited:
            nxt = min(unvisited, key=lambda x: dist[current, x])
            tour.append(nxt)
            unvisited.remove(nxt)
            current = nxt
        return np.array(tour, dtype=np.int64)
    
    # GA params
    pop_size = max(50, min(200, 10 * n))
    elite_size = max(2, pop_size // 20)
    tournament_k = 5
    mutation_rate = 0.2
    max_generations = max(100, min(1000, 30 * n))
    stagnation_limit = max(50, max_generations // 4)
    
    # Initialize population
    population = []
    num_greedy = min(n, max(1, pop_size // 10))
    greedy_starts = random.sample(range(n), num_greedy)
    for s in greedy_starts:
        population.append(nearest_neighbor(s))
    while len(population) < pop_size:
        perm = np.random.permutation(n).astype(np.int64)
        population.append(perm)
    
    lengths = np.array([tour_length(t) for t in population])
    
    best_idx = int(np.argmin(lengths))
    best_tour = population[best_idx].copy()
    best_length = float(lengths[best_idx])
    
    def tournament_select():
        idxs = random.sample(range(pop_size), tournament_k)
        best = idxs[0]
        best_len = lengths[best]
        for i in idxs[1:]:
            if lengths[i] < best_len:
                best = i
                best_len = lengths[i]
        return population[best]
    
    def order_crossover(p1, p2):
        a, b = sorted(random.sample(range(n), 2))
        child = -np.ones(n, dtype=np.int64)
        child[a:b+1] = p1[a:b+1]
        in_child = np.zeros(n, dtype=bool)
        in_child[p1[a:b+1]] = True
        pos = (b + 1) % n
        for i in range(n):
            gene = p2[(b + 1 + i) % n]
            if not in_child[gene]:
                child[pos] = gene
                pos = (pos + 1) % n
        return child
    
    def swap_mutation(tour):
        t = tour.copy()
        i, j = random.sample(range(n), 2)
        t[i], t[j] = t[j], t[i]
        return t
    
    stagnation = 0
    for gen in range(max_generations):
        # Sort by fitness
        order = np.argsort(lengths)
        population = [population[i] for i in order]
        lengths = lengths[order]
        
        new_population = [population[i].copy() for i in range(elite_size)]
        new_lengths = list(lengths[:elite_size])
        
        while len(new_population) < pop_size:
            p1 = tournament_select()
            p2 = tournament_select()
            child = order_crossover(p1, p2)
            if random.random() < mutation_rate:
                child = swap_mutation(child)
            new_population.append(child)
            new_lengths.append(tour_length(child))
        
        population = new_population
        lengths = np.array(new_lengths)
        
        cur_best_idx = int(np.argmin(lengths))
        cur_best_len = float(lengths[cur_best_idx])
        if cur_best_len < best_length:
            best_length = cur_best_len
            best_tour = population[cur_best_idx].copy()
            stagnation = 0
        else:
            stagnation += 1
        
        if stagnation >= stagnation_limit:
            break
    
    return best_tour.tolist(), best_length