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
        idx = np.asarray(tour)
        return float(dist[idx, np.roll(idx, -1)].sum())
    
    # GA parameters
    pop_size = min(200, max(50, 10 * n))
    generations = min(1000, max(100, 30 * n))
    tournament_k = 5
    mutation_rate = 0.2
    elite_size = 2
    crossover_rate = 0.9
    
    # Initialize population with random permutations
    population = []
    base = list(range(n))
    for _ in range(pop_size):
        perm = base[:]
        random.shuffle(perm)
        population.append(perm)
    
    # Add a greedy nearest-neighbor seed for a good start
    def nearest_neighbor(start):
        visited = [False] * n
        tour = [start]
        visited[start] = True
        cur = start
        for _ in range(n - 1):
            best_d = float('inf')
            best_j = -1
            for j in range(n):
                if not visited[j] and dist[cur, j] < best_d:
                    best_d = dist[cur, j]
                    best_j = j
            tour.append(best_j)
            visited[best_j] = True
            cur = best_j
        return tour
    
    population[0] = nearest_neighbor(0)
    if n > 1:
        population[1] = nearest_neighbor(random.randrange(n))
    
    fitness = [tour_length(ind) for ind in population]
    
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
        a = random.randrange(n)
        b = random.randrange(n)
        if a > b:
            a, b = b, a
        child = [-1] * n
        # Copy segment from p1
        segment = set()
        for i in range(a, b + 1):
            child[i] = p1[i]
            segment.add(p1[i])
        # Fill remainder from p2 in order starting after b
        pos = (b + 1) % n
        p2_pos = (b + 1) % n
        filled = b - a + 1
        while filled < n:
            gene = p2[p2_pos]
            if gene not in segment:
                child[pos] = gene
                pos = (pos + 1) % n
                filled += 1
            p2_pos = (p2_pos + 1) % n
        return child
    
    def swap_mutate(ind):
        i = random.randrange(n)
        j = random.randrange(n)
        ind[i], ind[j] = ind[j], ind[i]
        return ind
    
    best_idx = int(np.argmin(fitness))
    best_tour = population[best_idx][:]
    best_length = fitness[best_idx]
    
    stagnation = 0
    max_stagnation = max(50, n * 2)
    
    for gen in range(generations):
        # Sort by fitness for elitism
        order = sorted(range(pop_size), key=lambda i: fitness[i])
        new_pop = [population[order[i]][:] for i in range(elite_size)]
        new_fit = [fitness[order[i]] for i in range(elite_size)]
        
        while len(new_pop) < pop_size:
            p1 = tournament_select()
            p2 = tournament_select()
            if random.random() < crossover_rate:
                child = order_crossover(p1, p2)
            else:
                child = p1[:]
            if random.random() < mutation_rate:
                child = swap_mutate(child)
            new_pop.append(child)
            new_fit.append(tour_length(child))
        
        population = new_pop
        fitness = new_fit
        
        cur_best = int(np.argmin(fitness))
        if fitness[cur_best] < best_length - 1e-12:
            best_length = fitness[cur_best]
            best_tour = population[cur_best][:]
            stagnation = 0
        else:
            stagnation += 1
        
        if stagnation >= max_stagnation:
            break
    
    return list(best_tour), float(best_length)