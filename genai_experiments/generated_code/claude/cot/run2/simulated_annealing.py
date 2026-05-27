import numpy as np
import random
import math

def solve(dist_matrix):
    D = np.asarray(dist_matrix, dtype=float)
    n = D.shape[0]
    
    if n == 0:
        return [], 0.0
    if n == 1:
        return [0], 0.0
    if n == 2:
        return [0, 1], float(D[0, 1] + D[1, 0])
    
    def tour_length(t):
        idx = np.asarray(t)
        return float(D[idx, np.roll(idx, -1)].sum())
    
    # Initial tour: nearest neighbor heuristic
    def nearest_neighbor(start=0):
        unvisited = set(range(n))
        unvisited.remove(start)
        tour = [start]
        cur = start
        while unvisited:
            nxt = min(unvisited, key=lambda x: D[cur, x])
            tour.append(nxt)
            unvisited.remove(nxt)
            cur = nxt
        return tour
    
    current = nearest_neighbor(0)
    current_len = tour_length(current)
    best = list(current)
    best_len = current_len
    
    # Initial temperature: average edge length scale
    # Sample some distances to estimate
    nonzero = D[D > 0]
    if len(nonzero) == 0:
        return current, current_len
    avg_d = float(nonzero.mean())
    T0 = avg_d
    T_min = avg_d * 1e-6
    alpha = 0.9995
    
    # Number of iterations
    iters_per_temp = max(1, n)
    max_iters = max(20000, 200 * n * n)
    max_iters = min(max_iters, 500000)
    
    T = T0
    rng = random.Random(42)
    
    no_improve = 0
    it = 0
    while it < max_iters and T > T_min:
        for _ in range(iters_per_temp):
            it += 1
            # pick i < j, both in [0, n-1], avoid trivial (whole reversal equivalent)
            i = rng.randint(0, n - 1)
            j = rng.randint(0, n - 1)
            if i == j:
                continue
            if i > j:
                i, j = j, i
            # Skip if reversal covers whole tour (equivalent)
            if i == 0 and j == n - 1:
                continue
            
            a = current[i - 1] if i > 0 else current[n - 1]
            b = current[i]
            c = current[j]
            d = current[j + 1] if j + 1 < n else current[0]
            
            # If a==c or b==d we still compute correctly; edges adjacent special:
            # if j == i+0 impossible (i<j). If the segment is the whole tour minus one, etc.
            # The standard formula works as long as (a,b) and (c,d) are distinct edges.
            # When i==0 and j==n-1 we already skipped.
            
            removed = D[a, b] + D[c, d]
            added = D[a, c] + D[b, d]
            delta = added - removed
            
            if delta < 0 or rng.random() < math.exp(-delta / T):
                current[i:j + 1] = current[i:j + 1][::-1]
                current_len += delta
                if current_len < best_len - 1e-12:
                    best_len = current_len
                    best = list(current)
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                no_improve += 1
        
        T *= alpha
        
        # Optional restart from best if stuck for a long time
        if no_improve > 50 * n * n:
            current = list(best)
            current_len = best_len
            T = T0 * 0.3
            no_improve = 0
    
    # Final verification recompute to avoid floating drift
    best_len = tour_length(best)
    return best, best_len