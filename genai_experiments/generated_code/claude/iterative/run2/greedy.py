import numpy as np

def solve(dist_matrix):
    n = len(dist_matrix)
    if n == 0:
        return [], 0.0
    
    start = 0
    tour = [start]
    visited = np.zeros(n, dtype=bool)
    visited[start] = True
    current = start
    length = 0.0
    
    for _ in range(n - 1):
        dists = dist_matrix[current].copy()
        dists[visited] = np.inf
        next_city = int(np.argmin(dists))
        length += float(dist_matrix[current][next_city])
        tour.append(next_city)
        visited[next_city] = True
        current = next_city
    
    length += float(dist_matrix[current][start])
    return tour, length