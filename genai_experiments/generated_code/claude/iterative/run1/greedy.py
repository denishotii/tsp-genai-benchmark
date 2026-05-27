import numpy as np

def solve(dist_matrix):
    n = len(dist_matrix)
    if n == 0:
        return [], 0.0
    
    visited = [False] * n
    tour = [0]
    visited[0] = True
    current = 0
    length = 0.0
    
    for _ in range(n - 1):
        dists = dist_matrix[current].copy()
        dists = np.where(visited, np.inf, dists)
        next_city = int(np.argmin(dists))
        length += float(dist_matrix[current][next_city])
        tour.append(next_city)
        visited[next_city] = True
        current = next_city
    
    length += float(dist_matrix[current][tour[0]])
    return tour, length