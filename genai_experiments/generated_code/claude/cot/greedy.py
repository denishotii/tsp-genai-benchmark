import numpy as np

def solve(dist_matrix):
    dist_matrix = np.asarray(dist_matrix, dtype=float)
    n = dist_matrix.shape[0]
    
    if n == 0:
        return [], 0.0
    if n == 1:
        return [0], 0.0
    
    visited = np.zeros(n, dtype=bool)
    tour = [0]
    visited[0] = True
    current = 0
    
    for _ in range(n - 1):
        dists = dist_matrix[current].copy()
        dists[visited] = np.inf
        next_city = int(np.argmin(dists))
        tour.append(next_city)
        visited[next_city] = True
        current = next_city
    
    length = 0.0
    for i in range(n):
        length += float(dist_matrix[tour[i], tour[(i + 1) % n]])
    
    return tour, length