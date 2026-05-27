import numpy as np

def solve(dist_matrix):
    """Nearest-neighbor greedy heuristic for TSP starting from city 0."""
    n = len(dist_matrix)
    if n == 0:
        return [], 0.0
    
    visited = np.zeros(n, dtype=bool)
    tour = [0]
    visited[0] = True
    current = 0
    length = 0.0
    
    for _ in range(n - 1):
        dists = dist_matrix[current].copy()
        dists[visited] = np.inf
        nxt = int(np.argmin(dists))
        length += float(dist_matrix[current][nxt])
        tour.append(nxt)
        visited[nxt] = True
        current = nxt
    
    length += float(dist_matrix[current][tour[0]])
    return tour, length