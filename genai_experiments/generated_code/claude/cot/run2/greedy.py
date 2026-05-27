import numpy as np

def solve(dist_matrix):
    dist = np.asarray(dist_matrix, dtype=float)
    n = dist.shape[0]
    
    if n == 0:
        return [], 0.0
    if n == 1:
        return [0], 0.0
    
    visited = np.zeros(n, dtype=bool)
    tour = [0]
    visited[0] = True
    current = 0
    length = 0.0
    
    for _ in range(n - 1):
        row = dist[current].copy()
        row[visited] = np.inf
        nxt = int(np.argmin(row))
        length += float(dist[current, nxt])
        tour.append(nxt)
        visited[nxt] = True
        current = nxt
    
    length += float(dist[current, tour[0]])
    return tour, length