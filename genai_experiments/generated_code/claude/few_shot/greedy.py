import numpy as np

def solve(dist_matrix):
    """Nearest-neighbor greedy heuristic for TSP starting from city 0."""
    dist_matrix = np.asarray(dist_matrix)
    n = dist_matrix.shape[0]

    if n == 0:
        return [], 0.0
    if n == 1:
        return [0], 0.0

    start = 0
    tour = [start]
    visited = np.zeros(n, dtype=bool)
    visited[start] = True
    length = 0.0
    current = start

    for _ in range(n - 1):
        dists = dist_matrix[current].copy().astype(float)
        dists[visited] = np.inf
        next_city = int(np.argmin(dists))
        length += float(dist_matrix[current, next_city])
        tour.append(next_city)
        visited[next_city] = True
        current = next_city

    length += float(dist_matrix[current, start])
    return tour, length