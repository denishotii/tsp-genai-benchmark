import numpy as np

def solve(dist_matrix):
    """Solve TSP using the nearest-neighbor greedy heuristic starting at city 0."""
    dist_matrix = np.asarray(dist_matrix, dtype=float)
    n = dist_matrix.shape[0]

    if n == 0:
        return [], 0.0

    start = 0
    tour = [start]
    unvisited = set(range(1, n))
    length = 0.0
    current = start

    while unvisited:
        next_city = min(unvisited, key=lambda city: dist_matrix[current, city])
        length += dist_matrix[current, next_city]
        tour.append(next_city)
        unvisited.remove(next_city)
        current = next_city

    length += dist_matrix[current, start]

    return tour, float(length)