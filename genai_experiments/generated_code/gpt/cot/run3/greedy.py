import numpy as np


def solve(dist_matrix):
    """
    Nearest-neighbor greedy heuristic for the Traveling Salesman Problem.

    Step-by-step outline:
    1. Start from a fixed city, chosen here as city 0.
    2. Keep track of which cities have not yet been visited.
    3. From the current city, repeatedly choose the nearest unvisited city.
       If there is a tie, NumPy's argmin chooses the first occurrence, so the
       lowest-index tied city is selected.
    4. Once all cities have been visited, add the distance from the last city
       back to the starting city to close the tour.

    Edge cases:
    - n == 0: return an empty tour with length 0.0.
    - n == 1: return [0] with length 0.0.
    """

    dist_matrix = np.asarray(dist_matrix, dtype=float)

    if dist_matrix.ndim != 2 or dist_matrix.shape[0] != dist_matrix.shape[1]:
        raise ValueError("dist_matrix must be a square 2D array")

    n = dist_matrix.shape[0]

    if n == 0:
        return [], 0.0

    start = 0
    current = start

    tour = [start]
    visited = np.zeros(n, dtype=bool)
    visited[start] = True

    length = 0.0

    while len(tour) < n:
        unvisited_cities = np.where(~visited)[0]

        distances = dist_matrix[current, unvisited_cities]
        nearest_pos = int(np.argmin(distances))
        next_city = int(unvisited_cities[nearest_pos])

        length += float(dist_matrix[current, next_city])

        tour.append(next_city)
        visited[next_city] = True
        current = next_city

    length += float(dist_matrix[current, start])

    return tour, length