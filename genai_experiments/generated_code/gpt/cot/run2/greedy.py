import numpy as np


def solve(dist_matrix):
    """
    Nearest-neighbor greedy heuristic for the Traveling Salesman Problem.

    Step 1: Problem and algorithm
    - We are given an n x n symmetric distance matrix.
    - We build one closed tour greedily.
    - Start from a fixed city: city 0.
    - Repeatedly move to the nearest city that has not yet been visited.
    - After visiting all cities, return to the starting city.
    - The returned tour lists each city exactly once; the closing edge is included
      only in the returned length, not appended to the tour.

    Step 2: Data structures and main loop
    - `visited`: boolean NumPy array marking which cities have been visited.
    - `tour`: Python list storing the visiting order.
    - `current`: the current city.
    - At each iteration:
        * collect indices of unvisited cities,
        * choose the one with minimum distance from `current`,
        * append it to the tour,
        * mark it visited,
        * add the traveled distance to `length`.

    Step 3: Edge cases
    - n == 0: return an empty tour and length 0.0.
    - n == 1: return [0] and length 0.0, since the only closing edge is 0 -> 0.
    - Ties are broken deterministically by NumPy's argmin over ascending city
      indices, so the smallest-index nearest unvisited city is chosen.
    """

    dist_matrix = np.asarray(dist_matrix, dtype=float)

    if dist_matrix.ndim != 2 or dist_matrix.shape[0] != dist_matrix.shape[1]:
        raise ValueError("dist_matrix must be a square 2D array")

    n = dist_matrix.shape[0]

    if n == 0:
        return [], 0.0

    start = 0
    current = start

    visited = np.zeros(n, dtype=bool)
    visited[start] = True

    tour = [start]
    length = 0.0

    for _ in range(n - 1):
        unvisited = np.flatnonzero(~visited)

        distances = dist_matrix[current, unvisited]
        next_city = int(unvisited[np.argmin(distances)])

        length += float(dist_matrix[current, next_city])
        tour.append(next_city)
        visited[next_city] = True
        current = next_city

    length += float(dist_matrix[current, start])

    return tour, length