"""Genetic Algorithm for the Traveling Salesman Problem.

Reference: Goldberg (1989), *Genetic Algorithms in Search, Optimization, and
Machine Learning*; Order Crossover from Davis (1985).

A population of candidate tours evolves over generations. Each generation:
parents are chosen by tournament selection, recombined with Order Crossover
(OX), occasionally mutated by swapping two cities, and the best individuals
are carried over unchanged (elitism).

Design choices (CLAUDE.md §4.3):
    - Representation : permutation encoding (list of city indices)
    - Initialization : random permutations (a "pure" GA, no warm start)
    - Selection      : tournament, k = 5
    - Crossover      : Order Crossover (OX) — preserves relative city order
    - Mutation       : swap two random cities, applied per child with a
                       fixed probability
    - Elitism        : the top ``elitism`` individuals survive intact
    - Stopping       : fixed number of generations
"""

import random

import numpy as np

from src.utils.tour import tour_length


def genetic_algorithm(
    dist_matrix: np.ndarray,
    *,
    population_size: int = 200,
    generations: int = 500,
    tournament_k: int = 5,
    mutation_rate: float = 0.1,
    elitism: int = 2,
    seed: int = 42,
) -> tuple[list[int], float]:
    """Run a genetic algorithm on a TSP distance matrix.

    Args:
        dist_matrix: square symmetric distance matrix of shape (n, n).
        population_size: number of individuals per generation.
        generations: number of generations to evolve.
        tournament_k: number of competitors per tournament selection.
        mutation_rate: probability that a given child is swap-mutated.
        elitism: number of top individuals carried over each generation.
        seed: RNG seed; seeds both ``random`` and ``numpy.random``.

    Returns:
        ``(best_tour, best_length)`` — the best tour seen across the run.
    """
    rng = random.Random(seed)
    np.random.seed(seed)
    n = dist_matrix.shape[0]

    population = [_random_tour(n, rng) for _ in range(population_size)]
    fitnesses = [tour_length(ind, dist_matrix) for ind in population]

    best_idx = min(range(population_size), key=lambda i: fitnesses[i])
    best = population[best_idx][:]
    best_length = fitnesses[best_idx]

    for _ in range(generations):
        ranked = sorted(range(population_size), key=lambda i: fitnesses[i])
        next_population = [population[ranked[e]][:] for e in range(elitism)]

        while len(next_population) < population_size:
            parent1 = _tournament_select(population, fitnesses, tournament_k, rng)
            parent2 = _tournament_select(population, fitnesses, tournament_k, rng)
            child = _order_crossover(parent1, parent2, rng)
            if rng.random() < mutation_rate:
                _swap_mutation(child, rng)
            next_population.append(child)

        population = next_population
        fitnesses = [tour_length(ind, dist_matrix) for ind in population]

        gen_best_idx = min(range(population_size), key=lambda i: fitnesses[i])
        if fitnesses[gen_best_idx] < best_length:
            best = population[gen_best_idx][:]
            best_length = fitnesses[gen_best_idx]

    return best, best_length


def _random_tour(n: int, rng: random.Random) -> list[int]:
    """Return a random permutation of ``range(n)``."""
    tour = list(range(n))
    rng.shuffle(tour)
    return tour


def _tournament_select(
    population: list[list[int]],
    fitnesses: list[float],
    k: int,
    rng: random.Random,
) -> list[int]:
    """Return the fittest (shortest-tour) of ``k`` random competitors."""
    competitors = rng.sample(range(len(population)), k)
    winner = min(competitors, key=lambda idx: fitnesses[idx])
    return population[winner]


def _order_crossover(
    parent1: list[int],
    parent2: list[int],
    rng: random.Random,
) -> list[int]:
    """Order Crossover (OX).

    Copies a random contiguous segment of ``parent1`` into the child at the
    same positions, then fills the remaining positions with the cities of
    ``parent2`` in their relative order — both the read order from
    ``parent2`` and the fill order in the child start just after the
    segment and wrap around. The result is always a valid permutation.
    """
    n = len(parent1)
    a, b = sorted(rng.sample(range(n), 2))

    child: list[int] = [-1] * n
    child[a:b + 1] = parent1[a:b + 1]
    used = set(parent1[a:b + 1])

    # parent2's cities, read starting just after the segment (wrapping),
    # skipping any already placed from parent1's segment.
    remaining = [parent2[(b + 1 + k) % n] for k in range(n)]
    remaining = [city for city in remaining if city not in used]

    fill = 0
    for k in range(n):
        pos = (b + 1 + k) % n
        if child[pos] == -1:
            child[pos] = remaining[fill]
            fill += 1
    return child


def _swap_mutation(tour: list[int], rng: random.Random) -> None:
    """Swap two distinct random cities in ``tour`` in place."""
    i, j = rng.sample(range(len(tour)), 2)
    tour[i], tour[j] = tour[j], tour[i]
