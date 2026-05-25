"""Tests for src/algorithms/* — reference TSP solvers."""

import random
from pathlib import Path

import pytest

from src.algorithms.genetic_algorithm import (
    _order_crossover,
    _random_tour,
    _swap_mutation,
    genetic_algorithm,
)
from src.algorithms.greedy import greedy_tour
from src.algorithms.simulated_annealing import simulated_annealing
from src.utils.distance import distance_matrix
from src.utils.parser import parse_tsp
from src.utils.tour import is_valid_tour, tour_length

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "tsplib"

TSPLIB_OPTIMA = {
    "eil51": 426,
    "berlin52": 7542,
    "ch130": 6110,
    "d198": 15780,
}


# --- greedy nearest-neighbor ---

def test_greedy_on_345_triangle_is_optimal():
    # Only one distinct closed tour exists on 3 cities, so greedy is trivially optimal.
    coords = [(0.0, 0.0), (3.0, 0.0), (0.0, 4.0)]
    D = distance_matrix(coords)
    tour, length = greedy_tour(D, start=0)
    assert is_valid_tour(tour, 3)
    assert length == 12.0


def test_greedy_on_axis_aligned_square_is_optimal():
    # Cities at the corners of a 10x10 square; greedy from (0,0) walks the
    # perimeter (length 40), which is also the true optimum.
    coords = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]
    D = distance_matrix(coords)
    tour, length = greedy_tour(D, start=0)
    assert is_valid_tour(tour, 4)
    assert length == 40.0
    assert tour == [0, 1, 2, 3]


def test_greedy_returns_consistent_length():
    coords = parse_tsp(DATA_DIR / "eil51.tsp")
    D = distance_matrix(coords)
    tour, length = greedy_tour(D)
    # Returned length must equal an independent recomputation.
    assert length == tour_length(tour, D)


def test_greedy_respects_start_city():
    coords = parse_tsp(DATA_DIR / "eil51.tsp")
    D = distance_matrix(coords)
    tour0, _ = greedy_tour(D, start=0)
    tour7, _ = greedy_tour(D, start=7)
    assert tour0[0] == 0
    assert tour7[0] == 7
    assert is_valid_tour(tour0, 51)
    assert is_valid_tour(tour7, 51)


@pytest.mark.parametrize("name", ["eil51", "berlin52", "ch130", "d198"])
def test_greedy_on_tsplib_produces_valid_tour(name):
    coords = parse_tsp(DATA_DIR / f"{name}.tsp")
    D = distance_matrix(coords)
    n = len(coords)
    tour, length = greedy_tour(D)
    assert is_valid_tour(tour, n)
    assert length == tour_length(tour, D)
    # Greedy is suboptimal but should not be catastrophic. The well-known
    # bound for EUC_2D nearest-neighbor is roughly 25% above optimum;
    # we check a safe 2x ceiling to catch any gross regression.
    assert length < 2 * TSPLIB_OPTIMA[name]


# --- simulated annealing ---

def test_sa_returns_valid_tour_with_consistent_length():
    coords = parse_tsp(DATA_DIR / "eil51.tsp")
    D = distance_matrix(coords)
    tour, length = simulated_annealing(D)
    assert is_valid_tour(tour, 51)
    assert length == tour_length(tour, D)


def test_sa_beats_greedy():
    coords = parse_tsp(DATA_DIR / "eil51.tsp")
    D = distance_matrix(coords)
    _, greedy_len = greedy_tour(D)
    _, sa_len = simulated_annealing(D)
    assert sa_len < greedy_len


def test_sa_is_reproducible_for_fixed_seed():
    coords = parse_tsp(DATA_DIR / "eil51.tsp")
    D = distance_matrix(coords)
    tour_a, len_a = simulated_annealing(D, seed=42)
    tour_b, len_b = simulated_annealing(D, seed=42)
    assert tour_a == tour_b
    assert len_a == len_b


@pytest.mark.parametrize(
    "name,target",
    [
        ("eil51", 5.0),
        ("berlin52", 5.0),
        ("ch130", 10.0),
        ("d198", 10.0),
    ],
)
def test_sa_meets_quality_target(name, target):
    # CLAUDE.md §11 targets: < 5% gap on eil51/berlin52, < 10% on the larger
    # instances. With the default seed these are met comfortably; the test
    # guards against regressions in the cooling schedule or move operator.
    coords = parse_tsp(DATA_DIR / f"{name}.tsp")
    D = distance_matrix(coords)
    n = len(coords)
    tour, length = simulated_annealing(D)
    assert is_valid_tour(tour, n)
    assert length == tour_length(tour, D)
    gap = (length - TSPLIB_OPTIMA[name]) / TSPLIB_OPTIMA[name] * 100
    assert gap < target, f"{name}: gap {gap:.2f}% exceeds {target}% target"


# --- genetic algorithm operators ---

def test_ox_always_produces_valid_permutation():
    rng = random.Random(0)
    for _ in range(500):
        parent1 = _random_tour(20, rng)
        parent2 = _random_tour(20, rng)
        child = _order_crossover(parent1, parent2, rng)
        assert sorted(child) == list(range(20))


def test_ox_of_identical_parents_returns_same_tour():
    # Recombining a tour with itself must reconstruct it exactly: the copied
    # segment plus the in-order fill from the (identical) second parent can
    # only reproduce the original.
    rng = random.Random(5)
    parent = _random_tour(20, rng)
    assert _order_crossover(parent, parent, rng) == parent


def test_swap_mutation_changes_exactly_two_positions():
    rng = random.Random(1)
    tour = list(range(10))
    original = tour[:]
    _swap_mutation(tour, rng)
    changed = [k for k in range(10) if tour[k] != original[k]]
    assert len(changed) == 2
    assert sorted(tour) == list(range(10))


# --- genetic algorithm (correctness, not quality — pure GA quality varies) ---

def test_ga_returns_valid_tour_with_consistent_length():
    coords = parse_tsp(DATA_DIR / "eil51.tsp")
    D = distance_matrix(coords)
    tour, length = genetic_algorithm(D, generations=50)
    assert is_valid_tour(tour, 51)
    assert length == tour_length(tour, D)


def test_ga_is_reproducible_for_fixed_seed():
    coords = parse_tsp(DATA_DIR / "eil51.tsp")
    D = distance_matrix(coords)
    tour_a, len_a = genetic_algorithm(D, generations=30, seed=42)
    tour_b, len_b = genetic_algorithm(D, generations=30, seed=42)
    assert tour_a == tour_b
    assert len_a == len_b


def test_ga_improves_substantially_over_random_tour():
    # The GA need not beat greedy (pure GA is weak on TSP), but it must
    # evolve far better tours than an unoptimized random permutation.
    coords = parse_tsp(DATA_DIR / "eil51.tsp")
    D = distance_matrix(coords)
    rng = random.Random(0)
    random_len = tour_length(_random_tour(51, rng), D)
    _, ga_len = genetic_algorithm(D, generations=200)
    assert ga_len < 0.5 * random_len
