"""Unit tests for src/utils: parser, distance matrix, tour helpers."""

from pathlib import Path

import numpy as np
import pytest

from src.utils.distance import distance_matrix
from src.utils.parser import parse_tsp
from src.utils.tour import is_valid_tour, tour_length

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "tsplib"


# --- parser ---

def test_eil51_has_51_nodes():
    coords = parse_tsp(DATA_DIR / "eil51.tsp")
    assert len(coords) == 51
    assert coords[0] == (37.0, 52.0)


def test_berlin52_has_52_nodes():
    coords = parse_tsp(DATA_DIR / "berlin52.tsp")
    assert len(coords) == 52
    assert coords[0] == (565.0, 575.0)


def test_ch130_has_130_nodes():
    coords = parse_tsp(DATA_DIR / "ch130.tsp")
    assert len(coords) == 130


def test_d198_scientific_notation_parses():
    coords = parse_tsp(DATA_DIR / "d198.tsp")
    assert len(coords) == 198
    assert coords[0] == (0.0, 0.0)
    assert coords[-1] == pytest.approx((3952.10, 1010.30))


# --- distance matrix ---

def test_distance_matrix_345_right_triangle():
    coords = [(0.0, 0.0), (3.0, 0.0), (0.0, 4.0)]
    D = distance_matrix(coords)
    assert D[0, 1] == 3.0
    assert D[0, 2] == 4.0
    assert D[1, 2] == 5.0


def test_distance_matrix_is_symmetric_with_zero_diagonal():
    coords = parse_tsp(DATA_DIR / "eil51.tsp")
    D = distance_matrix(coords)
    assert D.shape == (51, 51)
    assert np.array_equal(D, D.T)
    assert np.all(np.diag(D) == 0)
    assert np.all(D >= 0)


def test_distance_matrix_values_are_integers_per_tsplib_spec():
    coords = parse_tsp(DATA_DIR / "berlin52.tsp")
    D = distance_matrix(coords)
    assert np.all(D == np.round(D))


# --- tour helpers ---

def test_tour_length_closed_triangle():
    coords = [(0.0, 0.0), (3.0, 0.0), (0.0, 4.0)]
    D = distance_matrix(coords)
    # closed tour 0 -> 1 -> 2 -> 0: edges 3 + 5 + 4 = 12
    assert tour_length([0, 1, 2], D) == 12.0


def test_tour_length_is_start_invariant():
    coords = [(0.0, 0.0), (3.0, 0.0), (0.0, 4.0)]
    D = distance_matrix(coords)
    assert tour_length([0, 1, 2], D) == tour_length([1, 2, 0], D) == tour_length([2, 0, 1], D)


def test_is_valid_tour_accepts_permutations():
    assert is_valid_tour([0, 1, 2, 3], 4)
    assert is_valid_tour([3, 1, 0, 2], 4)


def test_is_valid_tour_rejects_duplicates():
    assert not is_valid_tour([0, 1, 1, 2], 4)


def test_is_valid_tour_rejects_wrong_length():
    assert not is_valid_tour([0, 1, 2], 4)
    assert not is_valid_tour([0, 1, 2, 3, 4], 4)


def test_is_valid_tour_rejects_out_of_range_city():
    assert not is_valid_tour([0, 1, 2, 5], 4)
