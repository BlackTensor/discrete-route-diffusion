"""Tests for baseline TSP solvers (Task 1.3)."""

import math

import pytest
import torch

from data.generator import generate_instances
from routediff.graph import tour_length
from routediff.solvers import (
    ORTOOLS_AVAILABLE,
    is_valid_tour,
    nearest_neighbor,
    solve,
    solve_or_tools,
    two_opt,
)


def _square():
    # Unit square; optimal tour is the perimeter with length 4.
    return torch.tensor([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])


def test_nearest_neighbor_returns_valid_tour():
    coords = generate_instances(num_cities=12, num_instances=1, seed=0)[0]
    tour = nearest_neighbor(coords)
    assert is_valid_tour(tour, 12)


def test_nearest_neighbor_respects_start():
    coords = generate_instances(num_cities=8, num_instances=1, seed=1)[0]
    tour = nearest_neighbor(coords, start=3)
    assert int(tour[0]) == 3
    assert is_valid_tour(tour, 8)


def test_two_opt_does_not_worsen():
    coords = generate_instances(num_cities=15, num_instances=1, seed=2)[0]
    nn = nearest_neighbor(coords)
    opt = two_opt(coords, nn)
    assert is_valid_tour(opt, 15)
    assert tour_length(coords, opt) <= tour_length(coords, nn) + 1e-6


def test_two_opt_finds_square_optimum():
    coords = _square()
    # Start from a crossing tour (0-2-1-3) which is suboptimal.
    bad = torch.tensor([0, 2, 1, 3])
    opt = two_opt(coords, bad)
    assert math.isclose(tour_length(coords, opt).item(), 4.0, rel_tol=1e-5)


@pytest.mark.parametrize("n", [1, 2, 3])
def test_two_opt_tiny_instances(n):
    coords = generate_instances(num_cities=n, num_instances=1, seed=0)[0]
    tour = two_opt(coords, nearest_neighbor(coords))
    assert is_valid_tour(tour, n)


def test_solve_auto_returns_valid_tour():
    coords = generate_instances(num_cities=20, num_instances=1, seed=4)[0]
    tour = solve(coords)
    assert is_valid_tour(tour, 20)


def test_solve_nn2opt_method():
    coords = generate_instances(num_cities=18, num_instances=1, seed=5)[0]
    tour = solve(coords, method="nn2opt")
    assert is_valid_tour(tour, 18)


def test_solve_unknown_method_raises():
    coords = _square()
    with pytest.raises(ValueError):
        solve(coords, method="bogus")


@pytest.mark.skipif(not ORTOOLS_AVAILABLE, reason="OR-Tools not installed")
def test_or_tools_returns_valid_tour():
    coords = generate_instances(num_cities=20, num_instances=1, seed=6)[0]
    tour = solve_or_tools(coords, time_limit_s=2.0)
    assert is_valid_tour(tour, 20)


@pytest.mark.skipif(not ORTOOLS_AVAILABLE, reason="OR-Tools not installed")
def test_or_tools_beats_or_matches_nearest_neighbor():
    coords = generate_instances(num_cities=25, num_instances=1, seed=7)[0]
    nn = nearest_neighbor(coords)
    ort = solve_or_tools(coords, time_limit_s=2.0)
    assert tour_length(coords, ort) <= tour_length(coords, nn) + 1e-6


@pytest.mark.skipif(not ORTOOLS_AVAILABLE, reason="OR-Tools not installed")
def test_or_tools_solves_square():
    coords = _square()
    tour = solve_or_tools(coords, time_limit_s=1.0)
    assert math.isclose(tour_length(coords, tour).item(), 4.0, rel_tol=1e-5)
