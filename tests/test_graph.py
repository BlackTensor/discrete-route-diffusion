"""Tests for graph and tour representation (Task 1.2)."""

import math

import pytest
import torch

from data.generator import generate_instances
from routediff.graph import (
    adjacency_from_edge_set,
    distance_matrix,
    edge_set_to_tour,
    tour_length,
    tour_to_edge_set,
)


def test_distance_matrix_unbatched():
    coords = torch.tensor([[0.0, 0.0], [3.0, 0.0], [3.0, 4.0]])
    d = distance_matrix(coords)
    assert d.shape == (3, 3)
    assert torch.allclose(torch.diagonal(d), torch.zeros(3), atol=1e-6)
    assert math.isclose(d[0, 1].item(), 3.0, rel_tol=1e-5)
    assert math.isclose(d[0, 2].item(), 5.0, rel_tol=1e-5)


def test_distance_matrix_is_symmetric():
    coords = generate_instances(num_cities=8, num_instances=1, seed=0)[0]
    d = distance_matrix(coords)
    assert torch.allclose(d, d.T, atol=1e-6)


def test_distance_matrix_batched():
    coords = generate_instances(num_cities=6, num_instances=4, seed=1)
    d = distance_matrix(coords)
    assert d.shape == (4, 6, 6)


def test_tour_to_edge_set_degree_two():
    tour = torch.tensor([0, 1, 2, 3, 4])
    edges = tour_to_edge_set(tour)
    assert edges.shape == (5, 5)
    assert torch.allclose(edges, edges.T)
    assert torch.all(torch.diagonal(edges) == 0)
    assert torch.all(edges.sum(dim=1) == 2)


def test_permutation_edge_set_roundtrip():
    tour = torch.tensor([0, 3, 1, 4, 2])
    edges = tour_to_edge_set(tour)
    recovered = edge_set_to_tour(edges, start=0)
    # Same cyclic order (possibly reversed direction); compare edge sets.
    assert torch.equal(tour_to_edge_set(recovered), edges)


@pytest.mark.parametrize("seed", [0, 1, 2, 7])
def test_roundtrip_random_permutations(seed):
    g = torch.Generator().manual_seed(seed)
    tour = torch.randperm(12, generator=g)
    edges = tour_to_edge_set(tour)
    recovered = edge_set_to_tour(edges, start=int(tour[0].item()))
    assert torch.equal(tour_to_edge_set(recovered), edges)


def test_recovered_tour_is_valid_permutation():
    tour = torch.tensor([2, 0, 4, 1, 3])
    recovered = edge_set_to_tour(tour_to_edge_set(tour), start=2)
    assert torch.equal(torch.sort(recovered).values, torch.arange(5))


def test_tour_length_unit_square():
    # Unit square traversed in order: perimeter length is 4.
    coords = torch.tensor([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    tour = torch.tensor([0, 1, 2, 3])
    assert math.isclose(tour_length(coords, tour).item(), 4.0, rel_tol=1e-5)


def test_tour_length_is_order_invariant_to_rotation():
    coords = generate_instances(num_cities=9, num_instances=1, seed=3)[0]
    tour = torch.randperm(9, generator=torch.Generator().manual_seed(5))
    rotated = torch.roll(tour, shifts=3)
    a = tour_length(coords, tour)
    b = tour_length(coords, rotated)
    assert torch.allclose(a, b, atol=1e-5)


def test_adjacency_from_edge_set_is_float():
    edges = tour_to_edge_set(torch.tensor([0, 1, 2]))
    adj = adjacency_from_edge_set(edges)
    assert adj.dtype == torch.float32
    assert torch.equal(adj, edges)
