"""Tests for the synthetic TSP instance generator (Task 1.1)."""

import pytest
import torch

from data.generator import generate_instances


def test_shape():
    coords = generate_instances(num_cities=7, num_instances=4)
    assert coords.shape == (4, 7, 2)


def test_default_single_instance():
    coords = generate_instances(num_cities=5)
    assert coords.shape == (1, 5, 2)


def test_in_unit_square():
    coords = generate_instances(num_cities=20, num_instances=3, seed=1)
    assert torch.all(coords >= 0.0)
    assert torch.all(coords < 1.0)


def test_same_seed_is_reproducible():
    a = generate_instances(num_cities=10, num_instances=2, seed=42)
    b = generate_instances(num_cities=10, num_instances=2, seed=42)
    assert torch.equal(a, b)


def test_different_seed_differs():
    a = generate_instances(num_cities=10, num_instances=2, seed=1)
    b = generate_instances(num_cities=10, num_instances=2, seed=2)
    assert not torch.equal(a, b)


def test_seed_does_not_disturb_global_rng():
    torch.manual_seed(123)
    expected = torch.rand(3)
    torch.manual_seed(123)
    generate_instances(num_cities=5, num_instances=1, seed=99)
    after = torch.rand(3)
    assert torch.equal(expected, after)


@pytest.mark.parametrize("bad", [0, -1])
def test_invalid_num_cities(bad):
    with pytest.raises(ValueError):
        generate_instances(num_cities=bad)


@pytest.mark.parametrize("bad", [0, -1])
def test_invalid_num_instances(bad):
    with pytest.raises(ValueError):
        generate_instances(num_cities=5, num_instances=bad)
