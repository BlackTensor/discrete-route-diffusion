"""Tests for the forward (noising) diffusion process (Task 2.1)."""

import pytest
import torch

from data.generator import generate_instances
from routediff.diffusion import (
    DiffusionSchedule,
    corrupt,
    edge_prediction_loss,
    fraction_edges_retained,
    make_schedule,
    make_training_pair,
    q_present_probs,
    sample_prior,
    sample_timesteps,
)
from routediff.graph import tour_to_edge_set
from routediff.solvers import nearest_neighbor


def _clean_edge_set(num_cities=12, seed=0):
    coords = generate_instances(num_cities=num_cities, num_instances=1, seed=seed)[0]
    tour = nearest_neighbor(coords)
    return tour_to_edge_set(tour, num_cities)


@pytest.mark.parametrize("kind", ["cosine", "linear"])
def test_schedule_shapes_and_endpoints(kind):
    sched = make_schedule(50, kind=kind)
    assert isinstance(sched, DiffusionSchedule)
    assert sched.betas.shape == (51,)
    assert sched.alphas.shape == (51,)
    assert sched.alpha_bars.shape == (51,)
    # Clean endpoint.
    assert sched.betas[0].item() == 0.0
    assert pytest.approx(sched.alpha_bars[0].item(), abs=1e-6) == 1.0


@pytest.mark.parametrize("kind", ["cosine", "linear"])
def test_alpha_bars_monotonic_decreasing(kind):
    sched = make_schedule(100, kind=kind)
    diffs = sched.alpha_bars[1:] - sched.alpha_bars[:-1]
    assert torch.all(diffs <= 1e-6)
    # Substantial corruption is reached by the final step.
    assert sched.alpha_bars[-1].item() < 0.1


def test_betas_in_unit_interval():
    sched = make_schedule(100, kind="cosine")
    assert torch.all(sched.betas >= 0.0)
    assert torch.all(sched.betas <= 1.0)


def test_invalid_schedule_args():
    with pytest.raises(ValueError):
        make_schedule(0)
    with pytest.raises(ValueError):
        make_schedule(10, kind="quadratic")


def test_corrupt_t0_is_identity():
    clean = _clean_edge_set()
    sched = make_schedule(50)
    noisy = corrupt(clean, 0, sched)
    assert torch.equal(noisy, clean)


def test_corrupt_is_symmetric_binary_zero_diagonal():
    clean = _clean_edge_set(num_cities=15)
    sched = make_schedule(50)
    g = torch.Generator().manual_seed(1)
    noisy = corrupt(clean, 25, sched, generator=g)
    assert torch.allclose(noisy, noisy.T)
    assert torch.all(torch.diagonal(noisy) == 0)
    assert set(noisy.unique().tolist()).issubset({0.0, 1.0})


def test_corrupt_reproducible_with_generator():
    clean = _clean_edge_set()
    sched = make_schedule(50)
    a = corrupt(clean, 30, sched, generator=torch.Generator().manual_seed(7))
    b = corrupt(clean, 30, sched, generator=torch.Generator().manual_seed(7))
    assert torch.equal(a, b)


def test_q_present_probs_endpoints():
    clean = _clean_edge_set()
    sched = make_schedule(50)
    # At t=0 the present-probability equals the clean edge value exactly.
    p0 = q_present_probs(clean, 0, sched)
    assert torch.allclose(p0, clean.to(torch.float32))
    # At t=T present-probability is ~0.5 everywhere (uniform prior).
    pT = q_present_probs(clean, sched.num_timesteps, sched)
    assert torch.allclose(pT, torch.full_like(pT, 0.5), atol=0.05)


def test_overlap_decreases_monotonically_on_average():
    clean = _clean_edge_set(num_cities=14, seed=3)
    sched = make_schedule(60)
    g = torch.Generator().manual_seed(0)
    timesteps = [0, 10, 20, 30, 40, 50, 60]
    means = []
    for t in timesteps:
        samples = torch.stack([corrupt(clean, t, sched, generator=g) for _ in range(64)])
        retained = fraction_edges_retained(clean.expand_as(samples), samples)
        means.append(retained.mean().item())
    # Average retained fraction must not increase as t grows (allow tiny noise).
    for earlier, later in zip(means, means[1:]):
        assert later <= earlier + 1e-2
    assert means[0] == 1.0
    # By the final step retention has dropped substantially toward the 0.5 prior.
    assert means[-1] < 0.7


def test_corrupt_batched_shapes_and_per_item_timesteps():
    coords = generate_instances(num_cities=10, num_instances=4, seed=2)
    clean = torch.stack([tour_to_edge_set(nearest_neighbor(coords[i]), 10) for i in range(4)])
    sched = make_schedule(40)
    t = torch.tensor([0, 10, 20, 40])
    g = torch.Generator().manual_seed(4)
    noisy = corrupt(clean, t, sched, generator=g)
    assert noisy.shape == (4, 10, 10)
    # Per-item t=0 row is clean; later items diverge.
    assert torch.equal(noisy[0], clean[0])
    assert torch.allclose(noisy, noisy.transpose(-1, -2))


def test_sample_prior_shape_and_stats():
    g = torch.Generator().manual_seed(0)
    batch = sample_prior(20, batch_size=200, generator=g)
    assert batch.shape == (200, 20, 20)
    assert torch.allclose(batch, batch.transpose(-1, -2))
    assert torch.all(torch.diagonal(batch, dim1=-2, dim2=-1) == 0)
    # Off-diagonal edges are present roughly half the time.
    n = 20
    upper = torch.triu(torch.ones(n, n), diagonal=1)
    frac = (batch * upper).sum() / (batch.shape[0] * upper.sum())
    assert 0.45 < frac.item() < 0.55


def test_unbatched_requires_scalar_t():
    clean = _clean_edge_set()
    sched = make_schedule(50)
    with pytest.raises(ValueError):
        corrupt(clean, torch.tensor([1, 2]), sched)


def test_timestep_out_of_range_raises():
    clean = _clean_edge_set()
    sched = make_schedule(50)
    with pytest.raises(ValueError):
        corrupt(clean, 51, sched)


# --- Task 2.2: reverse target definition and loss --------------------------


def test_sample_timesteps_range_and_shape():
    sched = make_schedule(50)
    g = torch.Generator().manual_seed(0)
    t = sample_timesteps(256, sched, generator=g)
    assert t.shape == (256,)
    assert t.dtype == torch.long
    # Default excludes t=0 so every example carries noise.
    assert int(t.min()) >= 1
    assert int(t.max()) <= sched.num_timesteps
    t0 = sample_timesteps(256, sched, generator=g, include_zero=True)
    assert int(t0.min()) >= 0


def test_make_training_pair_shapes_and_target():
    clean = _clean_edge_set(num_cities=12)
    sched = make_schedule(50)
    noisy, target = make_training_pair(clean, 20, sched, generator=torch.Generator().manual_seed(1))
    assert noisy.shape == clean.shape
    assert target.shape == clean.shape
    # Target is exactly the clean edge set the model must reconstruct.
    assert torch.equal(target, clean.to(torch.float32))
    # Symmetric, binary, zero-diagonal noisy input.
    assert torch.allclose(noisy, noisy.T)
    assert torch.all(torch.diagonal(noisy) == 0)


def test_make_training_pair_t0_input_equals_target():
    clean = _clean_edge_set()
    sched = make_schedule(50)
    noisy, target = make_training_pair(clean, 0, sched)
    assert torch.equal(noisy, target)


def test_loss_is_scalar_with_dummy_predictions():
    clean = _clean_edge_set(num_cities=12)
    sched = make_schedule(50)
    _, target = make_training_pair(clean, 25, sched, generator=torch.Generator().manual_seed(2))
    dummy_logits = torch.zeros_like(target)
    loss = edge_prediction_loss(dummy_logits, target)
    assert loss.dim() == 0
    # Zero logits -> p=0.5 -> BCE = ln(2) per edge.
    assert pytest.approx(loss.item(), abs=1e-5) == float(torch.log(torch.tensor(2.0)))


def test_perfect_prediction_has_near_zero_loss():
    clean = _clean_edge_set()
    sched = make_schedule(50)
    _, target = make_training_pair(clean, 30, sched)
    confident_logits = (target * 2.0 - 1.0) * 20.0  # +20 for present, -20 for absent
    loss = edge_prediction_loss(confident_logits, target)
    assert loss.item() < 1e-6


def test_wrong_prediction_has_large_loss():
    clean = _clean_edge_set()
    sched = make_schedule(50)
    _, target = make_training_pair(clean, 30, sched)
    right = edge_prediction_loss((target * 2.0 - 1.0) * 20.0, target)
    wrong = edge_prediction_loss((target * 2.0 - 1.0) * -20.0, target)
    assert wrong.item() > right.item() + 1.0


def test_loss_batched_and_reductions():
    coords = generate_instances(num_cities=10, num_instances=4, seed=2)
    clean = torch.stack([tour_to_edge_set(nearest_neighbor(coords[i]), 10) for i in range(4)])
    sched = make_schedule(40)
    t = sample_timesteps(4, sched, generator=torch.Generator().manual_seed(3))
    _, target = make_training_pair(clean, t, sched, generator=torch.Generator().manual_seed(3))
    logits = torch.zeros_like(target)
    per_edge = edge_prediction_loss(logits, target, reduction="none")
    num_edges = 10 * 9 // 2
    assert per_edge.shape == (4, num_edges)
    assert edge_prediction_loss(logits, target, reduction="mean").dim() == 0
    assert edge_prediction_loss(logits, target, reduction="sum").dim() == 0


def test_loss_gradients_flow():
    clean = _clean_edge_set()
    sched = make_schedule(50)
    _, target = make_training_pair(clean, 20, sched)
    logits = torch.zeros_like(target, requires_grad=True)
    edge_prediction_loss(logits, target).backward()
    assert logits.grad is not None
    # Only upper-triangle edges receive gradient (diagonal/lower stay zero).
    assert torch.any(logits.grad != 0)


def test_pos_weight_increases_loss_on_missed_present_edges():
    clean = _clean_edge_set()
    sched = make_schedule(50)
    _, target = make_training_pair(clean, 30, sched)
    # Predict all-absent: present edges are all wrong.
    logits = torch.full_like(target, -5.0)
    base = edge_prediction_loss(logits, target)
    weighted = edge_prediction_loss(logits, target, pos_weight=10.0)
    assert weighted.item() > base.item()


def test_loss_shape_mismatch_raises():
    sched = make_schedule(50)
    _, target = make_training_pair(_clean_edge_set(num_cities=10), 10, sched)
    with pytest.raises(ValueError):
        edge_prediction_loss(torch.zeros(8, 8), target)
