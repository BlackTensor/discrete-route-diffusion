"""Tests for the GNN denoiser model (Tasks 3.1 and 3.2)."""

import torch

from data.generator import generate_instances
from routediff.diffusion import corrupt, edge_prediction_loss, make_schedule, make_training_pair
from routediff.graph import tour_to_edge_set
from routediff.model import RouteDiffusionDenoiser, sinusoidal_timestep_embedding
from routediff.solvers import nearest_neighbor


def _clean_edge_set(num_cities=12, seed=0):
    coords = generate_instances(num_cities=num_cities, num_instances=1, seed=seed)[0]
    tour = nearest_neighbor(coords)
    return coords, tour_to_edge_set(tour, num_cities)


def test_forward_unbatched_shape():
    n = 12
    coords, clean = _clean_edge_set(n)
    sched = make_schedule(50)
    noisy = corrupt(clean, 10, sched)
    model = RouteDiffusionDenoiser(hidden_dim=32, num_layers=3)
    logits = model(coords, noisy, 10)
    assert logits.shape == (n, n)
    assert torch.isfinite(logits).all()


def test_forward_batched_shape():
    n, b = 10, 4
    coords = generate_instances(num_cities=n, num_instances=b, seed=1)
    edge_sets = torch.stack([tour_to_edge_set(nearest_neighbor(c), n) for c in coords])
    sched = make_schedule(50)
    t = torch.randint(1, 51, (b,))
    noisy = corrupt(edge_sets, t, sched)
    model = RouteDiffusionDenoiser(hidden_dim=32, num_layers=3)
    logits = model(coords, noisy, t)
    assert logits.shape == (b, n, n)
    assert torch.isfinite(logits).all()


def test_output_is_symmetric_with_zero_diagonal():
    n = 14
    coords, clean = _clean_edge_set(n)
    sched = make_schedule(50)
    noisy = corrupt(clean, 25, sched)
    model = RouteDiffusionDenoiser(hidden_dim=32, num_layers=2)
    logits = model(coords, noisy, 25)
    assert torch.allclose(logits, logits.transpose(-1, -2), atol=1e-6)
    assert torch.allclose(torch.diagonal(logits), torch.zeros(n), atol=1e-6)


def test_sinusoidal_embedding_shape():
    emb = sinusoidal_timestep_embedding(torch.arange(5), 16)
    assert emb.shape == (5, 16)
    assert torch.isfinite(emb).all()


def test_distinct_timesteps_have_distinct_embeddings():
    # Task 3.2: every timestep must map to a unique embedding vector.
    emb = sinusoidal_timestep_embedding(torch.arange(51), 32)
    dist = torch.cdist(emb, emb)
    dist.fill_diagonal_(float("inf"))
    assert dist.min().item() > 1e-3


def test_output_changes_when_only_timestep_changes():
    # Task 3.2 Definition of Done: with all other inputs fixed, varying t alone
    # must change the model output.
    n = 12
    coords, clean = _clean_edge_set(n)
    sched = make_schedule(50)
    noisy = corrupt(clean, 25, sched)
    model = RouteDiffusionDenoiser(hidden_dim=32, num_layers=3).eval()
    with torch.no_grad():
        early = model(coords, noisy, 1)
        late = model(coords, noisy, 49)
    assert not torch.allclose(early, late, atol=1e-4)
    assert (early - late).abs().mean().item() > 1e-4


def test_per_item_timestep_conditioning_in_batch():
    # Two identical instances differing only in their timestep must produce
    # different per-item outputs.
    n = 10
    coords, clean = _clean_edge_set(n)
    sched = make_schedule(50)
    noisy = corrupt(clean, 30, sched)
    coords_b = torch.stack([coords, coords])
    noisy_b = torch.stack([noisy, noisy])
    t = torch.tensor([2, 48])
    model = RouteDiffusionDenoiser(hidden_dim=32, num_layers=3).eval()
    with torch.no_grad():
        out = model(coords_b, noisy_b, t)
    assert not torch.allclose(out[0], out[1], atol=1e-4)


def test_forward_feeds_diffusion_loss():
    n = 12
    coords, clean = _clean_edge_set(n)
    sched = make_schedule(50)
    noisy, target = make_training_pair(clean, 15, sched)
    model = RouteDiffusionDenoiser(hidden_dim=32, num_layers=3)
    logits = model(coords, noisy, 15)
    loss = edge_prediction_loss(logits, target)
    assert loss.ndim == 0 and torch.isfinite(loss)


def test_one_gradient_step_runs():
    n = 12
    coords, clean = _clean_edge_set(n)
    sched = make_schedule(50)
    noisy, target = make_training_pair(clean, 20, sched)
    model = RouteDiffusionDenoiser(hidden_dim=32, num_layers=3)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    logits = model(coords, noisy, 20)
    loss = edge_prediction_loss(logits, target)
    opt.zero_grad()
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert len(grads) > 0
    assert all(torch.isfinite(g).all() for g in grads)
    opt.step()
