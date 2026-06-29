"""Tests for reverse sampling and tour decoding (Task 5.1)."""

from __future__ import annotations

import json

import torch

from data.generator import generate_instances
from routediff.diffusion import make_schedule
from routediff.graph import tour_length, tour_to_edge_set
from routediff.inference import (
    build_timeline,
    export_timeline,
    greedy_decode,
    reverse_posterior_present_prob,
    reverse_sample,
)
from routediff.model import RouteDiffusionDenoiser
from routediff.solvers import is_valid_tour, nearest_neighbor
from routediff.train import TrainConfig, train


def _mean_random_tour_length(coords: torch.Tensor, trials: int = 50, seed: int = 0) -> float:
    n = coords.shape[0]
    g = torch.Generator().manual_seed(seed)
    total = 0.0
    for _ in range(trials):
        perm = torch.randperm(n, generator=g)
        total += float(tour_length(coords, perm))
    return total / trials


def test_posterior_probs_in_range():
    """Posterior present-probabilities are valid probabilities in [0, 1]."""
    schedule = make_schedule(50)
    n = 12
    e_t = (torch.rand(n, n) < 0.5).to(torch.float32)
    e_t = torch.triu(e_t, diagonal=1)
    e_t = e_t + e_t.t()
    p_e0 = torch.rand(n, n)
    for t in (1, 10, 25, 50):
        p_prev = reverse_posterior_present_prob(e_t, p_e0, t, schedule)
        assert torch.all(p_prev >= -1e-6)
        assert torch.all(p_prev <= 1.0 + 1e-6)


def test_posterior_at_t1_recovers_model_prediction():
    """At t=1 the next state is e_0, so the present-prob equals the model's p_e0."""
    schedule = make_schedule(40)
    n = 8
    e_t = (torch.rand(n, n) < 0.5).to(torch.float32)
    e_t = torch.triu(e_t, diagonal=1)
    e_t = e_t + e_t.t()
    p_e0 = torch.rand(n, n)
    p_prev = reverse_posterior_present_prob(e_t, p_e0, t=1, schedule=schedule)
    mask = torch.triu(torch.ones(n, n), diagonal=1).bool()
    assert torch.allclose(p_prev[mask], p_e0[mask], atol=1e-5)


def test_greedy_decode_produces_valid_tour():
    """Greedy decoding of arbitrary scores yields a valid Hamiltonian tour."""
    torch.manual_seed(0)
    for n in (3, 5, 10, 20):
        scores = torch.rand(n, n)
        scores = scores + scores.t()
        scores.fill_diagonal_(0.0)
        tour = greedy_decode(scores)
        assert is_valid_tour(tour, n)


def test_greedy_decode_recovers_clean_tour():
    """When scores mark exactly a tour's edges, decoding recovers that tour."""
    coords = generate_instances(num_cities=12, num_instances=1, seed=1)[0]
    clean = nearest_neighbor(coords)
    edge_set = tour_to_edge_set(clean)
    tour = greedy_decode(edge_set)
    assert is_valid_tour(tour, 12)
    # Same cycle (possibly rotated/reflected) -> identical length.
    assert abs(float(tour_length(coords, tour)) - float(tour_length(coords, clean))) < 1e-5


def test_reverse_sample_returns_valid_tour_untrained():
    """End-to-end reverse sampling runs and returns a valid tour shape/length."""
    coords = generate_instances(num_cities=10, num_instances=1, seed=2)[0]
    schedule = make_schedule(20)
    model = RouteDiffusionDenoiser(hidden_dim=16, num_layers=2)
    generator = torch.Generator().manual_seed(3)
    result = reverse_sample(model, coords, schedule, generator=generator)
    assert is_valid_tour(result.tour, 10)
    assert result.length > 0.0
    assert result.edge_probs.shape == (10, 10)


def test_reverse_sample_records_history():
    """History capture yields one snapshot per step plus the final state."""
    coords = generate_instances(num_cities=8, num_instances=1, seed=4)[0]
    schedule = make_schedule(15)
    model = RouteDiffusionDenoiser(hidden_dim=16, num_layers=2)
    result = reverse_sample(model, coords, schedule, record_history=True)
    assert len(result.history) == schedule.num_timesteps + 1
    assert result.history[0]["t"] == schedule.num_timesteps
    assert result.history[-1]["t"] == 0


def test_build_timeline_structure():
    """The timeline captures every step plus a clean final frame for one instance."""
    coords = generate_instances(num_cities=9, num_instances=1, seed=5)[0]
    schedule = make_schedule(12)
    model = RouteDiffusionDenoiser(hidden_dim=16, num_layers=2)
    result = reverse_sample(model, coords, schedule, record_history=True)

    timeline = build_timeline(coords, result)
    assert timeline["num_cities"] == 9
    assert len(timeline["coords"]) == 9
    # One frame per denoising step (T+1 snapshots) plus the terminal clean frame.
    assert timeline["num_steps"] == schedule.num_timesteps + 2
    assert len(timeline["steps"]) == timeline["num_steps"]
    assert timeline["steps"][0]["t"] == schedule.num_timesteps
    assert timeline["steps"][-1].get("final") is True
    # The final frame is a valid Hamiltonian cycle: exactly N edges.
    assert len(timeline["steps"][-1]["edges"]) == 9
    assert len(timeline["tour"]) == 9
    assert is_valid_tour(torch.tensor(timeline["tour"]), 9)
    # Every edge references valid, distinct city indices.
    for step in timeline["steps"]:
        for i, j in step["edges"]:
            assert 0 <= i < 9 and 0 <= j < 9 and i != j


def test_build_timeline_requires_history():
    """Building a timeline without recorded history is an error."""
    coords = generate_instances(num_cities=6, num_instances=1, seed=6)[0]
    schedule = make_schedule(10)
    model = RouteDiffusionDenoiser(hidden_dim=8, num_layers=2)
    result = reverse_sample(model, coords, schedule, record_history=False)
    try:
        build_timeline(coords, result)
        raised = False
    except ValueError:
        raised = True
    assert raised


def test_export_timeline_writes_valid_json(tmp_path):
    """Exported JSON round-trips and matches the in-memory timeline."""
    coords = generate_instances(num_cities=8, num_instances=1, seed=7)[0]
    schedule = make_schedule(10)
    model = RouteDiffusionDenoiser(hidden_dim=16, num_layers=2)
    result = reverse_sample(model, coords, schedule, record_history=True)

    out = tmp_path / "timeline.json"
    timeline = export_timeline(coords, result, str(out))
    assert out.exists()

    with open(out, encoding="utf-8") as f:
        loaded = json.load(f)
    assert loaded == timeline
    assert loaded["num_steps"] == len(loaded["steps"])


def test_trained_model_beats_random_tour():
    """A briefly trained model produces tours better than the random average."""
    cfg = TrainConfig(
        num_cities=10,
        num_instances=48,
        num_epochs=25,
        batch_size=16,
        num_timesteps=30,
        hidden_dim=32,
        num_layers=3,
        solver_method="nn2opt",
        seed=0,
        checkpoint_dir="checkpoints/test_infer",
        checkpoint_every=25,
        log_path="logs/test_infer.log",
    )
    result = train(cfg)
    model = result.model

    schedule = make_schedule(cfg.num_timesteps)
    coords = generate_instances(num_cities=cfg.num_cities, num_instances=1, seed=99)[0]
    random_mean = _mean_random_tour_length(coords)

    # Average a few stochastic samples to reduce variance.
    lengths = []
    for s in range(5):
        g = torch.Generator().manual_seed(s)
        lengths.append(reverse_sample(model, coords, schedule, generator=g).length)
    best = min(lengths)
    assert best < random_mean
