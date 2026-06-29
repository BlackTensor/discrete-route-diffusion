"""Tests for the training loop (Task 4.1)."""

import os

from routediff.train import (
    TrainConfig,
    build_dataset,
    get_logger,
    overfit_sanity_check,
    train,
)


def _tiny_config(tmp_path) -> TrainConfig:
    return TrainConfig(
        num_cities=10,
        num_instances=24,
        num_epochs=6,
        batch_size=8,
        num_timesteps=40,
        hidden_dim=32,
        num_layers=3,
        solver_method="nn2opt",
        seed=0,
        checkpoint_dir=str(tmp_path / "checkpoints"),
        checkpoint_every=3,
        log_path=str(tmp_path / "logs" / "train.log"),
    )


def test_build_dataset_shapes():
    cfg = TrainConfig(num_cities=8, num_instances=5, solver_method="nn2opt")
    coords, edge_sets = build_dataset(cfg)
    assert coords.shape == (5, 8, 2)
    assert edge_sets.shape == (5, 8, 8)
    # Each clean tour is a Hamiltonian cycle: every node has degree two.
    degrees = edge_sets.sum(dim=-1)
    assert (degrees == 2).all()
    # Symmetric with zero diagonal.
    assert (edge_sets == edge_sets.transpose(-1, -2)).all()
    assert (edge_sets.diagonal(dim1=-2, dim2=-1) == 0).all()


def test_train_loss_decreases_and_checkpoints(tmp_path):
    cfg = _tiny_config(tmp_path)
    result = train(cfg)

    # Loss should trend down: final epoch lower than the first.
    assert len(result.losses) == cfg.num_epochs
    assert result.losses[-1] < result.losses[0]

    # A checkpoint file was written.
    assert result.final_checkpoint is not None
    assert os.path.exists(result.final_checkpoint)

    # The log file was created and has content.
    assert os.path.exists(cfg.log_path)
    assert os.path.getsize(cfg.log_path) > 0


def test_checkpoint_is_loadable(tmp_path):
    import torch

    from routediff.model import RouteDiffusionDenoiser

    cfg = _tiny_config(tmp_path)
    cfg.num_epochs = 3
    result = train(cfg)

    ckpt = torch.load(result.final_checkpoint, weights_only=False)
    assert ckpt["epoch"] == cfg.num_epochs
    assert "model_state" in ckpt
    model = RouteDiffusionDenoiser(hidden_dim=cfg.hidden_dim, num_layers=cfg.num_layers)
    model.load_state_dict(ckpt["model_state"])


def test_overfit_drives_loss_near_zero(tmp_path):
    result = overfit_sanity_check(
        num_instances=5,
        num_cities=10,
        num_steps=400,
        num_timesteps=40,
        seed=0,
        solver_method="nn2opt",
        log_path=str(tmp_path / "overfit.log"),
    )
    assert len(result.losses) == 400
    # The fixed batch should be memorized: loss collapses toward zero.
    assert result.losses[-1] < 0.01
    assert result.losses[-1] < result.losses[0]


def test_logger_is_idempotent(tmp_path):
    path = str(tmp_path / "log.txt")
    a = get_logger(path)
    b = get_logger(path)
    assert a is b
    assert len(a.handlers) == 2
