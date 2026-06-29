"""Training loop for the diffusion denoiser.

Samples instances, corrupts tours, predicts, and backpropagates. Logs loss and
checkpoints the model.

Pipeline
--------
A fixed dataset of ``(coords, clean_edge_set)`` pairs is built once by generating
synthetic TSP instances (:mod:`data.generator`) and solving each with a baseline
solver (:mod:`routediff.solvers`) to obtain the clean target tour. Building the
dataset once and reusing it across epochs keeps the (relatively slow) solver off
the hot training path.

Each training step then:

1. draws a minibatch of clean edge sets,
2. samples a timestep ``t`` per item (:func:`routediff.diffusion.sample_timesteps`),
3. corrupts the clean edge set to ``e_t`` (the ``x_0`` parameterization target is
   the clean edge set itself),
4. predicts per-edge clean-tour logits with the GNN denoiser, and
5. minimizes the edge-wise binary cross-entropy and backpropagates.

The clean tour is sparse -- only ``N`` of the ``N (N - 1) / 2`` undirected edges
are present -- so the positive (present-edge) class is up-weighted by default via
``pos_weight`` to keep the model from collapsing to "predict absent".

Loss is logged to the console and to a log file, and the model is checkpointed
periodically.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch

from data.generator import generate_instances
from routediff.diffusion import (
    DiffusionSchedule,
    edge_prediction_loss,
    make_schedule,
    make_training_pair,
    sample_timesteps,
)
from routediff.graph import tour_to_edge_set
from routediff.model import RouteDiffusionDenoiser
from routediff.solvers import solve


@dataclass
class TrainConfig:
    """Hyperparameters and run settings for training.

    Attributes:
        num_cities: Number of cities ``N`` per instance.
        num_instances: Size of the (fixed) training dataset.
        num_epochs: Number of passes over the dataset.
        batch_size: Minibatch size.
        learning_rate: Adam learning rate.
        weight_decay: Adam weight decay.
        num_timesteps: Diffusion horizon ``T``.
        schedule_kind: ``"cosine"`` or ``"linear"`` noise schedule.
        hidden_dim: Model width ``H``.
        num_layers: Number of message-passing layers.
        solver_method: Baseline solver for clean labels (see ``solvers.solve``).
        pos_weight: Positive-class weight; ``None`` derives it from sparsity,
            a float sets it explicitly, ``0`` disables it.
        seed: Master seed for dataset generation and corruption.
        grad_clip: Optional gradient-norm clip; ``None`` disables clipping.
        checkpoint_dir: Directory for checkpoint files.
        checkpoint_every: Save a checkpoint every this many epochs (and at end).
        log_path: File the per-epoch loss log is appended to.
        device: Torch device string.
    """

    num_cities: int = 20
    num_instances: int = 512
    num_epochs: int = 50
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    num_timesteps: int = 100
    schedule_kind: str = "cosine"
    hidden_dim: int = 64
    num_layers: int = 4
    solver_method: str = "auto"
    pos_weight: Optional[float] = None
    seed: int = 0
    grad_clip: Optional[float] = 1.0
    checkpoint_dir: str = "checkpoints"
    checkpoint_every: int = 10
    log_path: str = "logs/train.log"
    device: str = "cpu"


def get_logger(log_path: str) -> logging.Logger:
    """Return a logger that writes to the console and to ``log_path``.

    Args:
        log_path: File the log is appended to; parent directories are created.

    Returns:
        A configured :class:`logging.Logger` (idempotent across calls per path).
    """
    logger = logging.getLogger(f"routediff.train:{log_path}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if logger.handlers:  # already configured for this path
        return logger

    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%Y-%m-%d %H:%M:%S")

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(fmt)
    logger.addHandler(stream_handler)
    return logger


def build_dataset(config: TrainConfig) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate instances and solve them for clean target edge sets.

    Args:
        config: Run configuration (uses ``num_cities``, ``num_instances``,
            ``seed``, ``solver_method``, ``device``).

    Returns:
        Tuple ``(coords, clean_edge_sets)`` of shapes
        ``(M, N, 2)`` and ``(M, N, N)`` on ``config.device``.
    """
    device = torch.device(config.device)
    coords = generate_instances(
        num_cities=config.num_cities,
        num_instances=config.num_instances,
        seed=config.seed,
    )

    edge_sets = torch.empty(
        (config.num_instances, config.num_cities, config.num_cities), dtype=torch.float32
    )
    for i in range(config.num_instances):
        tour = solve(coords[i], method=config.solver_method)
        edge_sets[i] = tour_to_edge_set(tour, config.num_cities)

    return coords.to(device), edge_sets.to(device)


def _resolve_pos_weight(config: TrainConfig) -> Optional[float]:
    """Resolve the positive-class weight from the config.

    ``None`` derives the weight from edge sparsity (negatives / positives for a
    Hamiltonian cycle: ``(N - 3) / 2``); ``0`` disables weighting; any other
    float is used as-is.
    """
    if config.pos_weight is None:
        n = config.num_cities
        num_pos = n  # a Hamiltonian cycle has exactly N undirected edges
        num_total = n * (n - 1) // 2
        num_neg = num_total - num_pos
        return float(num_neg) / float(max(num_pos, 1))
    if config.pos_weight == 0:
        return None
    return float(config.pos_weight)


def save_checkpoint(
    path: str,
    model: RouteDiffusionDenoiser,
    optimizer: torch.optim.Optimizer,
    schedule: DiffusionSchedule,
    config: TrainConfig,
    epoch: int,
    loss: float,
) -> None:
    """Write a checkpoint containing model, optimizer, schedule, and metadata."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "config": config.__dict__,
            "schedule": {
                "num_timesteps": schedule.num_timesteps,
                "betas": schedule.betas,
                "alphas": schedule.alphas,
                "alpha_bars": schedule.alpha_bars,
            },
            "epoch": epoch,
            "loss": loss,
        },
        path,
    )


@dataclass
class TrainResult:
    """Outcome of a training run.

    Attributes:
        losses: Mean training loss per epoch.
        final_checkpoint: Path to the last checkpoint written.
        model: The trained model (left on ``config.device``).
    """

    losses: list[float] = field(default_factory=list)
    final_checkpoint: Optional[str] = None
    model: Optional[RouteDiffusionDenoiser] = None


def train(config: TrainConfig) -> TrainResult:
    """Run the full training loop.

    Args:
        config: Hyperparameters and run settings.

    Returns:
        A :class:`TrainResult` with per-epoch losses and the final checkpoint
        path.
    """
    device = torch.device(config.device)
    logger = get_logger(config.log_path)
    logger.info(
        "starting training: N=%d instances=%d epochs=%d batch=%d T=%d device=%s",
        config.num_cities,
        config.num_instances,
        config.num_epochs,
        config.batch_size,
        config.num_timesteps,
        config.device,
    )

    # Reproducible corruption and shuffling, independent of the global RNG.
    generator = torch.Generator(device="cpu")
    generator.manual_seed(config.seed)

    coords, clean_edge_sets = build_dataset(config)
    logger.info("dataset built: %d instances", coords.shape[0])

    schedule = make_schedule(config.num_timesteps, kind=config.schedule_kind).to(device)
    model = RouteDiffusionDenoiser(
        hidden_dim=config.hidden_dim, num_layers=config.num_layers
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    pos_weight = _resolve_pos_weight(config)
    num_params = sum(p.numel() for p in model.parameters())
    logger.info("model ready: %d parameters, pos_weight=%s", num_params, pos_weight)

    n_data = coords.shape[0]
    result = TrainResult(model=model)
    ckpt_dir = Path(config.checkpoint_dir)

    for epoch in range(1, config.num_epochs + 1):
        model.train()
        perm = torch.randperm(n_data, generator=generator)
        epoch_loss = 0.0
        num_batches = 0
        start = time.time()

        for begin in range(0, n_data, config.batch_size):
            idx = perm[begin : begin + config.batch_size]
            coords_b = coords[idx]
            clean_b = clean_edge_sets[idx]

            t = sample_timesteps(idx.numel(), schedule, generator=generator).to(device)
            noisy, target = make_training_pair(clean_b, t, schedule, generator=generator)

            logits = model(coords_b, noisy, t)
            loss = edge_prediction_loss(logits, target, pos_weight=pos_weight)

            optimizer.zero_grad()
            loss.backward()
            if config.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()

            epoch_loss += float(loss.item())
            num_batches += 1

        mean_loss = epoch_loss / max(num_batches, 1)
        result.losses.append(mean_loss)
        logger.info(
            "epoch %d/%d  loss=%.5f  (%.1fs)",
            epoch,
            config.num_epochs,
            mean_loss,
            time.time() - start,
        )

        if epoch % config.checkpoint_every == 0 or epoch == config.num_epochs:
            ckpt_path = str(ckpt_dir / f"denoiser_epoch{epoch:04d}.pt")
            save_checkpoint(
                ckpt_path, model, optimizer, schedule, config, epoch, mean_loss
            )
            result.final_checkpoint = ckpt_path
            logger.info("checkpoint saved: %s", ckpt_path)

    logger.info("training complete: final loss=%.5f", result.losses[-1] if result.losses else float("nan"))
    return result


@dataclass
class OverfitResult:
    """Outcome of a sanity overfit run.

    Attributes:
        losses: Loss at each optimization step.
        model: The overfit model.
    """

    losses: list[float] = field(default_factory=list)
    model: Optional[RouteDiffusionDenoiser] = None


def overfit_sanity_check(
    num_instances: int = 5,
    num_cities: int = 10,
    num_steps: int = 400,
    num_timesteps: int = 40,
    hidden_dim: int = 64,
    num_layers: int = 4,
    learning_rate: float = 1e-3,
    solver_method: str = "auto",
    seed: int = 0,
    device: str = "cpu",
    log_every: int = 50,
    log_path: str = "logs/overfit.log",
) -> OverfitResult:
    """Overfit a fixed tiny batch to prove the pipeline can drive loss near zero.

    This is the Task 4.2 sanity check. Unlike :func:`train`, the corruption is
    sampled *once* and held fixed, so the model repeatedly faces a single
    deterministic ``(coords, noisy_edges, t) -> clean_edges`` batch. With the
    model, loss, and gradient path wired correctly the loss should collapse
    toward zero, confirming the pipeline can learn. ``pos_weight`` is disabled so
    a perfect fit reads as a clean near-zero loss.

    Args:
        num_instances: Size of the tiny fixed batch (e.g. 5).
        num_cities: Number of cities ``N`` per instance.
        num_steps: Number of optimization steps on the fixed batch.
        num_timesteps: Diffusion horizon ``T`` for the (one-shot) corruption.
        hidden_dim: Model width ``H``.
        num_layers: Number of message-passing layers.
        learning_rate: Adam learning rate.
        solver_method: Baseline solver for the clean labels.
        seed: Master seed for data, corruption, and init.
        device: Torch device string.
        log_every: Log the loss every this many steps.
        log_path: File the loss log is appended to.

    Returns:
        An :class:`OverfitResult` with the per-step losses and the model.
    """
    device_t = torch.device(device)
    logger = get_logger(log_path)
    torch.manual_seed(seed)

    data_cfg = TrainConfig(
        num_cities=num_cities,
        num_instances=num_instances,
        solver_method=solver_method,
        seed=seed,
        device=device,
    )
    coords, clean = build_dataset(data_cfg)

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    schedule = make_schedule(num_timesteps).to(device_t)

    # Sample the corruption once and hold it fixed for the whole run.
    t = sample_timesteps(num_instances, schedule, generator=generator).to(device_t)
    noisy, target = make_training_pair(clean, t, schedule, generator=generator)

    model = RouteDiffusionDenoiser(hidden_dim=hidden_dim, num_layers=num_layers).to(device_t)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    logger.info(
        "overfit sanity check: instances=%d N=%d steps=%d", num_instances, num_cities, num_steps
    )
    result = OverfitResult(model=model)
    model.train()
    for step in range(1, num_steps + 1):
        logits = model(coords, noisy, t)
        loss = edge_prediction_loss(logits, target, pos_weight=None)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        result.losses.append(float(loss.item()))
        if step % log_every == 0 or step == num_steps:
            logger.info("step %d/%d  loss=%.6f", step, num_steps, result.losses[-1])

    logger.info("overfit complete: final loss=%.6f", result.losses[-1])
    return result


if __name__ == "__main__":
    cfg = TrainConfig()
    train(cfg)
