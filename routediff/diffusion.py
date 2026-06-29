"""Discrete diffusion process over the tour edge set.

Forward (noising) corruption using a D3PM-style discrete categorical process.

Representation
--------------
A tour is encoded as a symmetric binary edge set (adjacency matrix) of shape
``(N, N)`` with zero diagonal (see ``routediff.graph``). We treat each of the
``N (N - 1) / 2`` *undirected* edges as an independent binary categorical
variable with two states::

    state 0 = edge absent      state 1 = edge present

The forward process corrupts these binary variables, not the city order, so a
clean Hamiltonian cycle is gradually dissolved into an unstructured random graph.

Forward process (D3PM, uniform / symmetric kernel)
--------------------------------------------------
Following Austin et al., "Structured Denoising Diffusion Models in Discrete
State-Spaces" (D3PM), each edge evolves under a Markov chain whose single-step
transition matrix over the two states is the uniform kernel

    Q_t = (1 - beta_t) I + beta_t * (1 / K) * 1 1^T ,    K = 2

i.e. with probability ``beta_t`` an edge resamples its state uniformly, otherwise
it is left unchanged. ``beta_t`` is the noise schedule and grows with ``t``.

Because the kernel is uniform, the ``t``-step marginal has a closed form. With
the cumulative survival probability

    alpha_bar_t = prod_{s=1..t} (1 - beta_s) ,

an edge keeps its clean state with probability ``alpha_bar_t`` and is otherwise
resampled uniformly. The marginal probability that an edge is *present* given its
clean value ``e0 in {0, 1}`` is therefore

    q(e_t = 1 | e_0) = 0.5 + (e_0 - 0.5) * alpha_bar_t .

This is the corruption rule implemented below. Limits:

- ``t = 0``  : ``alpha_bar_0 = 1`` -> edge set returned unchanged (clean).
- ``t = T``  : ``alpha_bar_T -> 0`` -> every edge present with probability 1/2,
  i.e. an Erdos-Renyi ``G(N, 1/2)`` graph. This uniform Bernoulli distribution is
  the stationary distribution of the chain and the prior we sample from at
  inference time.

Intermediate ``t`` interpolate smoothly, and the expected overlap with the clean
tour decreases monotonically in ``t``.

Reverse process (denoising target and loss)
--------------------------------------------
We use the ``x_0`` parameterization of D3PM: at every timestep the denoiser is
trained to reconstruct the *clean* edge set directly rather than the single-step
posterior. Concretely, given a noised edge set ``e_t`` and the timestep ``t``,
the model outputs one logit per undirected edge, interpreted as

    p_theta(e_0[i, j] = present | e_t, t) = sigmoid(logit[i, j]) .

The training target is therefore the clean edge set ``e_0`` itself (a binary
label per edge), and the loss is the binary cross-entropy between the predicted
edge logits and those labels, averaged over the ``N (N - 1) / 2`` undirected
edges (the strict upper triangle, so each edge is counted once and the diagonal
is ignored). Because the clean tour is sparse -- only ``N`` of the possible edges
are present -- an optional ``pos_weight`` can up-weight the present class.

A training step is: sample ``t``, corrupt the clean tour to ``e_t`` via the
forward process above, feed ``(e_t, t)`` to the model, and minimize this
edge-wise cross-entropy against ``e_0``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Union

import torch
import torch.nn.functional as F

IntOrTensor = Union[int, torch.Tensor]


@dataclass
class DiffusionSchedule:
    """A discrete noise schedule over ``T`` timesteps.

    All arrays are indexed by timestep ``t`` in ``[0, T]`` and have length
    ``T + 1``. Index ``0`` is the clean state: ``betas[0] = 0`` and
    ``alpha_bars[0] = 1``.

    Attributes:
        num_timesteps: The number of noising steps ``T``.
        betas: Per-step resample probability ``beta_t``, shape ``(T + 1,)``.
        alphas: ``1 - betas``, shape ``(T + 1,)``.
        alpha_bars: Cumulative survival ``prod (1 - beta_s)``, shape ``(T + 1,)``.
    """

    num_timesteps: int
    betas: torch.Tensor
    alphas: torch.Tensor
    alpha_bars: torch.Tensor

    def to(self, device: torch.device) -> "DiffusionSchedule":
        """Return a copy of the schedule with all tensors moved to ``device``."""
        return DiffusionSchedule(
            num_timesteps=self.num_timesteps,
            betas=self.betas.to(device),
            alphas=self.alphas.to(device),
            alpha_bars=self.alpha_bars.to(device),
        )


def make_schedule(
    num_timesteps: int,
    kind: str = "cosine",
    beta_start: float = 1e-4,
    beta_end: float = 0.2,
    cosine_s: float = 0.008,
    max_beta: float = 0.999,
) -> DiffusionSchedule:
    """Build a :class:`DiffusionSchedule`.

    Args:
        num_timesteps: Number of noising steps ``T``. Must be >= 1.
        kind: ``"cosine"`` (default) or ``"linear"``.
        beta_start: Initial ``beta`` for the linear schedule.
        beta_end: Final ``beta`` for the linear schedule.
        cosine_s: Small offset ``s`` for the cosine schedule (Nichol & Dhariwal).
        max_beta: Upper clamp on any single-step ``beta`` for numerical safety.

    Returns:
        A schedule whose ``alpha_bars`` decrease monotonically from 1 toward 0.
    """
    if num_timesteps < 1:
        raise ValueError(f"num_timesteps must be >= 1, got {num_timesteps}")
    if kind not in ("cosine", "linear"):
        raise ValueError(f"kind must be 'cosine' or 'linear', got {kind!r}")

    t_grid = torch.arange(num_timesteps + 1, dtype=torch.float64)

    if kind == "cosine":
        # Cosine schedule on the cumulative product, then derive per-step betas.
        f = torch.cos(((t_grid / num_timesteps) + cosine_s) / (1.0 + cosine_s) * (math.pi / 2.0)) ** 2
        alpha_bar_target = f / f[0]
        core = 1.0 - alpha_bar_target[1:] / alpha_bar_target[:-1]
        betas_core = torch.clamp(core, min=0.0, max=max_beta)
    else:  # linear
        betas_core = torch.linspace(beta_start, beta_end, num_timesteps, dtype=torch.float64)
        betas_core = torch.clamp(betas_core, min=0.0, max=max_beta)

    betas = torch.cat([torch.zeros(1, dtype=torch.float64), betas_core])
    alphas = 1.0 - betas
    # Recompute the cumulative product from the (possibly clamped) betas so the
    # schedule is internally consistent.
    alpha_bars = torch.cumprod(alphas, dim=0)

    return DiffusionSchedule(
        num_timesteps=num_timesteps,
        betas=betas.to(torch.float32),
        alphas=alphas.to(torch.float32),
        alpha_bars=alpha_bars.to(torch.float32),
    )


def _alpha_bar_for(
    schedule: DiffusionSchedule, t: IntOrTensor, batched: bool, batch_size: int
) -> torch.Tensor:
    """Gather ``alpha_bar_t`` and shape it for broadcasting against an edge set."""
    ab = schedule.alpha_bars
    t_tensor = torch.as_tensor(t, device=ab.device)
    if torch.any(t_tensor < 0) or torch.any(t_tensor > schedule.num_timesteps):
        raise ValueError(
            f"timestep t must be in [0, {schedule.num_timesteps}], got {t_tensor.tolist()}"
        )

    if not batched:
        if t_tensor.dim() != 0:
            raise ValueError("t must be a scalar when edge_set is unbatched (N, N)")
        return ab[t_tensor]  # scalar

    if t_tensor.dim() == 0:
        t_tensor = t_tensor.expand(batch_size)
    if t_tensor.shape != (batch_size,):
        raise ValueError(
            f"t must be a scalar or shape ({batch_size},) for a batch, got {tuple(t_tensor.shape)}"
        )
    return ab[t_tensor].view(batch_size, 1, 1)


def q_present_probs(
    edge_set: torch.Tensor, t: IntOrTensor, schedule: DiffusionSchedule
) -> torch.Tensor:
    """Marginal probability each edge is *present* at timestep ``t``.

    Implements ``q(e_t = 1 | e_0) = 0.5 + (e_0 - 0.5) * alpha_bar_t`` elementwise.

    Args:
        edge_set: Binary tensor of shape ``(N, N)`` or ``(B, N, N)``.
        t: Timestep in ``[0, T]``; scalar, or shape ``(B,)`` for a batch.
        schedule: The noise schedule.

    Returns:
        Float tensor of present-probabilities, same shape as ``edge_set``.
    """
    if edge_set.dim() not in (2, 3):
        raise ValueError(f"edge_set must be (N, N) or (B, N, N), got {tuple(edge_set.shape)}")
    batched = edge_set.dim() == 3
    e0 = edge_set.to(torch.float32)
    alpha_bar = _alpha_bar_for(schedule, t, batched, e0.shape[0] if batched else 0)
    return 0.5 + (e0 - 0.5) * alpha_bar


def corrupt(
    edge_set: torch.Tensor,
    t: IntOrTensor,
    schedule: DiffusionSchedule,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Sample a noised edge set ``e_t ~ q(e_t | e_0)`` at timestep ``t``.

    Sampling is done on the upper triangle and mirrored, so the result is a valid
    symmetric adjacency matrix with zero diagonal. At ``t = 0`` the clean edge set
    is returned unchanged; as ``t`` grows the edges approach independent fair
    coin flips (the ``G(N, 1/2)`` prior).

    Args:
        edge_set: Clean binary tensor of shape ``(N, N)`` or ``(B, N, N)``.
        t: Timestep in ``[0, T]``; scalar, or shape ``(B,)`` for a batch.
        schedule: The noise schedule.
        generator: Optional ``torch.Generator`` for reproducible sampling.

    Returns:
        Noised binary float tensor of the same shape as ``edge_set``.
    """
    p_present = q_present_probs(edge_set, t, schedule)
    n = edge_set.shape[-1]
    device = edge_set.device

    rand = torch.rand(p_present.shape, generator=generator, device=device)
    upper = torch.triu(torch.ones(n, n, device=device), diagonal=1)
    decision = (rand < p_present).to(torch.float32) * upper
    noisy = decision + decision.transpose(-1, -2)
    return noisy


def sample_prior(
    num_cities: int,
    batch_size: Optional[int] = None,
    generator: Optional[torch.Generator] = None,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Sample from the stationary prior: each edge present with probability 1/2.

    This is the ``t = T`` limit of the forward process and the starting point for
    reverse sampling at inference time.

    Args:
        num_cities: Number of cities ``N``.
        batch_size: If given, return a batch of shape ``(batch_size, N, N)``;
            otherwise a single ``(N, N)`` matrix.
        generator: Optional ``torch.Generator`` for reproducible sampling.
        device: Optional device for the returned tensor.

    Returns:
        Symmetric binary float adjacency tensor with zero diagonal.
    """
    if num_cities < 1:
        raise ValueError(f"num_cities must be >= 1, got {num_cities}")
    shape = (num_cities, num_cities) if batch_size is None else (batch_size, num_cities, num_cities)
    rand = torch.rand(shape, generator=generator, device=device)
    upper = torch.triu(torch.ones(num_cities, num_cities, device=device), diagonal=1)
    decision = (rand < 0.5).to(torch.float32) * upper
    return decision + decision.transpose(-1, -2)


def fraction_edges_retained(clean: torch.Tensor, noisy: torch.Tensor) -> torch.Tensor:
    """Fraction of the clean tour's present edges still present in ``noisy``.

    A diagnostic for the forward process: it should decrease monotonically (in
    expectation) as the corruption timestep increases. Operates on the upper
    triangle so each undirected edge is counted once.

    Args:
        clean: Clean binary edge set, shape ``(N, N)`` or ``(B, N, N)``.
        noisy: Corrupted binary edge set, same shape as ``clean``.

    Returns:
        Scalar tensor (unbatched) or shape ``(B,)`` tensor of retained fractions.
    """
    if clean.shape != noisy.shape:
        raise ValueError(f"shape mismatch: clean {tuple(clean.shape)} vs noisy {tuple(noisy.shape)}")
    if clean.dim() not in (2, 3):
        raise ValueError(f"edge sets must be (N, N) or (B, N, N), got {tuple(clean.shape)}")

    n = clean.shape[-1]
    upper = torch.triu(torch.ones(n, n, device=clean.device), diagonal=1)
    clean_u = (clean > 0.5).to(torch.float32) * upper
    noisy_u = (noisy > 0.5).to(torch.float32) * upper

    retained = (clean_u * noisy_u).sum(dim=(-2, -1))
    total = clean_u.sum(dim=(-2, -1)).clamp(min=1.0)
    return retained / total


def sample_timesteps(
    batch_size: int,
    schedule: DiffusionSchedule,
    generator: Optional[torch.Generator] = None,
    device: Optional[torch.device] = None,
    include_zero: bool = False,
) -> torch.Tensor:
    """Sample training timesteps uniformly.

    Args:
        batch_size: Number of timesteps to draw.
        schedule: The noise schedule (provides ``T``).
        generator: Optional ``torch.Generator`` for reproducible sampling.
        device: Optional device for the returned tensor.
        include_zero: If ``False`` (default) sample from ``{1, ..., T}`` so every
            training example carries some noise; if ``True`` include ``t = 0``.

    Returns:
        ``long`` tensor of shape ``(batch_size,)`` with values in ``[low, T]``.
    """
    low = 0 if include_zero else 1
    return torch.randint(
        low, schedule.num_timesteps + 1, (batch_size,), generator=generator, device=device
    )


def make_training_pair(
    clean_edge_set: torch.Tensor,
    t: IntOrTensor,
    schedule: DiffusionSchedule,
    generator: Optional[torch.Generator] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a ``(noisy_input, target)`` pair for the reverse model.

    The noisy input is the clean tour corrupted to timestep ``t`` via the forward
    process; the target is the clean edge set the model must reconstruct (the
    ``x_0`` parameterization). The model is also conditioned on ``t``, which the
    caller already holds.

    Args:
        clean_edge_set: Clean binary edge set, shape ``(N, N)`` or ``(B, N, N)``.
        t: Timestep in ``[0, T]``; scalar, or shape ``(B,)`` for a batch.
        schedule: The noise schedule.
        generator: Optional ``torch.Generator`` for reproducible corruption.

    Returns:
        Tuple ``(noisy_input, target)``, both float tensors shaped like
        ``clean_edge_set``.
    """
    noisy_input = corrupt(clean_edge_set, t, schedule, generator=generator)
    target = clean_edge_set.to(torch.float32)
    return noisy_input, target


def edge_prediction_loss(
    pred_logits: torch.Tensor,
    target: torch.Tensor,
    reduction: str = "mean",
    pos_weight: Optional[Union[float, torch.Tensor]] = None,
) -> torch.Tensor:
    """Binary cross-entropy between predicted edge logits and the clean target.

    The loss is evaluated only on the strict upper triangle so each undirected
    edge contributes once and the diagonal is ignored. ``pred_logits`` are raw
    logits (pre-sigmoid) for ``P(edge in clean tour)``.

    Args:
        pred_logits: Per-edge logits, shape ``(N, N)`` or ``(B, N, N)``.
        target: Clean binary edge set, same shape as ``pred_logits``.
        reduction: ``"mean"`` (default), ``"sum"``, or ``"none"``. With
            ``"none"`` the per-edge losses are returned, shape ``(..., E)`` where
            ``E = N (N - 1) / 2``.
        pos_weight: Optional weight on the positive (present-edge) class to offset
            the sparsity of true tour edges.

    Returns:
        Scalar loss (``"mean"``/``"sum"``) or per-edge losses (``"none"``).
    """
    if pred_logits.shape != target.shape:
        raise ValueError(
            f"shape mismatch: pred_logits {tuple(pred_logits.shape)} vs target {tuple(target.shape)}"
        )
    if pred_logits.dim() not in (2, 3):
        raise ValueError(
            f"inputs must be (N, N) or (B, N, N), got {tuple(pred_logits.shape)}"
        )

    n = pred_logits.shape[-1]
    mask = torch.triu(torch.ones(n, n, device=pred_logits.device), diagonal=1).bool()
    pred_e = pred_logits[..., mask]
    target_e = target.to(torch.float32)[..., mask]

    weight = None
    if pos_weight is not None:
        weight = torch.as_tensor(pos_weight, dtype=torch.float32, device=pred_logits.device)

    return F.binary_cross_entropy_with_logits(
        pred_e, target_e, pos_weight=weight, reduction=reduction
    )
