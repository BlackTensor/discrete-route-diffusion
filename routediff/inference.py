"""Reverse sampling and tour decoding.

Iteratively denoises from ``t = T`` to ``t = 0`` with the trained model and
decodes the predicted edge probabilities into a valid Hamiltonian tour.

Reverse (ancestral) sampling
----------------------------
The denoiser uses the ``x_0`` parameterization (see :mod:`routediff.diffusion`):
given a noised edge set ``e_t`` and timestep ``t`` it predicts, per undirected
edge, ``p_theta(e_0 = present | e_t, t) = sigmoid(logit)``. To sample we do
standard D3PM ancestral sampling: start from the stationary prior ``e_T`` (each
edge a fair coin flip) and, for ``t = T, ..., 1``, draw ``e_{t-1}`` from the
posterior

    p(e_{t-1} | e_t) = sum_{e_0} q(e_{t-1} | e_t, e_0) p_theta(e_0 | e_t) .

For the binary uniform kernel each edge is independent and the posterior has a
closed form. Writing ``a_t = alpha_bar_t`` and ``beta_t`` for the single-step
resample probability, the relevant factors over a state ``k in {0, 1}`` are

    Q_t[k, e_t]            = (1 - beta_t) [k == e_t] + 0.5 beta_t      (one step)
    Qbar_{t-1}[e_0, k]     = a_{t-1} [k == e_0] + 0.5 (1 - a_{t-1})    (cumulative)
    Z(e_0) = Qbar_t[e_0, e_t] = a_t [e_t == e_0] + 0.5 (1 - a_t)      (normalizer)

and ``q(e_{t-1} = k | e_t, e_0) = Q_t[k, e_t] Qbar_{t-1}[e_0, k] / Z(e_0)``.
Marginalizing over the model's distribution on ``e_0`` gives the present-edge
probability used to sample ``e_{t-1}``.

Decoding
--------
The final clean-edge probabilities are turned into a valid tour by a greedy
edge-matching decoder: sort all undirected edges by predicted probability and
add them one at a time, skipping any edge that would push a city past degree two
or close a subtour before every city is included. Because the candidate graph is
complete this always yields a single Hamiltonian cycle.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch

from routediff.diffusion import DiffusionSchedule, sample_prior
from routediff.graph import tour_length, tour_to_edge_set
from routediff.model import RouteDiffusionDenoiser


@dataclass
class SampleResult:
    """Outcome of a reverse-sampling run on one instance.

    Attributes:
        tour: Decoded permutation tour, 1-D ``long`` tensor of shape ``(N,)``.
        edge_probs: Final per-edge clean probabilities, shape ``(N, N)``.
        length: Total Euclidean length of ``tour``.
        history: Optional list of per-step snapshots (see ``reverse_sample``);
            empty unless ``record_history=True``.
    """

    tour: torch.Tensor
    edge_probs: torch.Tensor
    length: float
    history: list = field(default_factory=list)


def load_checkpoint(
    path: str, device: str = "cpu"
) -> tuple[RouteDiffusionDenoiser, DiffusionSchedule, dict]:
    """Load a trained denoiser, its noise schedule, and config from a checkpoint.

    Args:
        path: Path to a checkpoint written by :func:`routediff.train.save_checkpoint`.
        device: Torch device string for the restored tensors.

    Returns:
        Tuple ``(model, schedule, config)`` with the model in eval mode.
    """
    device_t = torch.device(device)
    ckpt = torch.load(path, map_location=device_t, weights_only=False)
    config = ckpt.get("config", {})

    model = RouteDiffusionDenoiser(
        hidden_dim=int(config.get("hidden_dim", 64)),
        num_layers=int(config.get("num_layers", 4)),
    ).to(device_t)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    sch = ckpt["schedule"]
    schedule = DiffusionSchedule(
        num_timesteps=int(sch["num_timesteps"]),
        betas=sch["betas"].to(device_t),
        alphas=sch["alphas"].to(device_t),
        alpha_bars=sch["alpha_bars"].to(device_t),
    )
    return model, schedule, config


def reverse_posterior_present_prob(
    e_t: torch.Tensor,
    p_e0_present: torch.Tensor,
    t: int,
    schedule: DiffusionSchedule,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Posterior probability each edge is *present* at timestep ``t - 1``.

    Implements the closed-form binary D3PM posterior marginalized over the
    model's ``e_0`` distribution (see the module docstring).

    Args:
        e_t: Current binary edge set at timestep ``t``, shape ``(N, N)``.
        p_e0_present: Model probability each edge is present in the clean tour,
            shape ``(N, N)``, values in ``[0, 1]``.
        t: Current timestep in ``[1, T]``.
        schedule: The noise schedule.
        eps: Small constant guarding the normalizer division.

    Returns:
        Float tensor of present-probabilities for ``e_{t-1}``, shape ``(N, N)``.
    """
    if t < 1 or t > schedule.num_timesteps:
        raise ValueError(f"t must be in [1, {schedule.num_timesteps}], got {t}")

    a_t = schedule.alpha_bars[t]
    a_prev = schedule.alpha_bars[t - 1]
    beta_t = schedule.betas[t]

    is_one = e_t > 0.5  # whether the observed edge state is "present"

    # One-step factor Q_t[k, e_t] for k = 1 (present).
    qt_same = 1.0 - 0.5 * beta_t
    qt_diff = 0.5 * beta_t
    qt1 = torch.where(is_one, qt_same.expand_as(e_t), qt_diff.expand_as(e_t))

    # Cumulative factor Qbar_{t-1}[e_0, k] for k = 1.
    qbar_same = 0.5 + 0.5 * a_prev  # k == e_0
    qbar_diff = 0.5 * (1.0 - a_prev)  # k != e_0

    # Normalizer Z(e_0) = Qbar_t[e_0, e_t].
    zt_same = 0.5 + 0.5 * a_t
    zt_diff = 0.5 * (1.0 - a_t)
    z_for_x0_1 = torch.where(is_one, zt_same.expand_as(e_t), zt_diff.expand_as(e_t))
    z_for_x0_0 = torch.where(~is_one, zt_same.expand_as(e_t), zt_diff.expand_as(e_t))

    # Posterior P(e_{t-1} = 1 | e_t, e_0) for each hypothesized clean state e_0.
    post1_given_x0_1 = qt1 * qbar_same / (z_for_x0_1 + eps)
    post1_given_x0_0 = qt1 * qbar_diff / (z_for_x0_0 + eps)

    p1 = p_e0_present
    p0 = 1.0 - p1
    return post1_given_x0_1 * p1 + post1_given_x0_0 * p0


def _sample_symmetric(p_present: torch.Tensor, generator: Optional[torch.Generator]) -> torch.Tensor:
    """Sample a symmetric binary edge set from per-edge present-probabilities."""
    n = p_present.shape[-1]
    device = p_present.device
    rand = torch.rand(p_present.shape, generator=generator, device=device)
    upper = torch.triu(torch.ones(n, n, device=device), diagonal=1)
    decision = (rand < p_present).to(torch.float32) * upper
    return decision + decision.transpose(-1, -2)


def greedy_decode(edge_scores: torch.Tensor) -> torch.Tensor:
    """Decode per-edge scores into a valid Hamiltonian tour (greedy matching).

    Edges are considered in descending score order and accepted unless they would
    raise a city above degree two or close a cycle before all cities are linked.
    The candidate graph is complete, so a single Hamiltonian cycle always results.

    Args:
        edge_scores: Symmetric per-edge scores, shape ``(N, N)`` (higher = more
            likely to be a tour edge). The diagonal is ignored.

    Returns:
        1-D ``long`` permutation tour of shape ``(N,)``.
    """
    if edge_scores.dim() != 2 or edge_scores.shape[0] != edge_scores.shape[1]:
        raise ValueError(f"edge_scores must be square (N, N), got {tuple(edge_scores.shape)}")
    n = edge_scores.shape[0]
    if n < 3:
        raise ValueError(f"need at least 3 cities to form a tour, got {n}")

    iu = torch.triu_indices(n, n, offset=1)
    scores = edge_scores[iu[0], iu[1]]
    order = torch.argsort(scores, descending=True).tolist()
    rows = iu[0].tolist()
    cols = iu[1].tolist()

    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    degree = [0] * n
    adj: list[list[int]] = [[] for _ in range(n)]
    num_edges = 0

    for k in order:
        i, j = rows[k], cols[k]
        if degree[i] >= 2 or degree[j] >= 2:
            continue
        ri, rj = find(i), find(j)
        if ri == rj and num_edges < n - 1:
            continue  # closing a subtour before every city is in
        parent[ri] = rj
        degree[i] += 1
        degree[j] += 1
        adj[i].append(j)
        adj[j].append(i)
        num_edges += 1
        if num_edges == n:
            break

    # Walk the resulting cycle starting from city 0.
    tour = [0]
    prev, cur = -1, 0
    for _ in range(n - 1):
        nbrs = adj[cur]
        nxt = nbrs[0] if nbrs[0] != prev else nbrs[1]
        tour.append(nxt)
        prev, cur = cur, nxt
    return torch.tensor(tour, dtype=torch.long)


def reverse_sample(
    model: RouteDiffusionDenoiser,
    coords: torch.Tensor,
    schedule: DiffusionSchedule,
    generator: Optional[torch.Generator] = None,
    record_history: bool = False,
) -> SampleResult:
    """Run reverse diffusion on one instance and decode a tour.

    Starts from the stationary prior ``e_T`` and denoises down to ``e_0`` using
    the model's ``x_0`` predictions, then greedily decodes the final clean-edge
    probabilities into a valid Hamiltonian tour.

    Args:
        model: Trained denoiser (set to eval mode by the caller; this also runs
            under ``torch.no_grad``).
        coords: City coordinates for one instance, shape ``(N, 2)``.
        schedule: The noise schedule used during training.
        generator: Optional ``torch.Generator`` for reproducible sampling.
        record_history: If ``True``, capture a snapshot per step into
            ``SampleResult.history`` (used by Task 5.2 for animation).

    Returns:
        A :class:`SampleResult` with the decoded tour, final edge probabilities,
        and tour length.
    """
    if coords.dim() != 2 or coords.shape[-1] != 2:
        raise ValueError(f"coords must be (N, 2), got {tuple(coords.shape)}")
    device = coords.device
    n = coords.shape[0]

    e_t = sample_prior(n, generator=generator, device=device)
    p_e0 = torch.full((n, n), 0.5, device=device)
    history: list = []

    was_training = model.training
    model.eval()
    with torch.no_grad():
        for t in range(schedule.num_timesteps, 0, -1):
            logits = model(coords, e_t, t)
            p_e0 = torch.sigmoid(logits)

            if record_history:
                history.append(
                    {"t": t, "edge_set": e_t.detach().cpu(), "edge_probs": p_e0.detach().cpu()}
                )

            p_prev = reverse_posterior_present_prob(e_t, p_e0, t, schedule)
            e_t = _sample_symmetric(p_prev, generator)

    if was_training:
        model.train()

    if record_history:
        history.append({"t": 0, "edge_set": e_t.detach().cpu(), "edge_probs": p_e0.detach().cpu()})

    tour = greedy_decode(p_e0)
    length = float(tour_length(coords, tour))
    return SampleResult(tour=tour, edge_probs=p_e0.detach(), length=length, history=history)


def sample_from_checkpoint(
    checkpoint_path: str,
    coords: torch.Tensor,
    device: str = "cpu",
    seed: Optional[int] = None,
    record_history: bool = False,
) -> SampleResult:
    """Convenience wrapper: load a checkpoint and reverse-sample one instance.

    Args:
        checkpoint_path: Path to a trained checkpoint.
        coords: City coordinates, shape ``(N, 2)``.
        device: Torch device string.
        seed: Optional seed for reproducible sampling.
        record_history: Forwarded to :func:`reverse_sample`.

    Returns:
        A :class:`SampleResult`.
    """
    model, schedule, _ = load_checkpoint(checkpoint_path, device=device)
    coords = coords.to(torch.device(device))
    generator = None
    if seed is not None:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
    return reverse_sample(model, coords, schedule, generator=generator, record_history=record_history)


def _present_edges(edge_set: torch.Tensor) -> list[list[int]]:
    """Return the present undirected edges of a symmetric edge set as ``[i, j]`` pairs."""
    n = edge_set.shape[0]
    idx = torch.triu_indices(n, n, offset=1)
    present = edge_set[idx[0], idx[1]] > 0.5
    ij = torch.stack([idx[0][present], idx[1][present]], dim=1)
    return ij.to(torch.long).tolist()


def _probs_grid(edge_probs: torch.Tensor, ndigits: int = 3) -> list[list[float]]:
    """Return the symmetric per-edge confidence as a rounded ``N x N`` grid.

    Powers the UI edge-confidence heatmap: each cell is the model's probability
    that the corresponding edge belongs to the clean tour at that step. The
    diagonal is zeroed since self-edges are meaningless.
    """
    n = edge_probs.shape[0]
    grid = edge_probs.detach().cpu().to(torch.float32).clone()
    grid.fill_diagonal_(0.0)
    return [[round(float(v), ndigits) for v in row] for row in grid.tolist()]


def build_timeline(coords: torch.Tensor, result: SampleResult) -> dict:
    """Build a JSON-serializable noise-to-clean timeline for one instance.

    Captures the graph state at every denoising step so the UI can animate the
    evolution from a chaotic random graph to the clean tour. Requires the result
    to carry per-step history (run ``reverse_sample(..., record_history=True)``).

    The timeline lists the city coordinates once and, for each step, the set of
    edges present at that step (ordered from ``t = T`` down to ``t = 0``). A final
    frame holds the decoded Hamiltonian tour so the animation ends on a clean
    cycle even if the last sampled state is slightly noisy.

    Args:
        coords: City coordinates for the instance, shape ``(N, 2)``.
        result: A :class:`SampleResult` produced with ``record_history=True``.

    Returns:
        A nested ``dict`` of plain Python types ready for :func:`json.dump`.
    """
    if coords.dim() != 2 or coords.shape[-1] != 2:
        raise ValueError(f"coords must be (N, 2), got {tuple(coords.shape)}")
    if not result.history:
        raise ValueError(
            "result has no history; call reverse_sample(..., record_history=True)"
        )

    n = coords.shape[0]
    tour_edges = _present_edges(tour_to_edge_set(result.tour, n))

    steps: list[dict] = []
    last_probs: list[list[float]] = [[0.0] * n for _ in range(n)]
    for snap in result.history:
        last_probs = _probs_grid(snap["edge_probs"])
        steps.append(
            {
                "t": int(snap["t"]),
                "edges": _present_edges(snap["edge_set"]),
                "probs": last_probs,
            }
        )
    # Terminal frame: the decoded, guaranteed-valid tour. Reuse the final clean
    # prediction as its confidence grid so the heatmap settles rather than blanks.
    steps.append({"t": 0, "edges": tour_edges, "probs": last_probs, "final": True})

    return {
        "num_cities": int(n),
        "num_steps": len(steps),
        "coords": coords.detach().cpu().to(torch.float32).tolist(),
        "steps": steps,
        "tour": result.tour.detach().cpu().to(torch.long).tolist(),
        "tour_edges": tour_edges,
        "length": float(result.length),
    }


def export_timeline(coords: torch.Tensor, result: SampleResult, path: str) -> dict:
    """Write a :func:`build_timeline` JSON file and return the timeline dict.

    Args:
        coords: City coordinates for the instance, shape ``(N, 2)``.
        result: A :class:`SampleResult` produced with ``record_history=True``.
        path: Destination JSON file; parent directories are created.

    Returns:
        The timeline ``dict`` that was written.
    """
    timeline = build_timeline(coords, result)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(timeline, f)
    return timeline


__all__ = [
    "SampleResult",
    "load_checkpoint",
    "reverse_posterior_present_prob",
    "greedy_decode",
    "reverse_sample",
    "sample_from_checkpoint",
    "build_timeline",
    "export_timeline",
]
