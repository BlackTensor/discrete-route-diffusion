"""GNN denoiser model.

A graph neural network that consumes the noisy graph, node coordinates, and a
timestep embedding and outputs per-edge logits.

Architecture
------------
The denoiser is an anisotropic message-passing network (a dense GatedGCN, in the
spirit of the graph-based TSP diffusion models such as DIFUSCO) that maintains
*both* node and edge embeddings throughout. Because the TSP graph is complete and
``N`` is small, every tensor is kept dense:

- node features  ``h``  : shape ``(B, N, H)``
- edge features  ``e``  : shape ``(B, N, N, H)``

Inputs are embedded as follows:

- nodes : the 2D coordinates are projected to ``H`` dimensions.
- edges : two scalars per edge -- the noisy edge state (present / absent at the
  current timestep) and the Euclidean distance between the two cities -- are
  concatenated and projected to ``H`` dimensions. The distance gives the network
  the geometry it needs to prefer short edges.
- time  : a sinusoidal embedding of the timestep ``t`` is passed through a small
  MLP and injected as an additive bias into every layer (Task 3.2 refines this).

Each :class:`GNNLayer` performs a GatedGCN update::

    e_ij <- e_ij + act( LN( A h_i + B h_j + C e_ij ) + time_e )
    eta_ij = sigmoid(e_ij)                       # soft edge gates
    h_i  <- h_i  + act( LN( U h_i + sum_j eta_ij * V h_j / sum_j eta_ij ) + time_h )

After the final layer an edge MLP maps each edge embedding to a single logit; the
output is symmetrized so ``logit[i, j] == logit[j, i]`` (the tour is undirected).
The logit is interpreted as ``P(edge in clean tour)`` -- exactly the target the
diffusion loss in :mod:`routediff.diffusion` expects.
"""

from __future__ import annotations

import math
from typing import Union

import torch
import torch.nn as nn

from routediff.graph import distance_matrix

IntOrTensor = Union[int, torch.Tensor]


def sinusoidal_timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Standard sinusoidal embedding of integer timesteps.

    Args:
        t: ``long`` or float tensor of shape ``(B,)`` with timestep values.
        dim: Embedding dimension. Should be even; an odd ``dim`` is zero-padded.

    Returns:
        Float tensor of shape ``(B, dim)``.
    """
    if t.dim() != 1:
        raise ValueError(f"t must be 1-D of shape (B,), got {tuple(t.shape)}")
    half = dim // 2
    device = t.device
    t = t.to(torch.float32)
    if half > 0:
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=device, dtype=torch.float32) / half
        )
        args = t[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    else:
        emb = torch.zeros((t.shape[0], 0), device=device)
    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros((t.shape[0], 1), device=device)], dim=-1)
    return emb


class TimestepEmbedding(nn.Module):
    """Sinusoidal timestep embedding followed by a small MLP."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Map timesteps ``(B,)`` to embeddings ``(B, hidden_dim)``."""
        return self.mlp(sinusoidal_timestep_embedding(t, self.hidden_dim))


class GNNLayer(nn.Module):
    """A single dense anisotropic (GatedGCN) message-passing layer."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.lin_a = nn.Linear(hidden_dim, hidden_dim)  # source node -> edge
        self.lin_b = nn.Linear(hidden_dim, hidden_dim)  # target node -> edge
        self.lin_c = nn.Linear(hidden_dim, hidden_dim)  # edge -> edge
        self.lin_u = nn.Linear(hidden_dim, hidden_dim)  # self node -> node
        self.lin_v = nn.Linear(hidden_dim, hidden_dim)  # neighbor node -> node

        self.time_node = nn.Linear(hidden_dim, hidden_dim)
        self.time_edge = nn.Linear(hidden_dim, hidden_dim)

        self.norm_node = nn.LayerNorm(hidden_dim)
        self.norm_edge = nn.LayerNorm(hidden_dim)
        self.act = nn.SiLU()

    def forward(
        self, h: torch.Tensor, e: torch.Tensor, t_emb: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Update node features ``h`` and edge features ``e``.

        Args:
            h: Node features, shape ``(B, N, H)``.
            e: Edge features, shape ``(B, N, N, H)``.
            t_emb: Timestep embedding, shape ``(B, H)``.

        Returns:
            Updated ``(h, e)`` of the same shapes.
        """
        t_node = self.time_node(t_emb)[:, None, :]  # (B, 1, H)
        t_edge = self.time_edge(t_emb)[:, None, None, :]  # (B, 1, 1, H)

        # Edge update: combine source node i, target node j, and current edge.
        ah = self.lin_a(h)[:, :, None, :]  # (B, N, 1, H) broadcast over j
        bh = self.lin_b(h)[:, None, :, :]  # (B, 1, N, H) broadcast over i
        ce = self.lin_c(e)  # (B, N, N, H)
        e_new = e + self.act(self.norm_edge(ah + bh + ce) + t_edge)

        # Node update: gated aggregation of neighbour messages.
        gates = torch.sigmoid(e_new)  # (B, N, N, H)
        vh = self.lin_v(h)[:, None, :, :]  # (B, 1, N, H) message from each j
        agg = (gates * vh).sum(dim=2) / (gates.sum(dim=2) + 1e-6)  # (B, N, H)
        h_new = h + self.act(self.norm_node(self.lin_u(h) + agg) + t_node)

        return h_new, e_new


class RouteDiffusionDenoiser(nn.Module):
    """GNN denoiser predicting per-edge clean-tour logits.

    Args:
        hidden_dim: Width ``H`` of node/edge embeddings.
        num_layers: Number of message-passing layers.
    """

    def __init__(self, hidden_dim: int = 64, num_layers: int = 4):
        super().__init__()
        if hidden_dim < 1:
            raise ValueError(f"hidden_dim must be >= 1, got {hidden_dim}")
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.node_in = nn.Linear(2, hidden_dim)  # from coordinates
        self.edge_in = nn.Linear(2, hidden_dim)  # from [noisy edge state, distance]
        self.time_embed = TimestepEmbedding(hidden_dim)

        self.layers = nn.ModuleList(GNNLayer(hidden_dim) for _ in range(num_layers))

        self.edge_out = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        coords: torch.Tensor,
        noisy_edge_set: torch.Tensor,
        t: IntOrTensor,
    ) -> torch.Tensor:
        """Predict per-edge clean-tour logits.

        Args:
            coords: City coordinates, shape ``(N, 2)`` or ``(B, N, 2)``.
            noisy_edge_set: Noised binary edge set, shape ``(N, N)`` or
                ``(B, N, N)``, as produced by :func:`routediff.diffusion.corrupt`.
            t: Timestep; scalar, or shape ``(B,)`` for a batch.

        Returns:
            Symmetric per-edge logits of shape ``(N, N)`` (unbatched input) or
            ``(B, N, N)`` (batched input), with a zero diagonal.
        """
        batched = coords.dim() == 3
        if coords.dim() not in (2, 3):
            raise ValueError(f"coords must be (N, 2) or (B, N, 2), got {tuple(coords.shape)}")
        if noisy_edge_set.dim() != coords.dim():
            raise ValueError(
                "noisy_edge_set rank must match coords: "
                f"got {tuple(noisy_edge_set.shape)} for coords {tuple(coords.shape)}"
            )

        c = coords if batched else coords.unsqueeze(0)  # (B, N, 2)
        adj = noisy_edge_set if batched else noisy_edge_set.unsqueeze(0)  # (B, N, N)
        b, n = c.shape[0], c.shape[1]
        device = c.device

        t_tensor = torch.as_tensor(t, device=device)
        if t_tensor.dim() == 0:
            t_tensor = t_tensor.expand(b)
        if t_tensor.shape != (b,):
            raise ValueError(f"t must be scalar or shape ({b},), got {tuple(t_tensor.shape)}")

        # Embed inputs.
        h = self.node_in(c.to(torch.float32))  # (B, N, H)
        dist = distance_matrix(c)  # (B, N, N)
        edge_feats = torch.stack([adj.to(torch.float32), dist], dim=-1)  # (B, N, N, 2)
        e = self.edge_in(edge_feats)  # (B, N, N, H)
        t_emb = self.time_embed(t_tensor)  # (B, H)

        for layer in self.layers:
            h, e = layer(h, e, t_emb)

        logits = self.edge_out(e).squeeze(-1)  # (B, N, N)
        logits = 0.5 * (logits + logits.transpose(-1, -2))  # undirected -> symmetric
        diag = torch.eye(n, device=device, dtype=torch.bool)
        logits = logits.masked_fill(diag, 0.0)

        return logits if batched else logits.squeeze(0)
