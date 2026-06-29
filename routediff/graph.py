"""Graph and tour representation.

Distance matrices, adjacency for the GNN, and conversions between a tour as a
permutation and as an edge set.

Conventions
-----------
- ``coords``: float tensor of shape ``(N, 2)`` for a single instance, or
  ``(B, N, 2)`` for a batch, with coordinates in the unit square.
- A *tour* (permutation) is a 1-D ``long`` tensor of shape ``(N,)`` listing the
  city visit order. It implies a closed cycle: the last city connects back to
  the first.
- An *edge set* is a symmetric binary adjacency matrix of shape ``(N, N)`` where
  ``A[i, j] == 1`` iff cities ``i`` and ``j`` are adjacent on the tour. Each row
  has exactly two non-zero entries for a valid Hamiltonian cycle. The diagonal
  is always zero.
"""

from __future__ import annotations

import torch


def distance_matrix(coords: torch.Tensor) -> torch.Tensor:
    """Euclidean distance matrix from coordinates.

    Args:
        coords: Tensor of shape ``(N, 2)`` or ``(B, N, 2)``.

    Returns:
        Tensor of shape ``(N, N)`` or ``(B, N, N)`` of pairwise distances.
    """
    if coords.dim() not in (2, 3):
        raise ValueError(f"coords must be (N, 2) or (B, N, 2), got {tuple(coords.shape)}")
    if coords.shape[-1] != 2:
        raise ValueError(f"last dim of coords must be 2, got {coords.shape[-1]}")
    # torch.cdist handles both batched and unbatched inputs.
    return torch.cdist(coords, coords, p=2.0)


def adjacency_from_edge_set(edge_set: torch.Tensor) -> torch.Tensor:
    """Return adjacency suitable for the GNN.

    The edge set is already a symmetric binary adjacency matrix; this is provided
    as a named entry point so callers do not depend on that being an identity.

    Args:
        edge_set: Binary tensor of shape ``(N, N)`` or ``(B, N, N)``.

    Returns:
        A float adjacency tensor of the same shape.
    """
    return edge_set.to(torch.float32)


def tour_to_edge_set(tour: torch.Tensor, num_cities: int | None = None) -> torch.Tensor:
    """Convert a permutation tour to a symmetric binary edge-set matrix.

    Args:
        tour: 1-D ``long`` tensor of shape ``(N,)`` giving the visit order.
        num_cities: Optional explicit ``N``. Defaults to ``tour.numel()``.

    Returns:
        Symmetric binary tensor of shape ``(N, N)`` with zero diagonal.
    """
    if tour.dim() != 1:
        raise ValueError(f"tour must be 1-D, got shape {tuple(tour.shape)}")
    n = num_cities if num_cities is not None else int(tour.numel())
    if tour.numel() != n:
        raise ValueError(f"tour length {tour.numel()} does not match num_cities {n}")

    tour = tour.to(torch.long)
    nxt = torch.roll(tour, shifts=-1)  # successor of each visited city
    edges = torch.zeros((n, n), dtype=torch.float32)
    edges[tour, nxt] = 1.0
    edges[nxt, tour] = 1.0
    edges.fill_diagonal_(0.0)
    return edges


def edge_set_to_tour(edge_set: torch.Tensor, start: int = 0) -> torch.Tensor:
    """Convert a symmetric edge-set matrix back to a permutation tour.

    Walks the cycle starting at ``start``. Assumes ``edge_set`` encodes a valid
    Hamiltonian cycle (every node has degree exactly two).

    Args:
        edge_set: Symmetric binary tensor of shape ``(N, N)``.
        start: City to begin the walk from.

    Returns:
        1-D ``long`` tensor of shape ``(N,)`` giving the visit order.
    """
    if edge_set.dim() != 2 or edge_set.shape[0] != edge_set.shape[1]:
        raise ValueError(f"edge_set must be square (N, N), got {tuple(edge_set.shape)}")
    n = edge_set.shape[0]
    adj = edge_set > 0.5

    tour = torch.empty(n, dtype=torch.long)
    tour[0] = start
    prev = -1
    current = start
    for step in range(1, n):
        neighbors = torch.nonzero(adj[current], as_tuple=False).flatten().tolist()
        nxt = None
        for cand in neighbors:
            if cand != prev and cand != current:
                nxt = cand
                break
        if nxt is None:
            raise ValueError(
                f"edge_set is not a valid Hamiltonian cycle: dead end at city {current}"
            )
        tour[step] = nxt
        prev, current = current, nxt
    return tour


def tour_length(coords: torch.Tensor, tour: torch.Tensor) -> torch.Tensor:
    """Total Euclidean length of a closed tour.

    Args:
        coords: Tensor of shape ``(N, 2)``.
        tour: 1-D ``long`` tensor of shape ``(N,)`` giving the visit order.

    Returns:
        Scalar tensor: the summed length including the closing edge back to the
        start.
    """
    if coords.dim() != 2 or coords.shape[-1] != 2:
        raise ValueError(f"coords must be (N, 2), got {tuple(coords.shape)}")
    if tour.dim() != 1:
        raise ValueError(f"tour must be 1-D, got shape {tuple(tour.shape)}")

    tour = tour.to(torch.long)
    ordered = coords[tour]
    rolled = torch.roll(ordered, shifts=-1, dims=0)
    seg = torch.linalg.vector_norm(ordered - rolled, dim=1)
    return seg.sum()
