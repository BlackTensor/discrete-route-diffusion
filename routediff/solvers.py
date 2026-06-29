"""Baseline TSP solvers.

Nearest-neighbor heuristic and OR-Tools (or 2-opt fallback) used both as
baselines and to produce clean target tours for training. See Task 1.3.

A *tour* here follows the convention in :mod:`routediff.graph`: a 1-D ``long``
tensor of shape ``(N,)`` listing the city visit order, with the closing edge
back to the start implied. Every solver returns a valid permutation of
``0..N-1`` (each city visited exactly once, cycle closed).
"""

from __future__ import annotations

import importlib.util

import torch

from routediff.graph import distance_matrix, tour_length

# Detect OR-Tools without importing it at module load (keeps import cheap and
# lets the package work when OR-Tools is not installed).
ORTOOLS_AVAILABLE = importlib.util.find_spec("ortools") is not None


def _as_single_instance(coords: torch.Tensor) -> torch.Tensor:
    if coords.dim() != 2 or coords.shape[-1] != 2:
        raise ValueError(f"coords must be (N, 2), got {tuple(coords.shape)}")
    return coords


def nearest_neighbor(coords: torch.Tensor, start: int = 0) -> torch.Tensor:
    """Greedy nearest-neighbor tour construction.

    Starting at ``start``, repeatedly move to the closest unvisited city.

    Args:
        coords: Tensor of shape ``(N, 2)``.
        start: Index of the starting city.

    Returns:
        1-D ``long`` tensor of shape ``(N,)``.
    """
    coords = _as_single_instance(coords)
    n = coords.shape[0]
    dist = distance_matrix(coords)

    visited = torch.zeros(n, dtype=torch.bool)
    tour = torch.empty(n, dtype=torch.long)
    current = int(start)
    tour[0] = current
    visited[current] = True

    for step in range(1, n):
        d = dist[current].clone()
        d[visited] = float("inf")
        current = int(torch.argmin(d).item())
        tour[step] = current
        visited[current] = True
    return tour


def two_opt(
    coords: torch.Tensor,
    tour: torch.Tensor,
    max_passes: int = 100,
) -> torch.Tensor:
    """Improve a tour with 2-opt local search until no swap helps.

    Repeatedly reverses tour segments whose reversal shortens the total length.

    Args:
        coords: Tensor of shape ``(N, 2)``.
        tour: Initial tour, 1-D ``long`` tensor of shape ``(N,)``.
        max_passes: Safety cap on full improvement sweeps.

    Returns:
        Improved tour, 1-D ``long`` tensor of shape ``(N,)``.
    """
    coords = _as_single_instance(coords)
    n = coords.shape[0]
    dist = distance_matrix(coords)
    best = tour.to(torch.long).clone()

    if n < 4:
        return best

    for _ in range(max_passes):
        improved = False
        for i in range(n - 1):
            a, b = int(best[i]), int(best[i + 1])
            for j in range(i + 2, n):
                # Edge (c, d) closes back to start when j == n - 1.
                c = int(best[j])
                d = int(best[(j + 1) % n])
                if d == a:
                    continue
                delta = (dist[a, c] + dist[b, d]) - (dist[a, b] + dist[c, d])
                if delta < -1e-9:
                    best[i + 1 : j + 1] = torch.flip(best[i + 1 : j + 1], dims=[0])
                    improved = True
                    b = int(best[i + 1])
        if not improved:
            break
    return best


def solve_or_tools(coords: torch.Tensor, time_limit_s: float = 2.0) -> torch.Tensor:
    """Solve a TSP instance with OR-Tools routing.

    Args:
        coords: Tensor of shape ``(N, 2)``.
        time_limit_s: Wall-clock budget for the local-search phase.

    Returns:
        1-D ``long`` tensor of shape ``(N,)``.

    Raises:
        RuntimeError: If OR-Tools is not installed or fails to find a solution.
    """
    if not ORTOOLS_AVAILABLE:
        raise RuntimeError("OR-Tools is not installed; use nearest_neighbor + two_opt")

    from ortools.constraint_solver import pywrapcp, routing_enums_pb2

    coords = _as_single_instance(coords)
    n = coords.shape[0]

    # OR-Tools needs integer costs; scale unit-square distances and round.
    scale = 1_000_000
    dist_int = (distance_matrix(coords) * scale).round().to(torch.long)

    manager = pywrapcp.RoutingIndexManager(n, 1, 0)
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index: int, to_index: int) -> int:
        i = manager.IndexToNode(from_index)
        j = manager.IndexToNode(to_index)
        return int(dist_int[i, j].item())

    transit_idx = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_idx)

    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )
    params.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    )
    params.time_limit.FromMilliseconds(int(time_limit_s * 1000))

    solution = routing.SolveWithParameters(params)
    if solution is None:
        raise RuntimeError("OR-Tools failed to find a solution")

    order: list[int] = []
    index = routing.Start(0)
    while not routing.IsEnd(index):
        order.append(manager.IndexToNode(index))
        index = solution.Value(routing.NextVar(index))
    return torch.tensor(order, dtype=torch.long)


def solve(coords: torch.Tensor, method: str = "auto", **kwargs) -> torch.Tensor:
    """Solve a TSP instance, producing a clean target tour.

    Args:
        coords: Tensor of shape ``(N, 2)``.
        method: One of ``"auto"``, ``"or_tools"``, ``"nn"``, ``"nn2opt"``.
            ``"auto"`` uses OR-Tools if available, otherwise nearest-neighbor
            refined by 2-opt.
        **kwargs: Forwarded to the chosen solver (e.g. ``time_limit_s``).

    Returns:
        1-D ``long`` tensor of shape ``(N,)``.
    """
    coords = _as_single_instance(coords)

    if method == "auto":
        method = "or_tools" if ORTOOLS_AVAILABLE else "nn2opt"

    if method == "or_tools":
        return solve_or_tools(coords, **kwargs)
    if method == "nn":
        return nearest_neighbor(coords, **kwargs)
    if method == "nn2opt":
        return two_opt(coords, nearest_neighbor(coords))
    raise ValueError(f"unknown method {method!r}")


def is_valid_tour(tour: torch.Tensor, num_cities: int) -> bool:
    """Return True iff ``tour`` is a permutation of ``0..num_cities - 1``."""
    if tour.dim() != 1 or tour.numel() != num_cities:
        return False
    return bool(torch.equal(torch.sort(tour).values, torch.arange(num_cities)))


if __name__ == "__main__":
    from data.generator import generate_instances

    pts = generate_instances(num_cities=15, num_instances=1, seed=0)[0]
    nn = nearest_neighbor(pts)
    opt = two_opt(pts, nn)
    print("nearest-neighbor length:", float(tour_length(pts, nn)))
    print("2-opt length:          ", float(tour_length(pts, opt)))
    if ORTOOLS_AVAILABLE:
        ort = solve_or_tools(pts)
        print("or-tools length:       ", float(tour_length(pts, ort)))
