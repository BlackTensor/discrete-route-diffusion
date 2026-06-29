"""Benchmark harness.

Compares the discrete diffusion model against classical baselines
(nearest-neighbor, 2-opt, and OR-Tools when installed) on tour quality and
runtime across many instances, then renders parchment-themed Matplotlib charts.

What is measured
----------------
For each method and each instance we record the decoded tour length and the
wall-clock time to produce it. Aggregated across the instance set we report:

- mean tour length (lower is better) with standard deviation,
- mean optimality gap, the per-instance excess over the best length any method
  found on that instance, in percent (so the best method on average sits near
  zero),
- mean runtime per instance in milliseconds.

The diffusion sampler is seeded per instance for reproducibility. The strong
classical reference is OR-Tools if it is installed, otherwise 2-opt-refined
nearest-neighbor (matching the project's ``solve(method="auto")`` fallback).

Usage
-----
    python -m benchmarks.compare --num-instances 50 --num-cities 15

Outputs land in ``--out-dir`` (default ``benchmarks/results``): a ``results.csv``
of per-instance lengths and times, and ``benchmark.png`` with the charts.
"""

from __future__ import annotations

import argparse
import csv
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import torch

from data.generator import generate_instances
from routediff.graph import tour_length
from routediff.inference import load_checkpoint, reverse_sample
from routediff.solvers import (
    ORTOOLS_AVAILABLE,
    nearest_neighbor,
    solve_or_tools,
    two_opt,
)

# Parchment theme matching the cartographic web UI.
BACKGROUND = "#f0e6d2"
PANEL = "#f5ecd7"
FOREGROUND = "#2d2a22"
MUTED = "#6b6453"
GRID = "#d8cbb0"
ACCENT = "#ea580c"
# A warm amber/terracotta palette for the per-method bars (no cool colors).
METHOD_COLORS = {
    "nearest_neighbor": "#d6b98c",
    "two_opt": "#c2410c",
    "or_tools": "#9a3412",
    "diffusion": ACCENT,
}
METHOD_LABELS = {
    "nearest_neighbor": "Nearest-neighbor",
    "two_opt": "2-opt",
    "or_tools": "OR-Tools",
    "diffusion": "Diffusion (ours)",
}


@dataclass
class MethodResult:
    """Per-method aggregate over the benchmark instance set."""

    name: str
    lengths: list[float]
    times_ms: list[float]
    gaps_pct: list[float]

    @property
    def mean_length(self) -> float:
        return float(sum(self.lengths) / len(self.lengths))

    @property
    def std_length(self) -> float:
        mean = self.mean_length
        var = sum((x - mean) ** 2 for x in self.lengths) / len(self.lengths)
        return float(var ** 0.5)

    @property
    def mean_time_ms(self) -> float:
        return float(sum(self.times_ms) / len(self.times_ms))

    @property
    def mean_gap_pct(self) -> float:
        return float(sum(self.gaps_pct) / len(self.gaps_pct))


def _build_methods(
    checkpoint: str,
    device: str,
    time_limit_s: float,
) -> dict[str, Callable[[torch.Tensor, int], tuple[torch.Tensor, float]]]:
    """Return a name -> solver mapping.

    Each solver takes ``(coords, seed)`` and returns ``(tour, length)``. The seed
    is only consumed by the stochastic diffusion sampler; classical methods are
    deterministic and ignore it.
    """
    model, schedule, _ = load_checkpoint(checkpoint, device=device)

    def run_nn(coords: torch.Tensor, _seed: int) -> tuple[torch.Tensor, float]:
        tour = nearest_neighbor(coords)
        return tour, float(tour_length(coords, tour))

    def run_two_opt(coords: torch.Tensor, _seed: int) -> tuple[torch.Tensor, float]:
        tour = two_opt(coords, nearest_neighbor(coords))
        return tour, float(tour_length(coords, tour))

    def run_diffusion(coords: torch.Tensor, seed: int) -> tuple[torch.Tensor, float]:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        result = reverse_sample(model, coords, schedule, generator=generator)
        return result.tour, result.length

    methods: dict[str, Callable[[torch.Tensor, int], tuple[torch.Tensor, float]]] = {
        "nearest_neighbor": run_nn,
        "two_opt": run_two_opt,
    }

    if ORTOOLS_AVAILABLE:
        def run_or_tools(coords: torch.Tensor, _seed: int) -> tuple[torch.Tensor, float]:
            tour = solve_or_tools(coords, time_limit_s=time_limit_s)
            return tour, float(tour_length(coords, tour))

        methods["or_tools"] = run_or_tools

    methods["diffusion"] = run_diffusion
    return methods


def run_benchmark(
    checkpoint: str,
    num_instances: int = 50,
    num_cities: int = 15,
    seed: int = 0,
    device: str = "cpu",
    time_limit_s: float = 2.0,
) -> list[MethodResult]:
    """Run every method on a shared set of instances and aggregate results.

    Args:
        checkpoint: Path to a trained diffusion checkpoint.
        num_instances: Number of TSP instances to evaluate on.
        num_cities: Cities per instance (should match the checkpoint's training).
        seed: Base seed; instances and per-instance sampling derive from it.
        device: Torch device string.
        time_limit_s: OR-Tools local-search budget per instance (if installed).

    Returns:
        One :class:`MethodResult` per method, in display order.
    """
    coords_batch = generate_instances(
        num_cities=num_cities, num_instances=num_instances, seed=seed, device=torch.device(device)
    )
    methods = _build_methods(checkpoint, device, time_limit_s)

    lengths: dict[str, list[float]] = {name: [] for name in methods}
    times_ms: dict[str, list[float]] = {name: [] for name in methods}
    gaps_pct: dict[str, list[float]] = {name: [] for name in methods}

    for idx in range(num_instances):
        coords = coords_batch[idx]
        per_instance_lengths: dict[str, float] = {}
        for name, solver in methods.items():
            start = time.perf_counter()
            _tour, length = solver(coords, seed + 1 + idx)
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            lengths[name].append(length)
            times_ms[name].append(elapsed_ms)
            per_instance_lengths[name] = length

        best = min(per_instance_lengths.values())
        for name, length in per_instance_lengths.items():
            gaps_pct[name].append(0.0 if best <= 0 else (length - best) / best * 100.0)

    return [
        MethodResult(
            name=name,
            lengths=lengths[name],
            times_ms=times_ms[name],
            gaps_pct=gaps_pct[name],
        )
        for name in methods
    ]


def format_table(results: list[MethodResult], num_instances: int, num_cities: int) -> str:
    """Render the aggregate comparison as a fixed-width text table."""
    header = (
        f"Benchmark: {num_instances} instances, N = {num_cities} cities, "
        f"OR-Tools {'available' if ORTOOLS_AVAILABLE else 'not installed'}"
    )
    cols = ["Method", "Mean len", "Std len", "Mean gap %", "Mean ms"]
    widths = [18, 10, 9, 11, 10]
    sep = "  ".join("-" * w for w in widths)
    lines = [header, "", "  ".join(c.ljust(w) for c, w in zip(cols, widths)), sep]
    for r in results:
        row = [
            METHOD_LABELS.get(r.name, r.name).ljust(widths[0]),
            f"{r.mean_length:.4f}".ljust(widths[1]),
            f"{r.std_length:.4f}".ljust(widths[2]),
            f"{r.mean_gap_pct:.2f}".ljust(widths[3]),
            f"{r.mean_time_ms:.2f}".ljust(widths[4]),
        ]
        lines.append("  ".join(row))
    return "\n".join(lines)


def write_csv(results: list[MethodResult], path: Path) -> None:
    """Write per-instance lengths and runtimes for every method to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    num_instances = len(results[0].lengths)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = ["instance"]
        for r in results:
            header += [f"{r.name}_length", f"{r.name}_ms"]
        writer.writerow(header)
        for i in range(num_instances):
            row: list[object] = [i]
            for r in results:
                row += [f"{r.lengths[i]:.6f}", f"{r.times_ms[i]:.4f}"]
            writer.writerow(row)


def _style_axes(ax) -> None:
    """Apply the shared parchment theme to one Matplotlib axis (clean, minimal)."""
    ax.set_facecolor(PANEL)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.yaxis.label.set_color(FOREGROUND)
    ax.xaxis.label.set_color(FOREGROUND)
    ax.title.set_color(FOREGROUND)
    ax.grid(True, axis="y", color=GRID, linewidth=0.8, alpha=0.6)
    ax.set_axisbelow(True)


def render_charts(
    results: list[MethodResult],
    path: Path,
    num_instances: int,
    num_cities: int,
) -> None:
    """Render parchment-themed quality and speed comparison charts to ``path``."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [r.name for r in results]
    labels = [METHOD_LABELS.get(n, n) for n in names]
    colors = [METHOD_COLORS.get(n, ACCENT) for n in names]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.patch.set_facecolor(BACKGROUND)
    fig.suptitle(
        f"Diffusion vs classical TSP solvers  -  {num_instances} instances, N = {num_cities}",
        color=FOREGROUND,
        fontsize=14,
        fontweight="bold",
    )

    # Panel 1: mean tour length with std error bars.
    ax = axes[0]
    means = [r.mean_length for r in results]
    stds = [r.std_length for r in results]
    bars = ax.bar(labels, means, color=colors, yerr=stds, ecolor=MUTED, capsize=4)
    ax.set_title("Tour length (lower is better)")
    ax.set_ylabel("Mean tour length")
    _style_axes(ax)
    _annotate_bars(ax, bars, means, "{:.3f}")

    # Panel 2: mean optimality gap.
    ax = axes[1]
    gaps = [r.mean_gap_pct for r in results]
    bars = ax.bar(labels, gaps, color=colors)
    ax.set_title("Optimality gap (lower is better)")
    ax.set_ylabel("Mean gap vs best (%)")
    _style_axes(ax)
    _annotate_bars(ax, bars, gaps, "{:.1f}%")

    # Panel 3: mean runtime per instance, log scale (orders of magnitude differ).
    ax = axes[2]
    times = [r.mean_time_ms for r in results]
    bars = ax.bar(labels, times, color=colors)
    ax.set_title("Runtime (lower is better)")
    ax.set_ylabel("Mean ms / instance")
    ax.set_yscale("log")
    _style_axes(ax)
    _annotate_bars(ax, bars, times, "{:.1f}")

    for ax in axes:
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=20, ha="right")

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, facecolor=BACKGROUND)
    plt.close(fig)


def _annotate_bars(ax, bars, values, fmt: str) -> None:
    """Write each bar's value just above it in muted text."""
    for bar, value in zip(bars, values):
        ax.annotate(
            fmt.format(value),
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            color=MUTED,
            fontsize=8,
        )


def _resolve_checkpoint(explicit: Optional[str]) -> str:
    """Pick a checkpoint: the explicit path, else the newest under ``checkpoints``."""
    if explicit:
        return explicit
    candidates = sorted(Path("checkpoints").glob("**/*.pt"))
    if not candidates:
        raise FileNotFoundError(
            "no checkpoint found under checkpoints/; pass --checkpoint explicitly"
        )
    return str(candidates[-1])


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark the diffusion TSP solver.")
    parser.add_argument("--checkpoint", default=None, help="Path to a trained checkpoint.")
    parser.add_argument("--num-instances", type=int, default=50)
    parser.add_argument("--num-cities", type=int, default=15)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--time-limit", type=float, default=2.0, help="OR-Tools seconds/instance.")
    parser.add_argument("--out-dir", default="benchmarks/results")
    args = parser.parse_args()

    checkpoint = _resolve_checkpoint(args.checkpoint)
    out_dir = Path(args.out_dir)

    print(f"Using checkpoint: {checkpoint}")
    results = run_benchmark(
        checkpoint=checkpoint,
        num_instances=args.num_instances,
        num_cities=args.num_cities,
        seed=args.seed,
        device=args.device,
        time_limit_s=args.time_limit,
    )

    table = format_table(results, args.num_instances, args.num_cities)
    print()
    print(table)

    write_csv(results, out_dir / "results.csv")
    render_charts(results, out_dir / "benchmark.png", args.num_instances, args.num_cities)
    print()
    print(f"Wrote {out_dir / 'results.csv'}")
    print(f"Wrote {out_dir / 'benchmark.png'}")


if __name__ == "__main__":
    main()
