"""Synthetic TSP instance generator.

Samples batches of 2D city coordinates in the unit square for training and
evaluation.

Coordinates are returned as a float tensor of shape ``(num_instances, num_cities, 2)``
with every value in ``[0, 1)``. Passing a ``seed`` makes generation deterministic.
"""

from __future__ import annotations

from typing import Optional

import torch


def generate_instances(
    num_cities: int,
    num_instances: int = 1,
    seed: Optional[int] = None,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Generate a batch of random TSP instances.

    Each instance is a set of ``num_cities`` points sampled uniformly from the
    unit square ``[0, 1) x [0, 1)``.

    Args:
        num_cities: Number of cities per instance. Must be >= 1.
        num_instances: Number of independent instances to generate. Must be >= 1.
        seed: Optional seed for reproducibility. The same seed (with the same
            shape arguments) always yields the same coordinates.
        device: Optional torch device for the returned tensor.
        dtype: Floating point dtype for the returned tensor.

    Returns:
        A tensor of shape ``(num_instances, num_cities, 2)``.
    """
    if num_cities < 1:
        raise ValueError(f"num_cities must be >= 1, got {num_cities}")
    if num_instances < 1:
        raise ValueError(f"num_instances must be >= 1, got {num_instances}")

    generator = None
    if seed is not None:
        # Use a local generator so callers' global RNG state is untouched.
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))

    coords = torch.rand(
        (num_instances, num_cities, 2),
        generator=generator,
        dtype=dtype,
    )
    if device is not None:
        coords = coords.to(device)
    return coords


if __name__ == "__main__":
    sample = generate_instances(num_cities=5, num_instances=3, seed=0)
    print("shape:", tuple(sample.shape))
    print(sample)
