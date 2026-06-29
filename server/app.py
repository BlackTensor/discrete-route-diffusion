"""FastAPI server.

Exposes endpoints to generate TSP instances, run reverse-diffusion inference,
and return the step-by-step denoising timeline, and serves the static frontend.

Endpoints
---------
``GET  /``                 Serve the single-page frontend (``static/index.html``).
``GET  /api/health``       Liveness plus model/checkpoint status.
``POST /api/generate``     Sample a fresh instance, returns city coordinates.
``POST /api/infer``        Denoise an instance, returns the animation timeline.
``POST /api/run``          Convenience: generate then infer in one call.

The trained denoiser is loaded lazily from a checkpoint (resolved from the
``ROUTEDIFF_CHECKPOINT`` environment variable, or the most recently modified
``*.pt`` under ``checkpoints/``) and cached for reuse across requests. Because
the model is a graph network it runs on any number of cities at inference time,
independent of the size it was trained on.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Optional

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from data.generator import generate_instances
from routediff.diffusion import DiffusionSchedule
from routediff.inference import build_timeline, load_checkpoint, reverse_sample
from routediff.model import RouteDiffusionDenoiser

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
PROJECT_ROOT = BASE_DIR.parent
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
LOG_DIR = PROJECT_ROOT / "logs"

# Matches training-log lines like "epoch 12/250  loss=0.41234".
_EPOCH_RE = re.compile(r"epoch\s+(\d+)\s*/\s*(\d+)\s+loss=([0-9.eE+-]+)")

MIN_CITIES = 4
MAX_CITIES = 60

app = FastAPI(
    title="RouteDiff",
    description="Discrete diffusion for the Traveling Salesperson Problem.",
    version="0.1.0",
)

# Lazily populated cache of the loaded model and its noise schedule.
_model_cache: dict[str, object] = {}


def _find_checkpoint() -> Optional[Path]:
    """Resolve the checkpoint to load.

    Prefers the ``ROUTEDIFF_CHECKPOINT`` environment variable; otherwise returns
    the most recently modified ``*.pt`` file found anywhere under
    ``checkpoints/``. Returns ``None`` if nothing is available.
    """
    env_path = os.environ.get("ROUTEDIFF_CHECKPOINT")
    if env_path:
        path = Path(env_path)
        return path if path.is_file() else None

    if not CHECKPOINT_DIR.is_dir():
        return None
    candidates = sorted(
        CHECKPOINT_DIR.rglob("*.pt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def _get_model() -> tuple[RouteDiffusionDenoiser, DiffusionSchedule]:
    """Return the cached (model, schedule), loading from a checkpoint on first use.

    Raises:
        HTTPException: 503 if no checkpoint is available to load.
    """
    if "model" in _model_cache:
        return _model_cache["model"], _model_cache["schedule"]  # type: ignore[return-value]

    ckpt = _find_checkpoint()
    if ckpt is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "No model checkpoint found. Train a model (routediff.train) or set "
                "ROUTEDIFF_CHECKPOINT to a checkpoint path."
            ),
        )
    model, schedule, _ = load_checkpoint(str(ckpt), device="cpu")
    _model_cache["model"] = model
    _model_cache["schedule"] = schedule
    _model_cache["path"] = str(ckpt)
    return model, schedule


class GenerateRequest(BaseModel):
    """Request body for generating a fresh instance."""

    num_cities: int = Field(default=15, ge=MIN_CITIES, le=MAX_CITIES)
    seed: Optional[int] = None


class InstanceResponse(BaseModel):
    """A generated TSP instance."""

    num_cities: int
    coords: list[list[float]]


class InferRequest(BaseModel):
    """Request body for running inference on a supplied instance."""

    coords: list[list[float]]
    seed: Optional[int] = None


class RunRequest(BaseModel):
    """Request body for the generate-then-infer convenience endpoint."""

    num_cities: int = Field(default=15, ge=MIN_CITIES, le=MAX_CITIES)
    seed: Optional[int] = None


def _coords_to_tensor(coords: list[list[float]]) -> torch.Tensor:
    """Validate and convert a list of ``[x, y]`` pairs into a ``(N, 2)`` tensor."""
    if not coords or len(coords) < MIN_CITIES:
        raise HTTPException(
            status_code=422,
            detail=f"need at least {MIN_CITIES} cities, got {len(coords)}",
        )
    if len(coords) > MAX_CITIES:
        raise HTTPException(
            status_code=422,
            detail=f"at most {MAX_CITIES} cities, got {len(coords)}",
        )
    if any(len(p) != 2 for p in coords):
        raise HTTPException(status_code=422, detail="each city must be an [x, y] pair")
    return torch.tensor(coords, dtype=torch.float32)


def _run_inference(coords: torch.Tensor, seed: Optional[int]) -> dict:
    """Run reverse sampling on ``coords`` and build the animation timeline."""
    model, schedule = _get_model()
    generator: Optional[torch.Generator] = None
    if seed is not None:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))
    result = reverse_sample(
        model, coords, schedule, generator=generator, record_history=True
    )
    return build_timeline(coords, result)


def _find_train_log() -> Optional[Path]:
    """Resolve the training log to read for the loss curve.

    Prefers ``ROUTEDIFF_TRAIN_LOG``; otherwise the most recently modified
    ``*.log`` under ``logs/`` that actually contains epoch/loss lines.
    """
    env_path = os.environ.get("ROUTEDIFF_TRAIN_LOG")
    if env_path:
        path = Path(env_path)
        return path if path.is_file() else None

    if not LOG_DIR.is_dir():
        return None
    candidates = sorted(LOG_DIR.glob("*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
    with_epochs = [
        p
        for p in candidates
        if _EPOCH_RE.search(p.read_text(encoding="utf-8", errors="ignore"))
    ]
    if not with_epochs:
        return None
    # Prefer real training logs over short test artifacts; newest within the
    # preferred group wins.
    preferred = [p for p in with_epochs if "test" not in p.stem.lower()]
    return (preferred or with_epochs)[0]


def _parse_loss_curve(path: Path) -> tuple[list[int], list[float]]:
    """Extract ``(epochs, losses)`` from a training log file."""
    epochs: list[int] = []
    losses: list[float] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        match = _EPOCH_RE.search(line)
        if match is None:
            continue
        try:
            epochs.append(int(match.group(1)))
            losses.append(float(match.group(3)))
        except ValueError:
            continue
    return epochs, losses


@app.get("/api/loss-curve")
def loss_curve() -> dict:
    """Return the training loss curve parsed from the training log.

    Reads per-epoch loss values logged by :mod:`routediff.train` and returns
    them as parallel ``steps`` (epoch numbers) and ``loss`` arrays for the UI
    training-loss chart.
    """
    log_path = _find_train_log()
    if log_path is None:
        raise HTTPException(
            status_code=503,
            detail="No training log with loss values found under logs/.",
        )
    epochs, losses = _parse_loss_curve(log_path)
    if not losses:
        raise HTTPException(status_code=503, detail="Training log has no loss entries.")
    return {
        "source": log_path.name,
        "num_points": len(losses),
        "steps": epochs,
        "loss": losses,
    }


@app.get("/api/health")
def health() -> dict:
    """Report liveness and whether a model checkpoint is available."""
    ckpt = _find_checkpoint()
    return {
        "status": "ok",
        "model_loaded": "model" in _model_cache,
        "checkpoint": str(ckpt) if ckpt is not None else None,
    }


@app.post("/api/generate", response_model=InstanceResponse)
def generate(req: GenerateRequest) -> InstanceResponse:
    """Generate a fresh random TSP instance in the unit square."""
    coords = generate_instances(
        num_cities=req.num_cities, num_instances=1, seed=req.seed
    )[0]
    return InstanceResponse(
        num_cities=req.num_cities,
        coords=coords.to(torch.float32).tolist(),
    )


@app.post("/api/infer")
def infer(req: InferRequest) -> dict:
    """Run reverse-diffusion inference on a supplied instance, return the timeline."""
    coords = _coords_to_tensor(req.coords)
    return _run_inference(coords, req.seed)


@app.post("/api/run")
def run(req: RunRequest) -> dict:
    """Generate a new instance and immediately return its denoising timeline."""
    coords = generate_instances(
        num_cities=req.num_cities, num_instances=1, seed=req.seed
    )[0]
    return _run_inference(coords, req.seed)


@app.get("/")
def index() -> FileResponse:
    """Serve the single-page frontend."""
    index_path = STATIC_DIR / "index.html"
    if not index_path.is_file():
        raise HTTPException(status_code=404, detail="frontend not built yet")
    return FileResponse(index_path)


# Serve CSS/JS and any other static assets under /static.
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
