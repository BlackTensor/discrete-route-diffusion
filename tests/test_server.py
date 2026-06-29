"""Tests for the FastAPI server (Task 6.1).

The model cache is populated with a small untrained denoiser so the endpoints
can be exercised without depending on a (gitignored) checkpoint on disk. Output
quality is irrelevant here -- these tests assert the HTTP contract: valid JSON,
correct shapes, validation behavior, and that the frontend is served.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from routediff.diffusion import make_schedule
from routediff.model import RouteDiffusionDenoiser
from server import app as app_module
from server.app import app


@pytest.fixture()
def client() -> TestClient:
    """A TestClient with a tiny model preloaded into the server cache."""
    model = RouteDiffusionDenoiser(hidden_dim=16, num_layers=2)
    model.eval()
    schedule = make_schedule(8, kind="cosine")
    app_module._model_cache.clear()
    app_module._model_cache["model"] = model
    app_module._model_cache["schedule"] = schedule
    app_module._model_cache["path"] = "test://tiny"
    try:
        yield TestClient(app)
    finally:
        app_module._model_cache.clear()


def test_health_reports_loaded_model(client: TestClient) -> None:
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_generate_returns_instance(client: TestClient) -> None:
    r = client.post("/api/generate", json={"num_cities": 9, "seed": 1})
    assert r.status_code == 200
    body = r.json()
    assert body["num_cities"] == 9
    assert len(body["coords"]) == 9
    assert all(len(p) == 2 for p in body["coords"])


def test_generate_is_reproducible(client: TestClient) -> None:
    a = client.post("/api/generate", json={"num_cities": 7, "seed": 42}).json()
    b = client.post("/api/generate", json={"num_cities": 7, "seed": 42}).json()
    assert a["coords"] == b["coords"]


def test_generate_rejects_out_of_range(client: TestClient) -> None:
    assert client.post("/api/generate", json={"num_cities": 2}).status_code == 422
    assert client.post("/api/generate", json={"num_cities": 999}).status_code == 422


def test_infer_returns_timeline(client: TestClient) -> None:
    inst = client.post("/api/generate", json={"num_cities": 8, "seed": 3}).json()
    r = client.post("/api/infer", json={"coords": inst["coords"], "seed": 0})
    assert r.status_code == 200
    tl = r.json()
    assert tl["num_cities"] == 8
    assert tl["num_steps"] == len(tl["steps"])
    # Timeline runs from t = T down to t = 0 and ends on the decoded tour.
    assert tl["steps"][0]["t"] >= tl["steps"][-2]["t"]
    assert tl["steps"][-1].get("final") is True
    assert sorted(tl["tour"]) == list(range(8))
    assert tl["length"] > 0


def test_infer_seed_is_reproducible(client: TestClient) -> None:
    inst = client.post("/api/generate", json={"num_cities": 8, "seed": 3}).json()
    a = client.post("/api/infer", json={"coords": inst["coords"], "seed": 7}).json()
    b = client.post("/api/infer", json={"coords": inst["coords"], "seed": 7}).json()
    assert a["tour"] == b["tour"]


def test_infer_rejects_too_few_cities(client: TestClient) -> None:
    r = client.post("/api/infer", json={"coords": [[0.1, 0.2], [0.3, 0.4]]})
    assert r.status_code == 422


def test_infer_rejects_malformed_points(client: TestClient) -> None:
    bad = [[0.1, 0.2], [0.3], [0.5, 0.6], [0.7, 0.8]]
    r = client.post("/api/infer", json={"coords": bad})
    assert r.status_code == 422


def test_run_generates_and_infers(client: TestClient) -> None:
    r = client.post("/api/run", json={"num_cities": 10, "seed": 5})
    assert r.status_code == 200
    tl = r.json()
    assert tl["num_cities"] == 10
    assert sorted(tl["tour"]) == list(range(10))


def test_index_and_static_served(client: TestClient) -> None:
    r = client.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    assert client.get("/static/style.css").status_code == 200


def test_missing_checkpoint_returns_503(monkeypatch) -> None:
    monkeypatch.setenv("ROUTEDIFF_CHECKPOINT", "does-not-exist.pt")
    app_module._model_cache.clear()
    c = TestClient(app)
    r = c.post("/api/run", json={"num_cities": 8})
    assert r.status_code == 503
