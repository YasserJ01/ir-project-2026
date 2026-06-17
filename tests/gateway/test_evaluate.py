"""Route tests for ``POST /api/evaluate`` (Phase 8b live evaluation).

We mock ``services.gateway.app.evaluate.run_evaluation`` so the test
is fast and deterministic (no real queries, no ir_measures calls).
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

from services.gateway.app import evaluate as evaluate_mod


@pytest.fixture(autouse=True)
def _mock_evaluate(monkeypatch: pytest.MonkeyPatch) -> AsyncMock:
    """Replace the real ``run_evaluation`` with a mock that returns
    canned metrics."""
    mock = AsyncMock(return_value={
        "dataset_id": "nq",
        "representation": "bm25",
        "condition": "baseline",
        "queries": 200,
        "success": 198,
        "errors": 2,
        "time_s": 4.5,
        "metrics": {
            "MAP": 0.2930,
            "P@10": 0.0610,
            "nDCG@10": 0.3540,
            "R@10": 0.5183,
        },
    })
    monkeypatch.setattr(evaluate_mod, "run_evaluation", mock)
    return mock


# ─────────────────────────────────────────────────────────────────────────
# POST /api/evaluate — success
# ─────────────────────────────────────────────────────────────────────────


def test_evaluate_basic(client: TestClient, _mock_evaluate: AsyncMock) -> None:
    r = client.post("/api/evaluate", json={
        "dataset_id": "nq",
        "representation": "bm25",
        "mode": "basic",
        "fusion": "rrf",
    })
    assert r.status_code == 200
    body = r.json()
    assert body["dataset_id"] == "nq"
    assert body["success"] == 198
    assert body["metrics"]["MAP"] == 0.2930
    assert body["metrics"]["P@10"] == 0.0610
    assert _mock_evaluate.await_count == 1


def test_evaluate_with_features(client: TestClient, _mock_evaluate: AsyncMock) -> None:
    r = client.post("/api/evaluate", json={
        "dataset_id": "touche2020",
        "representation": "embedding",
        "mode": "with_features",
        "fusion": "combsum",
        "bm25_k1": 1.2,
        "bm25_b": 0.5,
        "use_multi": True,
    })
    assert r.status_code == 200
    body = r.json()
    assert body["dataset_id"] == "nq"  # from the canned return
    assert _mock_evaluate.await_count == 1
    # Verify the mock was called with the right params
    call_kwargs = _mock_evaluate.call_args[1]
    assert call_kwargs["dataset_id"] == "touche2020"
    assert call_kwargs["representation"] == "embedding"
    assert call_kwargs["mode"] == "with_features"
    assert call_kwargs["fusion"] == "combsum"
    assert call_kwargs["bm25_k1"] == 1.2
    assert call_kwargs["bm25_b"] == 0.5
    assert call_kwargs["use_multi"] is True


# ─────────────────────────────────────────────────────────────────────────
# POST /api/evaluate — error
# ─────────────────────────────────────────────────────────────────────────


def test_evaluate_downstream_failure(
    client: TestClient, _mock_evaluate: AsyncMock
) -> None:
    _mock_evaluate.side_effect = RuntimeError("Backend unreachable")
    r = client.post("/api/evaluate", json={
        "dataset_id": "nq",
        "representation": "bm25",
    })
    assert r.status_code == 500
    body = r.json()
    assert "Backend unreachable" in body["detail"]


# ─────────────────────────────────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────────────────────────────────


def test_evaluate_invalid_dataset(client: TestClient) -> None:
    r = client.post("/api/evaluate", json={
        "dataset_id": "nonexistent",
        "representation": "bm25",
    })
    assert r.status_code == 422


def test_evaluate_missing_dataset(client: TestClient) -> None:
    r = client.post("/api/evaluate", json={
        "representation": "bm25",
    })
    assert r.status_code == 422
