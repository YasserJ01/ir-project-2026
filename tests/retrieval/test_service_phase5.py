"""Service-level tests for the Phase 5 hybrid / multi-encoder endpoints.

These tests build a minimal in-memory world so they don't need a real
FAISS index on disk. They exercise the routing, error handling, and
response shapes of the 3 new endpoints:

  * ``POST /hybrid/{ds}/search``
  * ``POST /multi-encoder/{ds}/search``
  * ``GET  /hybrid/{ds}/health``
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from services.retrieval.app import config as config_mod
from services.retrieval.app import hybrid as hybrid_mod
from services.retrieval.app import multi_encoder as me_mod
from services.retrieval.app import service as service_mod
from services.retrieval.app import vector_store as vector_store_mod
from shared.ir_common.schemas import (
    HybridSearchHit,
    HybridSearchRequest,
    HybridSearchResponse,
    MultiEncoderSearchRequest,
)

# ─────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────


@pytest.fixture
def fake_faiss(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Make ``index_dir()`` return ``tmp_path``.

    We don't create any actual index files -- the test only needs the
    path resolution to be predictable. Endpoints that need an index
    present are mocked separately.

    The service module imports ``index_dir`` at top-level, so we have
    to patch BOTH ``config_mod.index_dir`` AND ``service_mod.index_dir``
    (the latter is the reference the endpoint actually uses).
    """
    monkeypatch.setattr(config_mod, "index_dir", lambda ds: tmp_path)
    monkeypatch.setattr(service_mod, "index_dir", lambda ds: tmp_path)
    return tmp_path


@pytest.fixture
def monkeypatch_second_encoder_exists(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pretend the 2nd-encoder FAISS index is on disk."""
    monkeypatch.setattr(config_mod, "has_second_encoder_index", lambda ds: True)


@pytest.fixture
def monkeypatch_second_encoder_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pretend the 2nd-encoder FAISS index is NOT on disk."""
    monkeypatch.setattr(config_mod, "has_second_encoder_index", lambda ds: False)


@pytest.fixture
def client() -> TestClient:
    return TestClient(service_mod.app)


# ─────────────────────────────────────────────────────────────────────────
# Regression tests for the real _dense_search_closure()
# ─────────────────────────────────────────────────────────────────────────


def test_dense_search_closure_signature_matches_dense_search_fn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression: the real closure must accept ``(query_text, dataset_id, k, model_name)``.

    The :data:`HybridOrchestrator.DenseSearchFn` type alias is the contract.
    An earlier version of the closure had the parameters in the wrong order
    (``model_name`` came before ``k``), which made the embedding and
    hybrid_parallel representations crash with
    ``'int' object has no attribute 'replace'`` at runtime. This test
    calls the real closure with ``k=3`` and ``model_name=None`` and
    verifies that the embedder receives the model_name (not the int ``k``).
    """
    # Force the closure to rebuild (it's a module-level singleton).
    service_mod._DENSE_CLOSURE = None

    captured: dict[str, Any] = {}

    class FakeEmbedder:
        def encode_query(self, text: str, model_name: str | None = None) -> Any:
            captured["text"] = text
            captured["model_name"] = model_name
            import numpy as np

            return np.zeros(4, dtype=np.float32)

    class FakeIndex:
        def __init__(self) -> None:
            self.doc_ids = ["d1", "d2", "d3", "d4"]

        def search(self, _q: Any, k: int) -> tuple[list[float], list[str]]:
            return [0.9, 0.8, 0.7], [0, 1, 2]

    monkeypatch.setattr(service_mod, "_embedder", lambda: FakeEmbedder())
    monkeypatch.setattr(service_mod, "_load_faiss", lambda ds, **kw: FakeIndex())

    closure = service_mod._dense_search_closure()
    import asyncio

    scores, ids = asyncio.run(closure("hello world", "touche2020", 3, None))
    assert captured["model_name"] is None
    assert captured["text"] == "hello world"
    assert scores == [0.9, 0.8, 0.7]
    assert ids == ["d1", "d2", "d3"]


def test_dense_search_closure_with_l12_picks_l12_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When ``model_name`` matches the L12 encoder name, the closure must
    request the L12-specific index files."""
    service_mod._DENSE_CLOSURE = None
    captured: dict[str, Any] = {}

    class FakeEmbedder:
        def encode_query(self, text: str, model_name: str | None = None) -> Any:
            import numpy as np

            return np.zeros(4, dtype=np.float32)

    def fake_load(ds: str, **kw: Any) -> Any:
        captured.update(kw)
        captured["ds"] = ds

        class _Idx:
            doc_ids = ["d1", "d2"]

            def search(self, _q: Any, k: int) -> tuple[list[float], list[str]]:
                return [0.5], [0]

        return _Idx()

    monkeypatch.setattr(service_mod, "_embedder", lambda: FakeEmbedder())
    monkeypatch.setattr(service_mod, "_load_faiss", fake_load)

    closure = service_mod._dense_search_closure()
    import asyncio

    asyncio.run(closure("hi", "nq", 1, me_mod.DEFAULT_ENCODER_2))
    assert captured["ds"] == "nq"
    assert captured.get("index_filename") == "faiss_l12.index"
    assert captured.get("embeddings_filename") == "embeddings_l12.npy"


# ─────────────────────────────────────────────────────────────────────────
# /hybrid/{ds}/health
# ─────────────────────────────────────────────────────────────────────────


def test_hybrid_health_unknown_dataset(client: TestClient, fake_faiss: Path) -> None:
    r = client.get("/hybrid/not-a-real-dataset/health")
    assert r.status_code == 400
    assert "Unknown dataset_id" in r.json()["detail"]


def test_hybrid_health_known_dataset_no_artifacts(
    client: TestClient, fake_faiss: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No FAISS, no docs.jsonl -- everything is False."""
    # Stub the upstream probe so the test doesn't depend on whether
    # :8002 / :8004 happen to be running in the test environment.
    monkeypatch.setattr(service_mod, "_probe_upstreams", lambda: (False, False))
    r = client.get("/hybrid/touche2020/health")
    assert r.status_code == 200
    body = r.json()
    assert body["dataset_id"] == "touche2020"
    assert body["dense_loaded"] is False
    assert body["second_encoder_built"] is False
    # The /health probes are stubbed -- both flags should be False.
    assert body["bm25_endpoint_reachable"] is False
    assert body["refinement_endpoint_reachable"] is False


def test_hybrid_health_with_docs_no_faiss(
    client: TestClient, fake_faiss: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """FAISS L6 present -> dense_loaded True; second encoder False."""
    (fake_faiss / vector_store_mod.INDEX_FILENAME).write_bytes(b"")
    monkeypatch.setattr(config_mod, "has_second_encoder_index", lambda ds: False)
    r = client.get("/hybrid/touche2020/health")
    assert r.status_code == 200
    body = r.json()
    assert body["dense_loaded"] is True
    assert body["second_encoder_built"] is False


# ─────────────────────────────────────────────────────────────────────────
# /hybrid/{ds}/search -- dispatch + error paths
# ─────────────────────────────────────────────────────────────────────────


def test_hybrid_search_unknown_dataset(client: TestClient) -> None:
    r = client.post(
        "/hybrid/wat/search",
        json={"query": "test", "k": 5, "representation": "tfidf"},
    )
    assert r.status_code == 400


def test_hybrid_search_k_out_of_range(client: TestClient) -> None:
    r = client.post(
        "/hybrid/touche2020/search",
        json={"query": "test", "k": 0, "representation": "tfidf"},
    )
    # Pydantic validation: 422.
    assert r.status_code == 422


def test_hybrid_search_dispatches_to_orchestrator(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The endpoint must call the orchestrator's search() method."""

    called: dict[str, Any] = {}

    async def fake_search(
        self: Any, dataset_id: str, req: HybridSearchRequest
    ) -> HybridSearchResponse:
        called["dataset_id"] = dataset_id
        called["representation"] = req.representation
        called["fusion"] = req.fusion
        return HybridSearchResponse(
            dataset_id=dataset_id,
            representation=req.representation,
            fusion=req.fusion,
            k=req.k,
            latency_ms=1,
            results=[
                HybridSearchHit(rank=1, doc_id="d1", score=1.0),
            ],
            per_retriever_latency_ms={"bm25": 1},
            refined_query=None,
            stages={"bm25": "fake"},
        )

    monkeypatch.setattr(hybrid_mod.HybridOrchestrator, "search", fake_search)
    r = client.post(
        "/hybrid/touche2020/search",
        json={
            "query": "test",
            "k": 3,
            "representation": "hybrid_parallel",
            "fusion": "rrf",
            "candidate_k": 10,
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["dataset_id"] == "touche2020"
    assert body["representation"] == "hybrid_parallel"
    assert body["fusion"] == "rrf"
    assert called["dataset_id"] == "touche2020"
    assert called["representation"] == "hybrid_parallel"
    assert called["fusion"] == "rrf"


# ─────────────────────────────────────────────────────────────────────────
# /multi-encoder/{ds}/search -- error paths
# ─────────────────────────────────────────────────────────────────────────


def test_multi_encoder_search_unknown_dataset(client: TestClient) -> None:
    r = client.post(
        "/multi-encoder/foo/search",
        json={"query": "test", "k": 5},
    )
    assert r.status_code == 400


def test_multi_encoder_search_503_when_l12_missing(
    client: TestClient,
    monkeypatch_second_encoder_missing: None,
) -> None:
    r = client.post(
        "/multi-encoder/touche2020/search",
        json={"query": "test", "k": 5},
    )
    assert r.status_code == 503
    assert "not built" in r.json()["detail"]


def test_multi_encoder_search_400_when_same_encoder(
    client: TestClient,
    monkeypatch_second_encoder_exists: None,
) -> None:
    r = client.post(
        "/multi-encoder/touche2020/search",
        json={
            "query": "test",
            "k": 5,
            "encoder_1": "sentence-transformers/all-MiniLM-L6-v2",
            "encoder_2": "sentence-transformers/all-MiniLM-L6-v2",
        },
    )
    assert r.status_code == 400
    assert "must be different" in r.json()["detail"]


def test_multi_encoder_search_dispatches_to_runner(
    client: TestClient,
    monkeypatch_second_encoder_exists: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The endpoint must call the runner's search() method with a stub."""
    called: dict[str, Any] = {}

    async def fake_search(
        self: Any, dataset_id: str, req: MultiEncoderSearchRequest
    ) -> HybridSearchResponse:
        called["dataset_id"] = dataset_id
        called["fusion"] = req.fusion
        return HybridSearchResponse(
            dataset_id=dataset_id,
            representation="embedding",
            fusion=req.fusion,
            k=req.k,
            latency_ms=2,
            results=[HybridSearchHit(rank=1, doc_id="dx", score=1.0)],
            per_retriever_latency_ms={"l6": 1, "l12": 1},
            refined_query=None,
            stages={"l6": "fake", "l12": "fake"},
        )

    monkeypatch.setattr(me_mod.MultiEncoderRunner, "search", fake_search)
    r = client.post(
        "/multi-encoder/touche2020/search",
        json={"query": "test", "k": 3, "fusion": "combsum"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["fusion"] == "combsum"
    assert called["dataset_id"] == "touche2020"
    assert called["fusion"] == "combsum"
