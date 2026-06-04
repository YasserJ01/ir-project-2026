"""HTTP-level tests for the dense-retrieval service (port 8003).

Uses :class:`fastapi.testclient.TestClient` + the in-process
:func:`client` fixture from ``conftest.py`` so no real model or
FAISS file is needed.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from services.retrieval.app import service as service_mod

# ─────────────────────────────────────────────────────────────────────────
# /health
# ─────────────────────────────────────────────────────────────────────────


def test_health(client: TestClient) -> None:
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["service"] == "retrieval"
    assert body["model_loaded"] is True
    assert body["model_name"]


# ─────────────────────────────────────────────────────────────────────────
# /retrieval/{ds}/exists + /stats
# ─────────────────────────────────────────────────────────────────────────


def test_exists_true(client: TestClient) -> None:
    r = client.get("/retrieval/touche2020/exists")
    assert r.status_code == 200
    assert r.json() == {"exists": True}


def test_exists_false(client: TestClient) -> None:
    r = client.get("/retrieval/nq/exists")
    assert r.status_code == 200
    assert r.json() == {"exists": False}


def test_exists_unknown_dataset(client: TestClient) -> None:
    r = client.get("/retrieval/foo/exists")
    assert r.status_code == 400
    assert "Unknown dataset_id" in r.json()["detail"]


def test_stats_reads_meta(client: TestClient) -> None:
    r = client.get("/retrieval/touche2020/stats")
    assert r.status_code == 200
    body = r.json()
    assert body["dataset_id"] == "touche2020"
    assert body["exists"] is True
    assert body["num_vectors"] == 5
    assert body["dim"] == 16
    assert body["index_type"] == "IndexFlatIP"


def test_stats_unknown_dataset(client: TestClient) -> None:
    r = client.get("/retrieval/foo/stats")
    assert r.status_code == 400


def test_stats_missing_index(
    client: TestClient,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Point the service at a fresh empty dir so there's no FAISS index.
    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.setattr(service_mod, "index_dir", lambda ds: empty / ds)
    r = client.get("/retrieval/touche2020/stats")
    assert r.status_code == 200
    body = r.json()
    assert body["exists"] is False


# ─────────────────────────────────────────────────────────────────────────
# /retrieval/{ds}/search
# ─────────────────────────────────────────────────────────────────────────


def test_search_returns_hits(client: TestClient) -> None:
    r = client.post(
        "/retrieval/touche2020/search",
        json={"query": "fox", "k": 3},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["dataset_id"] == "touche2020"
    assert body["k"] == 3
    assert body["model_name"]
    assert isinstance(body["latency_ms"], int)
    assert len(body["results"]) > 0
    hit = body["results"][0]
    assert hit["rank"] == 1
    assert isinstance(hit["doc_id"], str)
    assert isinstance(hit["score"], float)


def test_search_missing_query_returns_422(client: TestClient) -> None:
    r = client.post("/retrieval/touche2020/search", json={"k": 3})
    assert r.status_code == 422
    assert "query" in r.json()["detail"]


def test_search_empty_query_returns_422(client: TestClient) -> None:
    r = client.post("/retrieval/touche2020/search", json={"query": "", "k": 3})
    assert r.status_code == 422


def test_search_invalid_k_returns_422(client: TestClient) -> None:
    r = client.post(
        "/retrieval/touche2020/search",
        json={"query": "fox", "k": 0},
    )
    assert r.status_code == 422
    r = client.post(
        "/retrieval/touche2020/search",
        json={"query": "fox", "k": 9999},
    )
    assert r.status_code == 422


def test_search_unknown_dataset_returns_400(client: TestClient) -> None:
    r = client.post("/retrieval/foo/search", json={"query": "fox", "k": 3})
    assert r.status_code == 400


# ─────────────────────────────────────────────────────────────────────────
# /retrieval/embed
# ─────────────────────────────────────────────────────────────────────────


def test_embed_one_text(client: TestClient) -> None:
    r = client.post("/retrieval/embed", json={"texts": ["hello world"]})
    assert r.status_code == 200
    body = r.json()
    assert body["model_name"]
    assert body["dim"] == 16
    assert len(body["vectors"]) == 1
    assert len(body["vectors"][0]) == 16


def test_embed_many_texts(client: TestClient) -> None:
    texts = [f"text {i}" for i in range(10)]
    r = client.post("/retrieval/embed", json={"texts": texts})
    assert r.status_code == 200
    body = r.json()
    assert len(body["vectors"]) == 10
    for v in body["vectors"]:
        assert len(v) == 16


def test_embed_empty_texts_returns_422(client: TestClient) -> None:
    r = client.post("/retrieval/embed", json={"texts": []})
    assert r.status_code == 422


# ─────────────────────────────────────────────────────────────────────────
# /load
# ─────────────────────────────────────────────────────────────────────────


def test_load_warms_cache(client: TestClient) -> None:
    """After /load, the dataset id is in the LRU cache."""
    r = client.post("/retrieval/touche2020/load")
    assert r.status_code == 200
    body = r.json()
    assert body["exists"] is True
    assert service_mod._LOADED_DATASET == "touche2020"


def test_load_unknown_dataset(client: TestClient) -> None:
    r = client.post("/retrieval/foo/load")
    assert r.status_code == 400


# ─────────────────────────────────────────────────────────────────────────
# /build (async via BackgroundTasks; we just check the 202 + job_id)
# ─────────────────────────────────────────────────────────────────────────


def test_build_accepted(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The /build endpoint kicks off a background encode + FAISS write.

    We monkeypatch ``_do_build`` to a no-op so the test doesn't
    touch the real 382K-doc corpus (or load the real 90 MB model).
    """

    def _noop_build(*args: object, **kwargs: object) -> None:
        return None

    monkeypatch.setattr(service_mod, "_do_build", _noop_build)
    r = client.post(
        "/retrieval/touche2020/build",
        json={"model_name": "fake-model", "batch_size": 8},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["dataset_id"] == "touche2020"
    assert body["started"] is True
    assert body["job_id"]


def test_build_unknown_dataset(client: TestClient) -> None:
    r = client.post("/retrieval/foo/build", json={})
    assert r.status_code == 400
