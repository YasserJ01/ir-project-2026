"""Tests for the FastAPI indexing service (in-process via TestClient)."""

from __future__ import annotations


def test_health(client) -> None:
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["service"] == "indexing"
    assert body["loaded_dataset"] is None


def test_exists_false_before_build(client) -> None:
    r = client.get(f"/index/{__import__('tests').indexing.conftest.TEST_DATASET_ID}/exists")
    # In this test we haven't built; the service has no files.
    # But the fake_index_dir fixture builds them, so this test
    # actually uses that fixture. See test_exists_true_after_build.
    # This test verifies the default behaviour without the fixture.
    # We can't easily skip the fixture here -- instead we test
    # /exists with a non-existent dataset_id.
    r = client.get("/index/nonexistent/exists")
    assert r.status_code == 400


def test_exists_true_after_build(client, fake_index_dir) -> None:
    from tests.indexing.conftest import TEST_DATASET_ID

    r = client.get(f"/index/{TEST_DATASET_ID}/exists")
    assert r.status_code == 200
    assert r.json() == {"exists": True}


def test_stats_after_build(client, fake_index_dir) -> None:
    from tests.indexing.conftest import TEST_DATASET_ID

    r = client.get(f"/index/{TEST_DATASET_ID}/stats")
    assert r.status_code == 200
    body = r.json()
    assert body["dataset_id"] == TEST_DATASET_ID
    assert body["exists"] is True
    assert body["total_docs"] == 5
    assert body["vocab_size"] == 3
    assert body["loaded"] is False  # not loaded yet
    assert body["cap"] == {"min_df": 1, "max_df_ratio": 1.0}


def test_stats_unknown_dataset_returns_400(client) -> None:
    r = client.get("/index/nonexistent/stats")
    assert r.status_code == 400


def test_search_bm25_returns_results(client, fake_index_dir) -> None:
    from tests.indexing.conftest import TEST_DATASET_ID

    r = client.post(
        f"/index/{TEST_DATASET_ID}/search",
        json={"query_tokens": ["fox"], "model": "bm25", "k": 3},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["dataset_id"] == TEST_DATASET_ID
    assert body["model"] == "bm25"
    assert body["k"] == 3
    assert body["latency_ms"] >= 0
    assert len(body["results"]) > 0
    assert body["results"][0]["rank"] == 1
    assert body["results"][0]["doc_id"] in {"d1", "d3", "d5"}
    assert body["k1"] == 1.5
    assert body["b"] == 0.75


def test_search_tfidf_returns_results(client, fake_index_dir) -> None:
    from tests.indexing.conftest import TEST_DATASET_ID

    r = client.post(
        f"/index/{TEST_DATASET_ID}/search",
        json={"query_tokens": ["fox"], "model": "tfidf", "k": 3},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["model"] == "tfidf"
    assert body["k1"] is None
    assert body["b"] is None
    assert len(body["results"]) > 0


def test_search_inverted_returns_results(client, fake_index_dir) -> None:
    from tests.indexing.conftest import TEST_DATASET_ID

    r = client.post(
        f"/index/{TEST_DATASET_ID}/search",
        json={"query_tokens": ["fox"], "model": "inverted", "k": 5},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["model"] == "inverted"
    # d1, d3, d5 contain "fox" -- at least 1 should be returned.
    assert len(body["results"]) > 0


def test_search_bad_model_returns_422(client, fake_index_dir) -> None:
    from tests.indexing.conftest import TEST_DATASET_ID

    r = client.post(
        f"/index/{TEST_DATASET_ID}/search",
        json={"query_tokens": ["fox"], "model": "not_a_model", "k": 3},
    )
    # Pydantic v2: Literal validation -> 422
    assert r.status_code == 422


def test_search_unknown_dataset_returns_400(client) -> None:
    r = client.post(
        "/index/nonexistent/search",
        json={"query_tokens": ["fox"], "model": "bm25", "k": 3},
    )
    assert r.status_code == 400


def test_search_under_1s_for_top_10(client, fake_index_dir) -> None:
    """Phase 2 exit criterion: < 1s for top-10 on the small fixture."""
    from tests.indexing.conftest import TEST_DATASET_ID

    r = client.post(
        f"/index/{TEST_DATASET_ID}/search",
        json={"query_tokens": ["fox", "cat", "dog"], "model": "bm25", "k": 10},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["latency_ms"] < 1000  # very generous for a 5-doc fixture


def test_load_warms_cache(client, fake_index_dir) -> None:
    from tests.indexing.conftest import TEST_DATASET_ID

    r = client.post(f"/index/{TEST_DATASET_ID}/load")
    assert r.status_code == 200
    body = r.json()
    assert body["loaded"] is True
    assert body["took_seconds"] >= 0
    # /health should now report the dataset as loaded
    h = client.get("/health").json()
    assert h["loaded_dataset"] == TEST_DATASET_ID


def test_postings_returns_postings(client, fake_index_dir) -> None:
    from tests.indexing.conftest import TEST_DATASET_ID

    r = client.get(f"/index/{TEST_DATASET_ID}/postings/fox")
    assert r.status_code == 200
    body = r.json()
    assert body["term"] == "fox"
    assert body["doc_freq"] == 3  # d1, d3, d5
    assert len(body["postings"]) == 3
    assert body["truncated"] is False
    # tf sum for fox = 2 + 1 + 3 = 6
    assert sum(p["tf"] for p in body["postings"]) == 6


def test_postings_cap_truncates(client, fake_index_dir) -> None:
    from tests.indexing.conftest import TEST_DATASET_ID

    r = client.get(f"/index/{TEST_DATASET_ID}/postings/fox?cap=2")
    assert r.status_code == 200
    body = r.json()
    assert body["doc_freq"] == 3  # total still reported
    assert len(body["postings"]) == 2
    assert body["truncated"] is True


def test_postings_missing_term_returns_empty(client, fake_index_dir) -> None:
    from tests.indexing.conftest import TEST_DATASET_ID

    r = client.get(f"/index/{TEST_DATASET_ID}/postings/nonexistent_xyz")
    assert r.status_code == 200
    body = r.json()
    assert body["doc_freq"] == 0
    assert body["postings"] == []
    assert body["truncated"] is False
