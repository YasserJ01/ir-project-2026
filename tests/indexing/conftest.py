"""Shared fixtures for the indexing service tests.

The service's hot-path endpoints (stats, load, search, postings) all
read from ``data/indexes/<dataset_id>/``. To keep these tests fast and
independent of whether the real indexes have been built, we use
``monkeypatch`` to redirect ``services.indexing.app.config.INDEX_ROOT``
(and therefore ``index_dir``) to a pytest-provided ``tmp_path`` that
we populate with a small in-memory-built index.

We also stub ``_is_known`` to accept the synthetic dataset id
``"testds"`` so we don't have to touch ``DATASET_IDS``.
"""

from __future__ import annotations

import pytest

from services.indexing.app import bm25 as bm25_mod
from services.indexing.app import config as config_mod
from services.indexing.app import inverted_index as inverted_index_mod
from services.indexing.app import service as service_mod
from services.indexing.app import tfidf as tfidf_mod

# ─────────────────────────────────────────────────────────────────────────
# Tiny fixture corpus
# ─────────────────────────────────────────────────────────────────────────

CORPUS: list[list[str]] = [
    ["fox", "fox", "dog"],
    ["cat", "dog"],
    ["fox", "cat"],
    ["dog"],
    ["fox", "fox", "fox", "cat"],
]
DOC_IDS: list[str] = ["d1", "d2", "d3", "d4", "d5"]
TEST_DATASET_ID = "testds"


@pytest.fixture
def fake_index_dir(tmp_path, monkeypatch):
    """Build a tiny in-memory index under tmp_path, redirect the service to it.

    Yields the tmp_path so individual tests can introspect or add files.
    """
    idx_dir = tmp_path / "indexes" / TEST_DATASET_ID
    idx_dir.mkdir(parents=True)

    # Build the InvertedIndex, TF-IDF, and BM25 over the tiny fixture
    # and save them under the temp dir.
    inv = inverted_index_mod.InvertedIndex(min_df=1, max_df_ratio=1.0)
    inv.build(list(zip(DOC_IDS, CORPUS, strict=True)))
    inv.save(idx_dir / inverted_index_mod.INDEX_FILENAME)

    tfidf = tfidf_mod.TfidfRetriever()
    tfidf.build(CORPUS, DOC_IDS)
    tfidf.save(idx_dir)

    bm25 = bm25_mod.BM25Retriever()
    bm25.build(CORPUS, DOC_IDS, k1=1.5, b=0.75, show_progress=False)
    bm25.save(idx_dir)

    # build_meta.json so /stats can show build time
    import json
    import time

    (idx_dir / "build_meta.json").write_text(
        json.dumps(
            {
                "dataset_id": TEST_DATASET_ID,
                "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "elapsed_seconds": 0.1,
                "total_docs": len(DOC_IDS),
                "inverted_vocab_post_cap": len(inv.inverted_index),
                "tfidf_vocab": len(tfidf.vectorizer.vocabulary_),
                "tfidf_nnz": int(tfidf.matrix.nnz),
                "bm25_vocab": len(bm25.vocab),
                "min_df": 1,
                "max_df_ratio": 1.0,
                "bm25_method": "lucene",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    # Redirect the service's path resolver to our tmp_path.
    monkeypatch.setattr(config_mod, "INDEX_ROOT", tmp_path / "indexes")
    monkeypatch.setattr(service_mod, "_is_known", lambda ds: ds == TEST_DATASET_ID)

    # Reset the LRU caches so a test that runs after another doesn't
    # see a stale entry.
    service_mod._TFIDF_CACHE.clear()
    service_mod._BM25_CACHE.clear()
    service_mod._INVIDX_CACHE.clear()

    return tmp_path


@pytest.fixture
def client():
    """A FastAPI TestClient that wraps the indexing service."""
    from fastapi.testclient import TestClient

    return TestClient(service_mod.app)
