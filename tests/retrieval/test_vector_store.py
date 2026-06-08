"""Tests for the FAISS-backed :class:`DenseIndex`."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from services.retrieval.app.vector_store import (
    DOC_IDS_FILENAME,
    EMBEDDINGS_FILENAME,
    INDEX_FILENAME,
    DenseIndex,
)

# ─────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────


@pytest.fixture
def random_vectors() -> tuple[np.ndarray, list[str]]:
    """Deterministic random float32 vectors in a 32-D space."""
    rng = np.random.default_rng(seed=42)
    n, dim = 50, 32
    vecs = rng.standard_normal((n, dim)).astype(np.float32)
    # L2-normalise rows so we can talk about cosine similarity.
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    ids = [f"doc_{i:04d}" for i in range(n)]
    return vecs, ids


@pytest.fixture
def built_index(random_vectors: tuple[np.ndarray, list[str]]) -> DenseIndex:
    vecs, ids = random_vectors
    idx = DenseIndex()
    idx.add(vecs, ids)
    return idx


# ─────────────────────────────────────────────────────────────────────────
# Build
# ─────────────────────────────────────────────────────────────────────────


def test_add_stores_vectors_and_ids(
    random_vectors: tuple[np.ndarray, list[str]],
) -> None:
    vecs, ids = random_vectors
    idx = DenseIndex()
    idx.add(vecs, ids)
    assert idx.size() == len(ids)
    assert idx.dim() == vecs.shape[1]
    assert idx.doc_ids == ids
    assert idx.vectors is not None and idx.vectors.shape == vecs.shape


def test_add_rejects_1d_vectors() -> None:
    idx = DenseIndex()
    with pytest.raises(ValueError, match="must be 2-D"):
        idx.add(np.zeros(8, dtype=np.float32), ["a"] * 8)  # type: ignore[arg-type]


def test_add_rejects_mismatched_ids(random_vectors: tuple[np.ndarray, list[str]]) -> None:
    vecs, _ = random_vectors
    idx = DenseIndex()
    with pytest.raises(ValueError, match="len\\(doc_ids\\)"):
        idx.add(vecs, ["only_one"])


def test_add_rejects_nan() -> None:
    vecs = np.array([[1.0, 2.0, 3.0], [np.nan, 0.0, 0.0]], dtype=np.float32)
    idx = DenseIndex()
    with pytest.raises(ValueError, match="NaN or Inf"):
        idx.add(vecs, ["a", "b"])


def test_add_casts_to_float32(random_vectors: tuple[np.ndarray, list[str]]) -> None:
    vecs, ids = random_vectors
    vecs_f64 = vecs.astype(np.float64)
    idx = DenseIndex()
    idx.add(vecs_f64, ids)
    assert idx.vectors is not None
    assert idx.vectors.dtype == np.float32


# ─────────────────────────────────────────────────────────────────────────
# Search
# ─────────────────────────────────────────────────────────────────────────


def test_search_returns_top_k(built_index: DenseIndex) -> None:
    idx = built_index
    # Use the first inserted vector as the query; it should be its
    # own nearest neighbour (cosine = 1.0).
    q = idx.vectors[0].copy()  # type: ignore[index]
    scores, ids = idx.search(q, k=5)
    assert scores.shape == (5,)
    assert ids.shape == (5,)
    # Top-1 is the query itself.
    assert int(ids[0]) == 0
    assert scores[0] == pytest.approx(1.0, abs=1e-4)
    # Scores are descending.
    assert (scores[1:] <= scores[:-1] + 1e-5).all()


def test_search_clamps_k(built_index: DenseIndex) -> None:
    idx = built_index
    q = idx.vectors[0].copy()  # type: ignore[index]
    # k larger than corpus returns the whole corpus.
    scores, ids = idx.search(q, k=10_000)
    assert scores.shape == (idx.size(),)


def test_search_1d_query(built_index: DenseIndex) -> None:
    idx = built_index
    q = idx.vectors[0].copy()  # type: ignore[index]
    # 1-D query should work (we reshape internally).
    scores, ids = idx.search(q, k=3)
    assert scores.shape == (3,)


def test_search_before_add_raises() -> None:
    idx = DenseIndex()
    with pytest.raises(RuntimeError, match="add\\(\\)"):
        idx.search(np.zeros(8, dtype=np.float32), k=3)


# ─────────────────────────────────────────────────────────────────────────
# Persistence
# ─────────────────────────────────────────────────────────────────────────


def test_save_and_load_roundtrip(
    built_index: DenseIndex,
    tmp_path: Path,
) -> None:
    built_index.save(tmp_path)
    # All three files exist.
    assert (tmp_path / INDEX_FILENAME).exists()
    assert (tmp_path / EMBEDDINGS_FILENAME).exists()
    assert (tmp_path / DOC_IDS_FILENAME).exists()
    # Load.
    loaded = DenseIndex.load(tmp_path)
    assert loaded.size() == built_index.size()
    assert loaded.dim() == built_index.dim()
    assert loaded.doc_ids == built_index.doc_ids
    # Search on the loaded index matches the original.
    q = built_index.vectors[0].copy()  # type: ignore[index]
    s_orig, ids_orig = built_index.search(q, k=5)
    s_loaded, ids_loaded = loaded.search(q, k=5)
    np.testing.assert_allclose(s_orig, s_loaded, atol=1e-5)
    np.testing.assert_array_equal(ids_orig, ids_loaded)


def test_load_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        DenseIndex.load(tmp_path)


def test_save_before_add_raises(tmp_path: Path) -> None:
    idx = DenseIndex()
    with pytest.raises(RuntimeError, match="add\\(\\)"):
        idx.save(tmp_path)


def test_stats_shape(built_index: DenseIndex) -> None:
    s = built_index.stats()
    assert s["num_vectors"] == built_index.size()
    assert s["dim"] == built_index.dim()
    assert s["index_type"] == "IndexFlatIP"


# ─────────────────────────────────────────────────────────────────────────
# Filename constants
# ─────────────────────────────────────────────────────────────────────────


def test_filenames_are_strings() -> None:
    assert isinstance(INDEX_FILENAME, str)
    assert isinstance(EMBEDDINGS_FILENAME, str)
    assert isinstance(DOC_IDS_FILENAME, str)


def test_load_uses_jsonl_safe_ids(tmp_path: Path) -> None:
    """doc_ids.json is read as JSON, not JSONL, so the array must be a list."""

    rng = np.random.default_rng(seed=1)
    vecs = rng.standard_normal((3, 4)).astype(np.float32)
    ids = ["a", "b", "c"]
    idx = DenseIndex()
    idx.add(vecs, ids)
    idx.save(tmp_path)
    # The on-disk file is a JSON array (not newline-delimited).
    raw = (tmp_path / DOC_IDS_FILENAME).read_text(encoding="utf-8")
    parsed = json.loads(raw)
    assert parsed == ids


# ─────────────────────────────────────────────────────────────────────────
# IndexIVFFlat
# ─────────────────────────────────────────────────────────────────────────


@patch("services.retrieval.app.vector_store.FAISS_INDEX_TYPE", "IndexIVFFlat")
@patch("services.retrieval.app.vector_store.FAISS_IVF_NLIST", 4)
def test_ivf_build_and_search(random_vectors: tuple[np.ndarray, list[str]]) -> None:
    """IndexIVFFlat should build without error and return reasonable results."""
    vecs, ids = random_vectors
    idx = DenseIndex()
    idx.add(vecs, ids, nlist=4)
    assert idx.size() == len(ids)
    assert idx.dim() == vecs.shape[1]
    assert idx.doc_ids == ids
    # Search — top-1 should be the query itself.
    q = vecs[0].copy()
    scores, idx_ids = idx.search(q, k=5)
    assert scores.shape == (5,)
    assert idx_ids.shape == (5,)
    # With 50 vectors and nlist=4, IVF should still find the exact
    # top-1 (the query itself) in most setups.
    top1_doc = int(idx_ids[0])
    assert scores[0] == pytest.approx(1.0, abs=0.1) or top1_doc == 0
    # Scores are descending.
    assert (scores[1:] <= scores[:-1] + 1e-5).all()
