"""Test fixtures for the dense-retrieval service.

Two fixtures do the heavy lifting:

  * ``small_corpus`` -- a deterministic 5-doc corpus, used to build
    a real ``DenseIndex`` in memory.
  * ``client`` -- a FastAPI ``TestClient`` with the embedder
    **dependency-overridden** so tests don't load the real
    sentence-transformer model (which would take 20+ seconds and
    ~400 MB of RAM per test).
"""

from __future__ import annotations

import sys
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

# Allow `pytest` from the repo root to import the service package.
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.retrieval.app import service as service_mod  # noqa: E402

SMALL_CORPUS: list[tuple[str, str]] = [
    ("d1", "the quick brown fox jumps over the lazy dog"),
    ("d2", "a stitch in time saves nine"),
    ("d3", "the early bird catches the worm"),
    ("d4", "all that glitters is not gold"),
    ("d5", "fox and dog are best friends"),
]


# ─────────────────────────────────────────────────────────────────────────
# Deterministic fake embedder
# ─────────────────────────────────────────────────────────────────────────


class _FakeEmbedder:
    """A deterministic drop-in for :class:`Embedder` in tests.

    Maps each token to a fixed pseudo-random vector and averages them
    (L2-normalised) so the relative order of documents is stable and
    meaningful. With this scheme:

      * d1 ("the quick brown fox ...") and d5 ("fox and dog ...") are
        the most similar (both heavy on "fox").
      * d4 ("all that glitters ...") is the outlier.

    The dimension is 16 to keep tests fast and the printed diffs short.
    """

    __slots__ = (
        "default_model_name",
        "batch_size",
        "max_seq_length",
        "device",
        "use_fp16",
        "dim",
        "_rng",
        "_vocab",
    )

    def __init__(self, dim: int = 16) -> None:
        self.default_model_name = "fake-model"
        self.batch_size = 32
        self.max_seq_length = 128
        self.device = "cpu"
        self.use_fp16 = False
        self.dim = dim
        self._rng = np.random.default_rng(seed=0xDEADBEEF)
        # Build a fixed vocabulary from the small corpus.
        vocab: dict[str, np.ndarray] = {}
        for _, text in SMALL_CORPUS:
            for tok in text.lower().split():
                if tok not in vocab:
                    vocab[tok] = self._rng.standard_normal(dim).astype(np.float32)
        self._vocab = vocab

    def warm_up(self, model_name: str | None = None) -> int:
        return self.dim

    def embedding_dim(self, model_name: str | None = None) -> int:
        return self.dim

    def loaded_models(self) -> list[str]:
        return ["fake-model"]

    def encode_documents(
        self,
        texts: list[str],
        model_name: str | None = None,
        batch_size: int | None = None,
        show_progress: bool = True,
    ) -> np.ndarray:
        out = np.stack([self._embed(t) for t in texts])
        return out

    def encode_query(self, text: str, model_name: str | None = None) -> np.ndarray:
        return self._embed(text)

    # ─────────────────────────────────────────────────────────────────
    # Internals
    # ─────────────────────────────────────────────────────────────────

    def _embed(self, text: str) -> np.ndarray:
        toks = text.lower().split()
        if not toks:
            return np.zeros(self.dim, dtype=np.float32)
        vecs = np.stack([self._vocab.get(t, np.zeros(self.dim, dtype=np.float32)) for t in toks])
        m = vecs.mean(axis=0)
        n = float(np.linalg.norm(m))
        if n == 0.0:
            return m.astype(np.float32)
        return (m / n).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────


@pytest.fixture
def fake_embedder() -> _FakeEmbedder:
    return _FakeEmbedder(dim=16)


@pytest.fixture
def small_corpus() -> list[tuple[str, str]]:
    return list(SMALL_CORPUS)


@pytest.fixture
def client(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fake_embedder: _FakeEmbedder,
) -> Iterator[TestClient]:
    """A TestClient with the real dataset dir redirected to ``tmp_path``."""
    # Patch the paths used by the service so any in-process /load or
    # /stats call looks under tmp_path.
    monkeypatch.setattr(service_mod, "index_dir", lambda ds: tmp_path / ds)
    # Build a real FAISS index under tmp_path with the fake embedder.
    from services.retrieval.app import vector_store as vector_store_mod

    ds = "touche2020"
    d = tmp_path / ds
    d.mkdir(parents=True, exist_ok=True)
    doc_ids = [doc_id for doc_id, _ in SMALL_CORPUS]
    texts = [text for _, text in SMALL_CORPUS]
    vectors = fake_embedder.encode_documents(texts)
    idx = vector_store_mod.DenseIndex()
    idx.add(vectors, doc_ids)
    idx.save(d)
    # Write a build_meta.json so /stats works.
    import json as _json

    (d / "build_meta.json").write_text(
        _json.dumps(
            {
                "dataset_id": ds,
                "built_at": "2026-06-04T00:00:00",
                "status": "ok",
                "model_name": "fake-model",
                "index_type": "IndexFlatIP",
                "num_vectors": len(doc_ids),
                "embedding_dim": fake_embedder.dim,
                "elapsed_seconds": 0.0,
                "size_mb": 0.0,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    # Inject the fake embedder into the service so /search and /embed
    # don't try to load the real model.
    import services.retrieval.app.embedder as embedder_mod

    monkeypatch.setattr(service_mod, "_EMBEDDER", fake_embedder, raising=False)
    monkeypatch.setattr(embedder_mod, "Embedder", lambda *a, **k: fake_embedder)
    # Reset the LRU cache so each test starts fresh.
    service_mod._FAISS_CACHE.clear()
    service_mod._LOADED_DATASET = None
    service_mod._LOADED_MODEL_NAME = ""
    with TestClient(service_mod.app) as c:
        yield c
    service_mod._FAISS_CACHE.clear()
