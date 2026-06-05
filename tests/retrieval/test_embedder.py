"""Tests for the :class:`Embedder` wrapper around sentence-transformers.

We don't load the real 90 MB model in CI. The :class:`_FakeEmbedder`
in ``conftest.py`` stands in for the real class via a monkeypatch on
the service. The tests here exercise the real wrapper's public API
on the fake (which has a matching shape) plus the validation paths
that don't need a model at all.
"""

from __future__ import annotations

import numpy as np
import pytest

from services.retrieval.app import embedder as embedder_mod
from services.retrieval.app.config import (
    DEFAULT_MODEL_NAME,
    EMBED_DEVICE,
    MODEL_CACHE_SIZE,
    USE_FP16,
    model_cache_dir,
)
from services.retrieval.app.embedder import Embedder

# ─────────────────────────────────────────────────────────────────────────
# Filename / config helpers
# ─────────────────────────────────────────────────────────────────────────


def test_default_model_name_is_minilm() -> None:
    assert DEFAULT_MODEL_NAME == "sentence-transformers/all-MiniLM-L6-v2"


def test_model_cache_dir_handles_slash() -> None:
    p = model_cache_dir("sentence-transformers/all-MiniLM-L6-v2")
    assert "sentence-transformers__all-MiniLM-L6-v2" in str(p)
    assert "/" not in p.name


def test_embed_device_is_valid() -> None:
    """EMBED_DEVICE is auto-detected to either cuda or cpu."""
    assert EMBED_DEVICE in ("cpu", "cuda")


def test_use_fp16_only_on_cuda() -> None:
    """USE_FP16 is True only when the device is cuda (no fp16 on CPU)."""
    if EMBED_DEVICE == "cuda":
        assert USE_FP16 is True
    else:
        assert USE_FP16 is False


# ─────────────────────────────────────────────────────────────────────────
# Construction
# ─────────────────────────────────────────────────────────────────────────


def test_embedder_default_construction() -> None:
    e = Embedder()
    assert e.default_model_name == DEFAULT_MODEL_NAME
    assert e.batch_size > 0
    assert e.max_seq_length > 0
    assert e.device in ("cpu", "cuda")
    # ``use_fp16`` is a public, read-only slot.
    assert isinstance(e.use_fp16, bool)
    # On CPU, use_fp16 is forced off even if a default-True is passed.
    if e.device == "cpu":
        assert e.use_fp16 is False
    assert e.loaded_models() == []


def test_embedder_custom_construction() -> None:
    e = Embedder(
        default_model_name="x/y",
        batch_size=64,
        max_seq_length=128,
        device="cpu",
    )
    assert e.default_model_name == "x/y"
    assert e.batch_size == 64
    assert e.max_seq_length == 128
    # Forcing device="cpu" disables fp16 regardless of USE_FP16 default.
    assert e.use_fp16 is False


def test_embedder_use_fp16_force_off_on_cpu() -> None:
    """Even with use_fp16=True, the device=cpu combo zeros it out."""
    e = Embedder(device="cpu", use_fp16=True)
    assert e.use_fp16 is False


def test_embedder_use_fp16_force_on_cuda() -> None:
    e = Embedder(device="cuda", use_fp16=True)
    assert e.use_fp16 is True


def test_embedder_use_fp16_force_off_cuda() -> None:
    e = Embedder(device="cuda", use_fp16=False)
    assert e.use_fp16 is False


# ─────────────────────────────────────────────────────────────────────────
# _load: cache + LRU
# ─────────────────────────────────────────────────────────────────────────


def test_load_caches_on_second_call() -> None:
    """Re-loading the same model returns the cached instance."""

    class _StubST:
        def __init__(self) -> None:
            self.max_seq_length = 0

        def get_sentence_embedding_dimension(self) -> int:
            return 4

    class _StubEmbedder(Embedder):
        # Bypass sentence_transformers.SentenceTransformer so we don't
        # load torch / the real model.
        def _load(self, model_name: str) -> tuple[_StubST, int]:  # type: ignore[override]
            if model_name in self._cache:
                self._cache.move_to_end(model_name)
                return self._cache[model_name]
            while len(self._cache) >= MODEL_CACHE_SIZE:
                self._cache.popitem(last=False)
            st = _StubST()
            dim = 4
            self._cache[model_name] = (st, dim)
            return st, dim

    e = _StubEmbedder()
    a, _ = e._load("m1")
    b, _ = e._load("m1")
    assert a is b
    assert e.loaded_models() == ["m1"]


def test_load_lru_eviction() -> None:
    class _StubEmbedder(Embedder):
        def _load(self, model_name: str) -> tuple[object, int]:  # type: ignore[override]
            if model_name in self._cache:
                self._cache.move_to_end(model_name)
                return self._cache[model_name]
            while len(self._cache) >= MODEL_CACHE_SIZE:
                self._cache.popitem(last=False)
            self._cache[model_name] = (object(), 4)
            return self._cache[model_name]

    e = _StubEmbedder()
    # Cache size is MODEL_CACHE_SIZE (set to 2 for Phase 5 multi-encoder
    # support: L6 + L12 must both stay resident). Loading two models
    # should keep BOTH resident, with m2 being most recently used.
    e._load("m1")
    e._load("m2")
    assert e.loaded_models() == ["m1", "m2"]
    # A third model evicts the least-recently-used (m1).
    e._load("m3")
    assert e.loaded_models() == ["m2", "m3"]


# ─────────────────────────────────────────────────────────────────────────
# time_block
# ─────────────────────────────────────────────────────────────────────────


def test_time_block_prints_elapsed(capsys: pytest.CaptureFixture[str]) -> None:
    with embedder_mod.time_block("phase-3-test"):
        _ = sum(range(1000))
    captured = capsys.readouterr()
    assert "phase-3-test" in captured.out
    assert "s" in captured.out


# ─────────────────────────────────────────────────────────────────────────
# Validation (no model load needed)
# ─────────────────────────────────────────────────────────────────────────


def test_encode_documents_empty_list_raises() -> None:
    e = Embedder()
    with pytest.raises(ValueError, match="non-empty"):
        e.encode_documents([])


def test_encode_query_empty_string_raises() -> None:
    e = Embedder()
    with pytest.raises(ValueError, match="non-empty"):
        e.encode_query("")


# ─────────────────────────────────────────────────────────────────────────
# NaN / Inf guard in encode
# ─────────────────────────────────────────────────────────────────────────


def test_encode_documents_replaces_nan_rows() -> None:
    """If the underlying model returns NaN, the wrapper zeros the row."""

    class _NaNModel:
        max_seq_length = 8

        def get_sentence_embedding_dimension(self) -> int:
            return 3

        def encode(
            self,
            texts: list[str],
            batch_size: int = 32,
            show_progress_bar: bool = False,
            convert_to_numpy: bool = True,
            normalize_embeddings: bool = True,
        ) -> np.ndarray:
            arr = np.ones((len(texts), 3), dtype=np.float32)
            arr[0] = np.nan
            return arr

    class _StubEmbedder(Embedder):
        def _load(self, model_name: str) -> tuple[_NaNModel, int]:  # type: ignore[override]
            self._cache[model_name] = (_NaNModel(), 3)
            return self._cache[model_name]

    e = _StubEmbedder()
    e._load("m")
    out = e.encode_documents(["good", "also good"], model_name="m", show_progress=False)
    assert out.shape == (2, 3)
    assert np.isfinite(out).all()
    # The NaN row was replaced by zeros.
    np.testing.assert_array_equal(out[0], np.zeros(3, dtype=np.float32))
