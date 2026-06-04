"""Sentence-transformer embedder used by the dense-retrieval service.

Wraps :class:`sentence_transformers.SentenceTransformer` with three
policies:

  1. **Lazy load.** The model is only loaded on the first call to
     :meth:`Embedder.encode_documents` or :meth:`Embedder.encode_query`.
     Loading takes 2-3 seconds and holds ~400 MB of RAM; we don't pay
     that cost at uvicorn startup.

  2. **LRU-1 cache.** Only one model is held in memory at a time. If a
     request comes in with a different model name, the old one is
     evicted before the new one is loaded. Configurable via
     ``MODEL_CACHE_SIZE`` (kept at 1 in Phase 3 because the guide ships
     a single model).

  3. **Local cache first.** Models are loaded from
     ``data/models/{safe_name}/`` when present (populated by
     ``make download-models``). If the local cache is missing, the
     Hub is hit; the downloaded model is then cached on disk for next
     time.

The embedder feeds the model **raw text** (not the Phase 1 preprocessed
tokens). The WordPiece BPE tokenizer in the encoder expects natural
language; feeding it Porter-stemmed lowercase alphanumeric strings
would silently destroy quality.
"""

from __future__ import annotations

import os
import sys
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any

# Force UTF-8 on Windows before any logging/output.
try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
except Exception:
    pass

import numpy as np

# Force UTF-8 on Windows before any logging/output.
from services.retrieval.app.config import (  # noqa: E402
    DEFAULT_BATCH_SIZE,
    DEFAULT_BATCH_SIZE_GPU,
    DEFAULT_MODEL_NAME,
    EMBED_DEVICE,
    MAX_SEQ_LENGTH,
    MODEL_CACHE_SIZE,
    USE_FP16,
    model_cache_dir,
)

# Set torch's intra-/inter-op thread counts before we import torch
# anywhere else. On Windows the default is often 1, which makes CPU
# encoding ~10x slower than it needs to be. We default to 6+6
# (intra/inter) which gave the best throughput on the 12-core test
# machine. Override with ``IR_TORCH_THREADS`` if needed.
if "IR_TORCH_THREADS" in os.environ:
    _n = int(os.environ["IR_TORCH_THREADS"])
    os.environ.setdefault("OMP_NUM_THREADS", str(_n))
    os.environ.setdefault("MKL_NUM_THREADS", str(_n))
os.environ.setdefault("OMP_NUM_THREADS", "6")
os.environ.setdefault("MKL_NUM_THREADS", "6")


class Embedder:
    """Lazy, LRU-cached wrapper around :class:`SentenceTransformer`.

    Use as a context manager or directly::

        e = Embedder()
        e.warm_up()                          # optional, for timing
        matrix = e.encode_documents(texts)  # (N, 384) float32, L2-normalised
        vec = e.encode_query(query)         # (384,) float32, L2-normalised
    """

    __slots__ = (
        "_cache",
        "default_model_name",
        "batch_size",
        "max_seq_length",
        "device",
        "use_fp16",
    )

    def __init__(
        self,
        default_model_name: str = DEFAULT_MODEL_NAME,
        batch_size: int | None = None,
        max_seq_length: int = MAX_SEQ_LENGTH,
        device: str = EMBED_DEVICE,
        use_fp16: bool = USE_FP16,
    ) -> None:
        self.default_model_name: str = default_model_name
        # Default to the GPU-friendly batch size when CUDA is available,
        # else the CPU-friendly size. Callers can still override.
        if batch_size is None:
            batch_size = DEFAULT_BATCH_SIZE_GPU if EMBED_DEVICE == "cuda" else DEFAULT_BATCH_SIZE
        self.batch_size: int = batch_size
        self.max_seq_length: int = max_seq_length
        self.device: str = device
        self.use_fp16: bool = bool(use_fp16 and device == "cuda")
        # LRU cache: model_name -> (SentenceTransformer, dim)
        self._cache: OrderedDict[str, tuple[Any, int]] = OrderedDict()

    # ─────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────

    def warm_up(self, model_name: str | None = None) -> int:
        """Load ``model_name`` (or the default) into the LRU cache.

        Returns the embedding dimension.
        """
        name = model_name or self.default_model_name
        self._load(name)
        _, dim = self._cache[name]
        return dim

    def encode_documents(
        self,
        texts: list[str],
        model_name: str | None = None,
        batch_size: int | None = None,
        show_progress: bool = True,
    ) -> np.ndarray:
        """Encode a batch of documents. Returns ``(N, dim)`` float32.

        The output is **L2-normalised** so that ``IndexFlatIP`` gives
        cosine similarity. NaN entries (from empty strings) are replaced
        with zero vectors so the FAISS index never has NaN.
        """
        if not texts:
            raise ValueError("encode_documents requires a non-empty list of texts.")
        name = model_name or self.default_model_name
        st, _ = self._load(name)
        bs = batch_size or self.batch_size
        # ``convert_to_numpy=True`` (default) returns ndarray directly.
        # ``normalize_embeddings=True`` L2-normalises each row.
        vectors: np.ndarray = st.encode(  # type: ignore[assignment]
            texts,
            batch_size=bs,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        # Defensive: replace any NaN rows (empty/garbage input) with zeros
        # so the index never contains NaN.
        if vectors.dtype != np.float32:
            vectors = vectors.astype(np.float32, copy=False)
        if not np.isfinite(vectors).all():
            bad = ~np.isfinite(vectors).all(axis=1)
            vectors[bad] = 0.0
        return vectors

    def encode_query(self, text: str, model_name: str | None = None) -> np.ndarray:
        """Encode a single query. Returns ``(dim,)`` float32, L2-normalised."""
        if not text:
            raise ValueError("encode_query requires a non-empty string.")
        name = model_name or self.default_model_name
        st, _ = self._load(name)
        vec: np.ndarray = st.encode(  # type: ignore[assignment]
            [text],
            batch_size=1,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        if vec.dtype != np.float32:
            vec = vec.astype(np.float32, copy=False)
        if not np.isfinite(vec).all():
            vec = np.zeros_like(vec)
        # Squeeze to 1-D for cleaner transport.
        return vec.reshape(-1)

    def embedding_dim(self, model_name: str | None = None) -> int:
        """Return the embedding dimension (loads the model if needed)."""
        name = model_name or self.default_model_name
        _, dim = self._load(name)
        return dim

    def loaded_models(self) -> list[str]:
        """Return the model names currently in the LRU cache."""
        return list(self._cache.keys())

    # ─────────────────────────────────────────────────────────────────────
    # Internals
    # ─────────────────────────────────────────────────────────────────────

    def _load(self, model_name: str) -> tuple[Any, int]:
        """Return ``(SentenceTransformer, dim)`` for ``model_name``."""
        cached = self._cache.get(model_name)
        if cached is not None:
            self._cache.move_to_end(model_name)
            return cached
        # Evict cold models if we're at capacity.
        while len(self._cache) >= MODEL_CACHE_SIZE:
            self._cache.popitem(last=False)
        # Lazy import: sentence_transformers pulls torch (~800 MB).
        import torch  # local; we want to set threads after torch is loaded
        from sentence_transformers import SentenceTransformer

        # Pin torch's thread counts so CPU encoding uses all cores. The
        # env vars above take effect at torch import time, but we also
        # call set_num_threads explicitly so users who already had
        # torch loaded still get the right config.
        # ``set_num_interop_threads`` MUST be called before any parallel
        # work has started; on the second ``Embedder`` in a test it will
        # raise ``RuntimeError``. We guard with try/except below so
        # multi-Embedder scenarios (e.g. a smoke test instantiating
        # several) don't crash on the second instance. The first call
        # still wins.
        torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "6")))
        try:
            current = torch.get_num_interop_threads()
        except Exception:
            current = 0
        if current != int(os.environ.get("MKL_NUM_THREADS", "6")):
            try:
                torch.set_num_interop_threads(int(os.environ.get("MKL_NUM_THREADS", "6")))
            except (RuntimeError, AttributeError):
                pass

        local = model_cache_dir(model_name)
        cache_folder = str(local) if local.exists() else None
        st = SentenceTransformer(
            model_name,
            device=self.device,
            cache_folder=cache_folder,
        )
        st.max_seq_length = self.max_seq_length
        # ``get_sentence_embedding_dimension`` was renamed to
        # ``get_embedding_dimension`` in sentence-transformers 5.x; the
        # old name is deprecated but still works. Use the new name with
        # a fallback for the few older 4.x versions in the wild.
        get_dim = getattr(st, "get_embedding_dimension", None) or getattr(
            st, "get_sentence_embedding_dimension"
        )
        dim = int(get_dim())
        # Cast the underlying transformer to half precision on GPU. This
        # roughly doubles throughput on Turing+ with < 1% recall drop
        # for MiniLM-L6-v2. The encoder's output is still cast to
        # float32 on the way out (sentence-transformers handles that),
        # so downstream FAISS / NumPy code is unaffected.
        if self.use_fp16:
            # ``st[0]`` is the bare ``Transformer`` module; ``.auto_model``
            # is the inner ``AutoModel`` whose parameters we cast.
            try:
                st[0].auto_model = st[0].auto_model.half()  # type: ignore[index]
            except Exception:
                # If the model architecture doesn't support half, just
                # leave it in fp32. The encode path will still work.
                pass
        self._cache[model_name] = (st, dim)
        return st, dim


def time_block(label: str) -> _TimeBlock:
    """Context manager for ``with time_block('encode'):`` blocks.

    Prints ``label: 12.3s`` to stdout on exit. Useful in the build
    script for per-step timing. Avoids pulling in a logger just for
    this.
    """
    return _TimeBlock(label)


class _TimeBlock:
    __slots__ = ("label", "t0")

    def __init__(self, label: str) -> None:
        self.label = label
        self.t0 = 0.0

    def __enter__(self) -> _TimeBlock:
        self.t0 = time.time()
        return self

    def __exit__(self, *exc: object) -> None:
        elapsed = time.time() - self.t0
        print(f"[{self.label}] {elapsed:.1f}s", flush=True)


def model_dir_for(model_name: str) -> Path:
    """Public helper to expose the local cache path for a model."""
    return model_cache_dir(model_name)
