"""FAISS-backed dense vector store.

One :class:`DenseIndex` per dataset. Wraps a ``faiss.IndexFlatIP`` with:

  * ``add(vectors, ids)`` -- bulk insert
  * ``search(query_vec, k)`` -- top-k by inner product (= cosine
    similarity, because vectors are L2-normalised by the embedder)
  * ``save(dir)`` / ``load(dir)`` -- ``faiss.write_index`` + a side
    ``doc_ids.json`` (FAISS only stores integer ids)

Why IndexFlatIP? Both corpora are < 1M vectors and we are CPU-bound
on encode, not search. Flat gives reproducible scores (vital for
Phase 9 evaluation) and is the simplest thing that works. The guide's
``IndexIVFFlat`` with ``nprobe=16`` is the next step up; it requires
a training pass and would make the on-disk index format more
elaborate.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING

# Force UTF-8 on Windows before any logging/output.
try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
except Exception:
    pass

import numpy as np

# Force UTF-8 on Windows before any logging/output.
from services.retrieval.app.config import FAISS_INDEX_TYPE  # noqa: E402

if TYPE_CHECKING:
    import faiss  # noqa: F401  (only imported for type hints)

# ─────────────────────────────────────────────────────────────────────────
# Filenames (public so the build script and the service can both use them)
# ─────────────────────────────────────────────────────────────────────────

INDEX_FILENAME: str = "faiss.index"
EMBEDDINGS_FILENAME: str = "embeddings.npy"
DOC_IDS_FILENAME: str = "doc_ids.json"


class DenseIndex:
    """A FAISS-backed index over a fixed corpus.

    Vectors are added once at build time and never mutated. Searches
    are O(N * dim) on IndexFlatIP, but NumPy+SIMD make it fast enough
    for 1M-vec corpora.
    """

    __slots__ = (
        "doc_ids",
        "vectors",
        "_index",
    )

    def __init__(self) -> None:
        self.doc_ids: list[str] = []
        # Keep a copy of the vectors in memory for save() and to make
        # test assertions easier. The FAISS index also holds the data,
        # so this is technically duplicated -- but the alternative
        # (re-loading from .npy at save time) costs an extra file read.
        self.vectors: np.ndarray | None = None
        # ``faiss.Index`` is imported lazily inside each method (heavy
        # native module). At runtime the slot holds a real Index; the
        # type alias is only for mypy.
        self._index: faiss.Index | None = None  # type: ignore[name-defined]

    # ─────────────────────────────────────────────────────────────────────
    # Build
    # ─────────────────────────────────────────────────────────────────────

    def add(self, vectors: np.ndarray, doc_ids: list[str]) -> None:
        """Build the FAISS index from ``vectors`` (shape ``(N, dim)``,
        float32, L2-normalised) and the matching ``doc_ids``.
        """
        if vectors.ndim != 2:
            raise ValueError(f"vectors must be 2-D (N, dim); got shape {vectors.shape!r}")
        if vectors.dtype != np.float32:
            vectors = vectors.astype(np.float32, copy=False)
        if len(doc_ids) != vectors.shape[0]:
            raise ValueError(
                f"len(doc_ids) ({len(doc_ids)}) != vectors.shape[0] " f"({vectors.shape[0]})"
            )
        # Defensive: NaN/Inf guard before faiss sees the data.
        if not np.isfinite(vectors).all():
            raise ValueError("vectors contain NaN or Inf")
        import faiss  # local import; native module

        if FAISS_INDEX_TYPE != "IndexFlatIP":
            raise NotImplementedError(
                f"Only IndexFlatIP is wired up; got {FAISS_INDEX_TYPE!r}. "
                "Add the new index type to DenseIndex.add()."
            )
        dim = int(vectors.shape[1])
        idx = faiss.IndexFlatIP(dim)
        # faiss wants a contiguous float32 array; copies if needed.
        idx.add(np.ascontiguousarray(vectors))
        self.vectors = vectors
        self.doc_ids = list(doc_ids)
        self._index = idx

    # ─────────────────────────────────────────────────────────────────────
    # Search
    # ─────────────────────────────────────────────────────────────────────

    def search(self, query_vec: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(scores, ids)`` for top-``k`` matches.

        ``scores`` are inner products (== cosine sims because the
        vectors are L2-normalised); ``ids`` are integer positions
        into ``self.doc_ids``.

        ``k`` is clamped to ``self.size()`` so callers can pass
        ``k=N`` without guarding.
        """
        if self._index is None:
            raise RuntimeError("DenseIndex.add() must be called before search().")
        if query_vec.ndim == 1:
            query_vec = query_vec.reshape(1, -1)
        if query_vec.dtype != np.float32:
            query_vec = query_vec.astype(np.float32, copy=False)
        n = self.size()
        k = max(1, min(int(k), n))

        scores, ids = self._index.search(np.ascontiguousarray(query_vec), k)  # type: ignore[attr-defined]
        # FAISS returns shape (1, k) arrays; flatten for the common
        # single-query case.
        return scores.reshape(-1), ids.reshape(-1)

    # ─────────────────────────────────────────────────────────────────────
    # Introspection
    # ─────────────────────────────────────────────────────────────────────

    def size(self) -> int:
        if self._index is None:
            return 0
        return int(self._index.ntotal)  # type: ignore[attr-defined]

    def dim(self) -> int:
        if self.vectors is None:
            return 0
        return int(self.vectors.shape[1])

    def stats(self) -> dict[str, int | float | str]:
        """Return a JSON-serialisable summary for the ``/stats`` endpoint."""
        return {
            "num_vectors": self.size(),
            "dim": self.dim(),
            "index_type": FAISS_INDEX_TYPE,
        }

    # ─────────────────────────────────────────────────────────────────────
    # Persistence
    # ─────────────────────────────────────────────────────────────────────

    def save(
        self,
        dirpath: str | Path,
        index_filename: str = INDEX_FILENAME,
        embeddings_filename: str = EMBEDDINGS_FILENAME,
    ) -> None:
        """Write three files to ``dirpath``:

        * ``faiss.index`` (or ``index_filename``) -- ``faiss.write_index`` of the FAISS index
        * ``embeddings.npy`` (or ``embeddings_filename``) -- the raw vectors
        * ``doc_ids.json`` -- position -> doc_id mapping (always the
          default filename; shared across indexes in the same dir)

        The ``index_filename`` / ``embeddings_filename`` overrides are
        used by the Phase 5 build script to write the L12 encoder's
        index as ``faiss_l12.index`` + ``embeddings_l12.npy``.
        """
        if self._index is None or self.vectors is None:
            raise RuntimeError("DenseIndex.add() must be called before save().")
        d = Path(dirpath)
        d.mkdir(parents=True, exist_ok=True)
        import faiss  # local import

        faiss.write_index(self._index, str(d / index_filename))  # type: ignore[attr-defined]
        # .npy is fastest when the file is a single contiguous array.
        np.save(d / embeddings_filename, self.vectors, allow_pickle=False)
        (d / DOC_IDS_FILENAME).write_text(
            json.dumps(self.doc_ids, ensure_ascii=False), encoding="utf-8"
        )

    @classmethod
    def load(
        cls,
        dirpath: str | Path,
        index_filename: str = INDEX_FILENAME,
        embeddings_filename: str = EMBEDDINGS_FILENAME,
    ) -> DenseIndex:
        """Load a previously-saved :class:`DenseIndex` from ``dirpath``.

        ``index_filename`` and ``embeddings_filename`` default to the
        Phase 3 filenames (``faiss.index`` + ``embeddings.npy``). The
        Phase 5 multi-encoder bonus calls this with
        ``index_filename="faiss_l12.index"`` and
        ``embeddings_filename="embeddings_l12.npy"`` to load the L12
        index. ``doc_ids.json`` is shared between the two indexes
        (same corpus, same row order) and is always read from the
        default path.
        """
        d = Path(dirpath)
        for name in (index_filename, embeddings_filename, DOC_IDS_FILENAME):
            if not (d / name).exists():
                raise FileNotFoundError(f"{d / name} not found")
        import faiss  # local import

        idx = faiss.read_index(str(d / index_filename))
        vectors = np.load(d / embeddings_filename)
        doc_ids = json.loads((d / DOC_IDS_FILENAME).read_text(encoding="utf-8"))
        if len(doc_ids) != int(idx.ntotal):  # type: ignore[attr-defined]
            raise ValueError(
                f"doc_ids length ({len(doc_ids)}) != faiss ntotal "
                f"({idx.ntotal})"  # type: ignore[attr-defined]
            )
        inst = cls()
        inst._index = idx  # type: ignore[assignment]
        inst.vectors = vectors
        inst.doc_ids = doc_ids
        return inst
