"""Dense-retrieval service configuration.

Centralizes paths and defaults for the FAISS + sentence-transformers
service on port 8003. Mirrors the structure of
``services.indexing.app.config`` so the two services feel symmetric.

The Phase 1 preprocessed corpora are *not* the input here -- the
sentence-transformer model has its own WordPiece BPE tokenizer and
expects natural text. So the build script reads from
``data/processed/{ds}/docs.jsonl`` (the raw text), not ``tokens.jsonl``.
"""

from __future__ import annotations

import os
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────
# Paths (relative to the project root; resolved at import time)
# ─────────────────────────────────────────────────────────────────────────

# services/retrieval/app/config.py -> project root is 4 levels up.
PROJECT_ROOT = Path(__file__).resolve().parents[3]

# Phase 1 writes the raw docs here.
DATA_ROOT: Path = PROJECT_ROOT / "data" / "processed"

# Phase 2/3 share the same per-dataset index directory.
INDEX_ROOT: Path = PROJECT_ROOT / "data" / "indexes"

# Where sentence-transformers caches downloaded model weights.
# ``make download-models`` populates this; the lazy load path in
# ``embedder.py`` also writes here if it has to download.
MODEL_CACHE_ROOT: Path = PROJECT_ROOT / "data" / "models"

# Allowed dataset ids. Mirrors ``shared.ir_common.schemas.DATASET_IDS``.
DATASETS: tuple[str, ...] = ("touche2020", "nq")

# ─────────────────────────────────────────────────────────────────────────
# Embedding model defaults
# ─────────────────────────────────────────────────────────────────────────

# Default encoder. 384-dim, fast on CPU, general-purpose.
# Guide §3.1: "all-MiniLM-L6-v2 (384-dim, fast on CPU)".
DEFAULT_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"

# Batch size for document encoding. 256 is the sweet spot on 12-core CPU
# without blowing RAM: 256 * 128 tokens * 4 bytes ≈ 130 KB activations.
# On a 4 GB GPU (e.g. GTX 1650) with fp16, 512 fits comfortably and is
# ~10% faster; the build script auto-bumps when a GPU is detected.
DEFAULT_BATCH_SIZE: int = 256
DEFAULT_BATCH_SIZE_GPU: int = 512

# Max sequence length the encoder will see. MiniLM truncates at 256
# tokens; longer docs are clipped (the head and tail of each doc).
MAX_SEQ_LENGTH: int = 256


# Embedding device. We auto-detect CUDA at import time: if a GPU is
# present and torch was built with CUDA support, the embedder lands on
# ``cuda``; otherwise it falls back to ``cpu``. Set the env var
# ``IR_EMBED_DEVICE=cuda|cpu`` to force one or the other.
def _detect_device() -> str:
    forced = os.environ.get("IR_EMBED_DEVICE")
    if forced in ("cpu", "cuda"):
        return forced
    try:
        import torch  # local; don't import at module top to keep cold

        # imports fast.
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


EMBED_DEVICE: str = _detect_device()

# Whether to cast the model to float16 on GPU. On Turing (GTX 1650, CC
# 7.5) and later, fp16 roughly doubles throughput with < 1% recall
# drop for MiniLM-L6-v2. Has no effect on CPU.
USE_FP16: bool = EMBED_DEVICE == "cuda"

# LRU cache size for loaded models. One model is ~400 MB; we keep at
# most one loaded at a time.
MODEL_CACHE_SIZE: int = 1


# ─────────────────────────────────────────────────────────────────────────
# FAISS defaults
# ─────────────────────────────────────────────────────────────────────────

# Both datasets are well under 1M vectors, so the exact IndexFlatIP
# wins on simplicity + reproducibility (the latter matters for Phase 9
# evaluation: every run with the same embeddings returns the same
# scores). When the corpus grows past ~1M, swap to IndexIVFFlat with
# nlist=4096, nprobe=16 (guide §3.3).
FAISS_INDEX_TYPE: str = "IndexFlatIP"


# ─────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────


def index_dir(dataset_id: str) -> Path:
    """Return the on-disk directory for a dataset's indexes."""
    return INDEX_ROOT / dataset_id


def docs_path(dataset_id: str) -> Path:
    """Return the path to a dataset's raw ``docs.jsonl`` (Phase 1 output)."""
    return DATA_ROOT / dataset_id / "docs.jsonl"


def model_cache_dir(model_name: str) -> Path:
    """Return the local cache directory for a sentence-transformers model.

    Maps the Hugging Face hub name to a filesystem-safe directory name
    (e.g. ``sentence-transformers/all-MiniLM-L6-v2`` ->
    ``sentence-transformers__all-MiniLM-L6-v2``).
    """
    safe = model_name.replace("/", "__")
    return MODEL_CACHE_ROOT / safe
