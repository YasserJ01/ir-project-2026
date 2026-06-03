"""Indexing service configuration.

Centralizes paths so the build script, the service, and the smoke
script all agree on where to find corpora and where to write indexes.

Why a dedicated config module? Three reasons:
  1. The ``scripts/build_indexes.py`` and the FastAPI service both need
     the same paths; one source of truth avoids drift.
  2. Tests can monkeypatch ``DATA_ROOT`` / ``INDEX_ROOT`` to point at a
     ``tmp_path`` without touching environment variables.
  3. Future phases (3+) will add FAISS / dense-representation paths
     here too, in the same shape.
"""

from __future__ import annotations

from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────
# Paths (relative to the project root; resolved at import time)
# ─────────────────────────────────────────────────────────────────────────

# services/indexing/app/config.py -> project root is 4 levels up.
PROJECT_ROOT = Path(__file__).resolve().parents[3]

# Phase 1 writes the tokenized corpora here.
DATA_ROOT: Path = PROJECT_ROOT / "data" / "processed"

# Phase 2 writes the per-dataset indexes here.
INDEX_ROOT: Path = PROJECT_ROOT / "data" / "indexes"

# Allowed dataset ids. Mirrors ``shared.ir_common.schemas.DATASET_IDS``
# so the service can use either. The single source of truth is the
# schema module; this constant exists for ergonomic imports.
DATASETS: tuple[str, ...] = ("touche2020", "nq")

# ─────────────────────────────────────────────────────────────────────────
# InvertedIndex defaults (overridable per build)
# ─────────────────────────────────────────────────────────────────────────

# Drop terms that appear in fewer than this many docs. Singleton terms
# are 30-50% of the vocabulary but contribute disproportionately to
# memory (each one holds its own dict + each entry holds a doc_id str
# and an int). Default 2 is the standard IR choice; raise to 3 or 5 on
# tighter RAM.
DEFAULT_MIN_DF: int = 2

# Drop terms that appear in more than this fraction of the corpus. 0.5
# drops the top half by df -- a common "stopword-by-frequency" cutoff
# for the *indexed* terms (the preprocessor has already removed NLTK
# stopwords). Set to 1.0 to disable.
DEFAULT_MAX_DF_RATIO: float = 0.5

# ─────────────────────────────────────────────────────────────────────────
# BM25 defaults (overridable at query time)
# ─────────────────────────────────────────────────────────────────────────

# These match the guide's recommendation (k1=1.5, b=0.75) and are also
# the Lucene BM25Okapi defaults.
DEFAULT_BM25_K1: float = 1.5
DEFAULT_BM25_B: float = 0.75

# LRU cache size for (k1, b) -> BM25 instances. With 8 slots, typical
# slider-tweaking stays in cache.
BM25_CACHE_SIZE: int = 8


def index_dir(dataset_id: str) -> Path:
    """Return the on-disk directory for a dataset's indexes."""
    return INDEX_ROOT / dataset_id


def tokens_path(dataset_id: str) -> Path:
    """Return the path to a dataset's tokens.jsonl (Phase 1 output)."""
    return DATA_ROOT / dataset_id / "tokens.jsonl"
