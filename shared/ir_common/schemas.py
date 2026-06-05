"""Pydantic schemas shared across the IR services.

Single source of truth for the wire format. The gateway (Phase 6) and the
React UI's ``src/types/api.ts`` (Phase 7) both import these (the latter
via JSON Schema generation). Phase 2 ships the search-related models;
later phases add refinement, RAG, and dataset-list models.

All fields are deliberately permissive (Pydantic v2 with ``model_config``
allowing extra fields forward-compat) so a Phase 5 caller can send a
``fusion`` field to a Phase 2 server without a 422.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

# ─────────────────────────────────────────────────────────────────────────
# Datasets
# ─────────────────────────────────────────────────────────────────────────

# Allowed dataset ids. Kept here (not in config) so the schema is the
# source of truth that the React UI and the gateway both validate against.
DATASET_IDS: tuple[str, ...] = ("touche2020", "nq")

SearchModel = Literal["inverted", "tfidf", "bm25", "dense"]


# ─────────────────────────────────────────────────────────────────────────
# Build
# ─────────────────────────────────────────────────────────────────────────


class BuildRequest(BaseModel):
    """Body for ``POST /index/{dataset_id}/build``."""

    model_config = ConfigDict(extra="forbid")

    min_df: int = Field(2, ge=1, description="Drop terms that appear in fewer than this many docs.")
    max_df_ratio: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Drop terms that appear in more than this fraction of docs.",
    )
    bm25_method: Literal["lucene", "atire", "robertson", "bm25l", "bm25plus"] = Field(
        default="lucene", description="BM25 variant. 'lucene' is the BM25Okapi equivalent."
    )


class BuildResponse(BaseModel):
    """Response to ``POST /index/{dataset_id}/build`` (202 Accepted)."""

    model_config = ConfigDict(extra="forbid")

    dataset_id: str
    started: bool = True
    job_id: str
    message: str = "Build started in background. Poll /index/{dataset_id}/stats to see completion."


# ─────────────────────────────────────────────────────────────────────────
# Stats
# ─────────────────────────────────────────────────────────────────────────


class StatsResponse(BaseModel):
    """Response to ``GET /index/{dataset_id}/stats``."""

    model_config = ConfigDict(extra="forbid")

    dataset_id: str
    exists: bool
    loaded: bool = False
    vocab_size: int = 0
    total_docs: int = 0
    avg_doc_length: float = 0.0
    build_seconds: float = 0.0
    build_at: str = ""
    size_mb: float = 0.0
    cap: dict[str, int | float] = Field(
        default_factory=dict, description="InvertedIndex vocabulary cap (min_df, max_df_ratio)."
    )


# ─────────────────────────────────────────────────────────────────────────
# Search
# ─────────────────────────────────────────────────────────────────────────


class SearchRequest(BaseModel):
    """Body for ``POST /index/{dataset_id}/search``."""

    # Forward-compat: Phase 5 callers may send a `fusion` field. We don't
    # 422 on it; we just ignore it at this layer (the gateway picks it up).
    model_config = ConfigDict(extra="ignore")

    query_tokens: list[str] | None = Field(
        default=None,
        min_length=1,
        max_length=512,
        description=(
            "Pre-tokenized, stemmed query (output of shared/ir_common/preprocess). "
            "Required for `model in {inverted, tfidf, bm25}`; ignored when "
            "`model='dense'`."
        ),
    )
    query: str | None = Field(
        default=None,
        max_length=2048,
        description=(
            "Raw query text. Required when `model='dense'` (the encoder has its "
            "own WordPiece BPE tokenizer); ignored for the other models."
        ),
    )
    model: SearchModel = Field("bm25", description="Which retriever to use.")
    k: int = Field(10, ge=1, le=1000, description="Number of results to return.")
    # BM25 hyperparameters. Ignored by `tfidf`, `inverted`, `dense` models.
    k1: float = Field(1.5, ge=0.0, le=10.0, description="BM25 term-frequency saturation.")
    b: float = Field(0.75, ge=0.0, le=1.0, description="BM25 length normalization.")


class SearchResult(BaseModel):
    """One hit in a ``SearchResponse``."""

    model_config = ConfigDict(extra="forbid")

    rank: int = Field(..., ge=1, description="1-based rank.")
    doc_id: str
    score: float


class SearchResponse(BaseModel):
    """Response to ``POST /index/{dataset_id}/search``."""

    model_config = ConfigDict(extra="forbid")

    dataset_id: str
    model: SearchModel
    k: int
    latency_ms: int
    results: list[SearchResult]
    # Echo back the params so the client can confirm what was used.
    k1: float | None = None
    b: float | None = None
    cached: bool = False


# ─────────────────────────────────────────────────────────────────────────
# Postings (debug)
# ─────────────────────────────────────────────────────────────────────────


class PostingsRequest(BaseModel):
    """Query for ``GET /index/{dataset_id}/postings/{term}``."""

    model_config = ConfigDict(extra="forbid")

    cap: int = Field(1000, ge=1, le=10000, description="Maximum number of postings to return.")


class Posting(BaseModel):
    """One (doc_id, tf) entry in a postings list."""

    model_config = ConfigDict(extra="forbid")

    doc_id: str
    tf: int = Field(..., ge=1)


class PostingsResponse(BaseModel):
    """Response for the postings debug endpoint."""

    model_config = ConfigDict(extra="forbid")

    dataset_id: str
    term: str
    doc_freq: int
    postings: list[Posting]
    truncated: bool = False


# ─────────────────────────────────────────────────────────────────────────
# Service health
# ─────────────────────────────────────────────────────────────────────────


class HealthResponse(BaseModel):
    """Response to ``GET /health``."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["ok"] = "ok"
    service: str = "indexing"
    loaded_dataset: str | None = None
    version: str = "0.1.0"


# ─────────────────────────────────────────────────────────────────────────
# Phase 3 — Dense retrieval (port 8003)
# ─────────────────────────────────────────────────────────────────────────


class DenseBuildRequest(BaseModel):
    """Body for ``POST /retrieval/{dataset_id}/build`` (Phase 3)."""

    model_config = ConfigDict(extra="forbid")

    model_name: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="Hugging Face model name. Must be a sentence-transformer.",
    )
    batch_size: int = Field(256, ge=1, le=2048, description="Encode batch size.")


class DenseStatsResponse(BaseModel):
    """Response to ``GET /retrieval/{dataset_id}/stats``."""

    model_config = ConfigDict(extra="forbid")

    dataset_id: str
    exists: bool
    loaded: bool = False
    num_vectors: int = 0
    dim: int = 0
    index_type: str = "IndexFlatIP"
    model_name: str = ""
    build_seconds: float = 0.0
    build_at: str = ""
    size_mb: float = 0.0


class DenseEmbedRequest(BaseModel):
    """Body for ``POST /retrieval/embed`` (one-shot embed)."""

    model_config = ConfigDict(extra="forbid")

    texts: list[str] = Field(
        ...,
        min_length=1,
        max_length=1024,
        description="Raw texts to embed (1-1024 per call).",
    )
    model_name: str | None = Field(
        default=None,
        description=(
            "Override the default model. If None, the service default "
            "(sentence-transformers/all-MiniLM-L6-v2) is used."
        ),
    )


class DenseEmbedResponse(BaseModel):
    """Response to ``POST /retrieval/embed``."""

    model_config = ConfigDict(extra="forbid")

    model_name: str
    dim: int
    vectors: list[list[float]]
    latency_ms: int


class DenseSearchHit(BaseModel):
    """One hit in a ``DenseSearchResponse``."""

    model_config = ConfigDict(extra="forbid")

    rank: int = Field(..., ge=1)
    doc_id: str
    score: float


class DenseSearchResponse(BaseModel):
    """Response to ``POST /retrieval/{dataset_id}/search``."""

    model_config = ConfigDict(extra="forbid")

    dataset_id: str
    model_name: str
    k: int
    latency_ms: int
    results: list[DenseSearchHit]
    cached: bool = False


class RetrievalHealthResponse(BaseModel):
    """Response to ``GET /health`` on the retrieval service (port 8003)."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["ok"] = "ok"
    service: str = "retrieval"
    loaded_dataset: str | None = None
    model_loaded: bool = False
    model_name: str = ""
    version: str = "0.1.0"


# ─────────────────────────────────────────────────────────────────────────
# Phase 4 — Query refinement (port 8004)
# ─────────────────────────────────────────────────────────────────────────


class RefineRequest(BaseModel):
    """Body for ``POST /refine`` (Phase 4).

    The request takes a raw user query and optional user-id + toggles for
    each refinement stage. The service returns the enriched query,
    pre-tokenized tokens, and a per-token weight map (defaults to 1.0,
    with ``enable_personalization=true`` boosting terms the user has
    previously clicked in).
    """

    model_config = ConfigDict(extra="ignore")

    query: str = Field(
        ...,
        min_length=1,
        max_length=2048,
        description="Raw user query (English assumed; mixed case OK).",
    )
    user_id: str = Field(
        default="anonymous",
        max_length=128,
        description=(
            "Identifier for personalization. Looks up "
            "``data/user_logs/<user_id>.jsonl``. Missing/empty log file = "
            "personalization is a no-op."
        ),
    )
    enable_spell: bool = Field(True, description="Apply SymSpell edit-distance correction.")
    enable_synonyms: bool = Field(
        True, description="Expand each non-stopword with 1-2 WordNet synonyms."
    )
    enable_grammar: bool = Field(
        False,
        description=(
            "Apply language-tool-python grammar correction. **Off by default** because "
            "it spins up a Java subprocess and downloads a ~200 MB .jar on first use. "
            "Enable for the highest quality (slowest cold start)."
        ),
    )
    enable_personalization: bool = Field(
        True, description="Apply user-log-based term-weight boost."
    )
    # Synonym expansion knob (per guide 4.2: "1-2 synonyms per non-stopword").
    synonym_count: int = Field(
        2,
        ge=0,
        le=5,
        description="Max synonyms per token (0 disables synonym stage regardless of toggle).",
    )


class RefinedToken(BaseModel):
    """One token in the refined response: (term, weight)."""

    model_config = ConfigDict(extra="forbid")

    token: str = Field(
        ..., description="Preprocessed + stemmed token (output of shared/ir_common/preprocess)."
    )
    weight: float = Field(
        1.0, ge=0.0, description="Personalization boost multiplier (1.0 = no boost)."
    )
    added_by: Literal["original", "spell", "synonym", "grammar", "personalization"] = Field(
        "original",
        description=(
            "Which refinement stage introduced this token. ``original`` means it "
            "was in the cleaned input; ``spell`` means a corrected form of an "
            "original token (same surface text but different letters were wrong); "
            "``synonym`` means it came from WordNet expansion; ``personalization`` "
            "means it was a user-history term that got added to the weight map."
        ),
    )


class RefineResponse(BaseModel):
    """Response to ``POST /refine``."""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(..., description="Original query, echoed back.")
    refined_query: str = Field(
        ..., description="Query after grammar + spell correction (before synonym expansion)."
    )
    expanded_query: str = Field(
        ...,
        description="Final query string used for tokenization (cleaned + synonyms joined).",
    )
    tokens: list[str] = Field(
        ..., description="Preprocessed, stemmed token list (shared preprocess)."
    )
    weighted_tokens: list[RefinedToken] = Field(
        ..., description="Same tokens + per-token weights for personalized scoring."
    )
    # Per-stage trace so a debug client can see what each module did.
    stages: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Mapping of stage-name -> output. Keys: ``grammar``, ``spell``, ``synonyms``, "
            "``personalization``. Empty string = stage was a no-op or disabled."
        ),
    )
    latency_ms: int = Field(..., ge=0, description="Total pipeline latency in milliseconds.")
    user_id: str = Field(..., description="User id that was looked up.")


class RefinementHealthResponse(BaseModel):
    """Response to ``GET /health`` on the refinement service (port 8004)."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["ok"] = "ok"
    service: str = "refinement"
    spell_loaded: bool = False
    wordnet_loaded: bool = False
    grammar_loaded: bool = False
    grammar_enabled: bool = False
    user_log_dir: str = ""
    version: str = "0.1.0"
