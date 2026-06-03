"""Corpus loading helpers for the indexing service.

Centralizes the (annoying) details of streaming the Phase 1
``tokens.jsonl`` files without holding the whole corpus in memory until
the caller actually needs it.

Three entry points:
  - ``stream_tokens(dataset_id)`` -- one (doc_id, tokens) at a time
  - ``load_tokenized_corpus(dataset_id)`` -- materializes the full
    list[list[str]] in memory (used by the build script and the test
    suite)
  - ``get_doc_ids(dataset_id)`` -- just the doc_ids, in JSONL order
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

from services.indexing.app.config import tokens_path

# Cap on how many docs we return from ``load_tokenized_corpus`` if a
# caller asks for a sample. 0 means "no cap" (return everything).
DEFAULT_LOAD_CAP: int = 0


def stream_tokens(dataset_id: str, path: Path | None = None) -> Iterator[tuple[str, list[str]]]:
    """Yield ``(doc_id, tokens)`` pairs from a dataset's tokens.jsonl.

    The caller controls memory by iterating; this function holds at most
    one line in RAM at a time. Used by the build script to build
    InvertedIndex / TF-IDF / BM25 in a single streaming pass.
    """
    p = path or tokens_path(dataset_id)
    if not p.exists():
        raise FileNotFoundError(
            f"tokens.jsonl for '{dataset_id}' not found at {p}. "
            "Run `make ingest-{a,b}` then `make tokenize` first."
        )
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            yield row["id"], row["tokens"]


def load_tokenized_corpus(
    dataset_id: str, cap: int = DEFAULT_LOAD_CAP
) -> tuple[list[str], list[list[str]]]:
    """Materialize the full tokenized corpus into memory.

    Returns ``(doc_ids, corpus_tokens)`` where ``corpus_tokens[i]`` is
    the list of stemmed tokens for ``doc_ids[i]``. Used by the build
    script to feed sklearn TfidfVectorizer and bm25s.

    ``cap > 0`` returns the first ``cap`` docs only (used by tests).
    """
    doc_ids: list[str] = []
    corpus: list[list[str]] = []
    for i, (doc_id, tokens) in enumerate(stream_tokens(dataset_id)):
        doc_ids.append(doc_id)
        corpus.append(tokens)
        if cap and i + 1 >= cap:
            break
    return doc_ids, corpus


def get_doc_ids(dataset_id: str) -> list[str]:
    """Return only the doc_ids from a dataset's tokens.jsonl.

    Cheaper than ``load_tokenized_corpus`` for the case where the
    caller only needs the id->position mapping. We still parse the
    whole file (the doc_id is in every line), but we don't allocate the
    tokens list -- saves a few hundred MB for the big corpora.
    """
    doc_ids: list[str] = []
    for doc_id, _ in stream_tokens(dataset_id):
        doc_ids.append(doc_id)
    return doc_ids
