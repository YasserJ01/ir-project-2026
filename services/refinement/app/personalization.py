"""User-log-based personalization for the refinement service.

The personalization module reads a per-user ``.jsonl`` log of past
queries + clicked doc_ids, decides which terms are "important" to
this user, and returns a weight map that the pipeline can apply to
its tokenized output.

File format (one JSON object per line, ``data/user_logs/<user_id>.jsonl``)::

    {"ts": 1717520000.0, "query": "what is the capital of france", "clicked_doc_ids": ["doc123", "doc456"]}
    {"ts": 1717520050.0, "query": "history of the eiffel tower",     "clicked_doc_ids": ["doc123"]}

Algorithm (per guide 4.2):

> For each token, if the user has clicked 3+ docs containing a related
> term in the past, boost that term's weight (simple +1 multiplier).

We interpret "containing a related term" as "the clicked doc's
*display* terms" -- but at write time we don't have a document index
to look that up. So we use a simpler proxy: **the union of tokens
across the user's past queries**, filtered by the click count of
documents that contained each token. Specifically:

1. Read up to ``PERSONALIZATION_LOG_MAX_LINES`` log lines.
2. Build ``token_to_clicks: dict[str, set[str]]`` = for each token, the
   set of doc_ids whose click brought us to that token (via the
   query that was searched).
3. If a token's click-set size >= ``PERSONALIZATION_CLICK_THRESHOLD``,
   mark it "boosted".

This is the simple-mock version the guide calls for: "Start by
simulating 'user 1' with 50 hand-crafted past queries (we'll
iterate)."

We then export two functions:

- ``build_weight_map(user_id) -> dict[str, float]`` -- the boost map.
- ``get_history_tokens(user_id, k) -> list[str]`` -- the top-K
  most-clicked tokens (for the "expansion" stage to consider).
"""

from __future__ import annotations

import json
import logging
import re
from collections import Counter
from pathlib import Path

from services.refinement.app.config import (
    PERSONALIZATION_BOOST,
    PERSONALIZATION_CLICK_THRESHOLD,
    PERSONALIZATION_LOG_MAX_LINES,
    USER_LOG_DIR,
    user_log_path,
)

__all__ = [
    "UserLogEntry",
    "load_user_log",
    "build_weight_map",
    "get_history_tokens",
    "weight_map_summary",
]

logger = logging.getLogger(__name__)

# Minimal tokenizer for log lines: lowercase, alpha-only, drop <2 char.
_WORD_RE = re.compile(r"[a-zA-Z]+")
_STOP = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "of",
    "to",
    "in",
    "on",
    "at",
    "for",
    "by",
    "with",
    "from",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "this",
    "that",
    "these",
    "those",
    "i",
    "you",
    "we",
    "they",
    "he",
    "she",
    "it",
    "my",
    "your",
    "our",
    "their",
    "do",
    "does",
    "did",
    "what",
    "which",
    "who",
    "whom",
    "how",
    "where",
    "when",
    "why",
}


def _tokenize_for_log(text: str) -> list[str]:
    return [w.lower() for w in _WORD_RE.findall(text) if len(w) >= 2 and w.lower() not in _STOP]


class UserLogEntry:
    """One row of a user-log .jsonl file."""

    __slots__ = ("ts", "query", "clicked_doc_ids")

    def __init__(self, ts: float, query: str, clicked_doc_ids: list[str]) -> None:
        self.ts = ts
        self.query = query
        self.clicked_doc_ids = clicked_doc_ids

    @classmethod
    def from_dict(cls, d: dict) -> UserLogEntry:
        return cls(
            ts=float(d.get("ts", 0.0)),
            query=str(d.get("query", "")),
            clicked_doc_ids=[str(x) for x in d.get("clicked_doc_ids", [])],
        )


def load_user_log(
    user_id: str, max_lines: int = PERSONALIZATION_LOG_MAX_LINES
) -> list[UserLogEntry]:
    """Return the (truncated) list of log entries for ``user_id``.

    Missing file or empty file = empty list. This is the **no-op**
    behavior the guide calls for: "If the user has clicked 3+ docs
    containing a related term in the past" -- 0 clicks means no
    boost, which is the correct fallback.
    """
    path = user_log_path(user_id)
    if not path.exists():
        return []
    entries: list[UserLogEntry] = []
    try:
        with path.open("r", encoding="utf-8") as fh:
            for i, line in enumerate(fh):
                if i >= max_lines:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning("Skipping malformed JSONL line in %s", path)
                    continue
                entries.append(UserLogEntry.from_dict(obj))
    except OSError as exc:
        logger.warning("Failed to read user log %s: %s", path, exc)
        return []
    return entries


def build_weight_map(user_id: str) -> dict[str, float]:
    """Return ``{token: weight}`` for tokens the user has clicked >= 3 times.

    Weight is 1.0 + ``PERSONALIZATION_BOOST`` (i.e. 2.0 by default).
    """
    entries = load_user_log(user_id)
    if not entries:
        return {}

    # token -> distinct doc_ids that triggered a click while this token
    # was in the searched query. We approximate "doc contains token"
    # by "token was in the query" -- close enough for the guide's
    # "simulate user 1 with 50 hand-crafted queries" use case.
    token_clicks: dict[str, set[str]] = {}
    for entry in entries:
        if not entry.clicked_doc_ids:
            continue
        tokens = set(_tokenize_for_log(entry.query))
        for tok in tokens:
            for doc_id in entry.clicked_doc_ids:
                token_clicks.setdefault(tok, set()).add(doc_id)

    weight_map: dict[str, float] = {}
    for tok, doc_ids in token_clicks.items():
        if len(doc_ids) >= PERSONALIZATION_CLICK_THRESHOLD:
            weight_map[tok] = 1.0 + PERSONALIZATION_BOOST
    return weight_map


def get_history_tokens(user_id: str, k: int = 20) -> list[str]:
    """Return the top-K most-frequent tokens in this user's query history.

    Used by the ``personalization`` stage of the pipeline to consider
    "expansion candidates" the user has shown interest in. Currently
    the pipeline only *weights* (not expands), so this is a hook for
    Phase 5+ personalization work.
    """
    entries = load_user_log(user_id)
    if not entries or k <= 0:
        return []
    counter: Counter[str] = Counter()
    for entry in entries:
        counter.update(_tokenize_for_log(entry.query))
    return [tok for tok, _ in counter.most_common(k)]


def weight_map_summary(weight_map: dict[str, float]) -> str:
    """Return a human-readable string for the ``stages`` trace field."""
    if not weight_map:
        return ""
    boosted = sorted(weight_map.items(), key=lambda kv: (-kv[1], kv[0]))
    parts = [f"{tok}={w:g}" for tok, w in boosted]
    return "boosted: " + ", ".join(parts)


def ensure_user_log_dir() -> Path:
    """Create the user-log dir if it doesn't exist. Returns the dir."""
    USER_LOG_DIR.mkdir(parents=True, exist_ok=True)
    return USER_LOG_DIR
