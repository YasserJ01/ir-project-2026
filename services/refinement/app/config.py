"""Query-refinement service configuration.

Centralizes paths and defaults for the spell-correction, synonym-expansion,
grammar-correction, and personalization pipeline on port 8004.

Mirrors the structure of ``services.indexing.app.config`` and
``services.retrieval.app.config`` so all three services feel symmetric.

The pipeline is **stateless** across requests -- the only persisted
state is the user-log directory, which lives on disk under
``data/user_logs/``.
"""

from __future__ import annotations

from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────
# Paths (relative to the project root; resolved at import time)
# ─────────────────────────────────────────────────────────────────────────

# services/refinement/app/config.py -> project root is 4 levels up.
PROJECT_ROOT = Path(__file__).resolve().parents[3]

# SymSpell word-frequency dictionary. Downloaded once by
# ``scripts/download_symspell_dict.py`` (~1.3 MB, 82,765 entries).
DICT_ROOT: Path = PROJECT_ROOT / "data" / "dicts"
SPELL_DICT_PATH: Path = DICT_ROOT / "frequency_dictionary_en_82_765.txt"

# User-log directory. One ``.jsonl`` per user_id; the personalization
# module reads them on demand (lazy). If the file is missing for a given
# user_id, personalization is a no-op for that request.
USER_LOG_DIR: Path = PROJECT_ROOT / "data" / "user_logs"

# Where language-tool-python's Java subprocess stores its downloaded
# ``.jar``. The library defaults to ``~/languagetool/`` -- we redirect
# into the project so it's tracked, deletable, and reproducible.
LT_DATA_DIR: Path = PROJECT_ROOT / "data" / "grammar"

# ─────────────────────────────────────────────────────────────────────────
# Spell correction (SymSpell)
# ─────────────────────────────────────────────────────────────────────────

# Max edit distance considered for spell suggestions. Guide 4.2 leaves
# this open; SymSpell's default is 2. 1 is too tight (typos like
# "recieve" -> "receive" need distance 2), 3 is too lax (suggesting
# nonsense). 2 is the standard choice.
SPELL_MAX_EDIT_DISTANCE: int = 2

# Prefix length for SymSpell's lookup table. 7 is the default and works
# well for English. Larger = faster lookup, more RAM.
SPELL_PREFIX_LENGTH: int = 7

# ─────────────────────────────────────────────────────────────────────────
# Synonym expansion (WordNet)
# ─────────────────────────────────────────────────────────────────────────

# WordNet POS tags to consider when fetching synonyms. We pull from all
# three (NOUN, VERB, ADJ/ADV/SAT) and dedupe. This is more permissive
# than the guide's "1-2 synonyms" but we still cap to ``synonym_count``
# in the request, so the wire format stays the same.
WORDNET_POS: tuple[str, ...] = ("n", "v", "a", "s", "r")

# Skip these (lowercase) when expanding -- a synonym of "the" is just
# noise. The module also drops NLTK's stopword list at run time.
SYNONYM_SKIP: frozenset[str] = frozenset(
    {
        "",
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
    }
)

# ─────────────────────────────────────────────────────────────────────────
# Grammar correction (language-tool-python)
# ─────────────────────────────────────────────────────────────────────────

# **Off by default** -- language-tool-python starts a Java subprocess
# and downloads a ~200 MB ``.jar`` on first use. That first-call cost
# is real (5-15 s on a cold JVM + 30+ s jar download on a 4 Mbps line).
# Tests + the smoke script leave it off; production callers can opt in
# per-request via ``enable_grammar=true``.
GRAMMAR_ENABLED_DEFAULT: bool = False

# Default language for grammar check. "en-US" is the safest English
# variant (also covers en-GB for the most part).
GRAMMAR_LANGUAGE: str = "en-US"

# Auto-download LanguageTool on first use. If False, the grammar module
# raises a helpful error pointing to ``make download-grammar``.
GRAMMAR_AUTO_DOWNLOAD: bool = True

# ─────────────────────────────────────────────────────────────────────────
# Personalization
# ─────────────────────────────────────────────────────────────────────────

# Number of past clicks required before a term gets a weight boost.
# Guide 4.2: "3+ docs".
PERSONALIZATION_CLICK_THRESHOLD: int = 3

# Weight multiplier applied to a boosted term. Guide 4.2: "simple +1
# multiplier" -- we interpret that as a 1.0 *additive* boost on top of
# the baseline weight of 1.0, so a boosted term has weight 2.0.
PERSONALIZATION_BOOST: float = 1.0

# Cap on how many user-log lines we read into memory. 10,000 is the
# default -- more than that and the user has a search history longer
# than any human's lifetime (~10 queries/day * 3 years).
PERSONALIZATION_LOG_MAX_LINES: int = 10_000

# ─────────────────────────────────────────────────────────────────────────
# Pipeline tuning
# ─────────────────────────────────────────────────────────────────────────

# If the spell module's input is longer than this, we still run, but
# we truncate the candidates list to keep latency predictable. 1,024
# chars is a generous bound (most queries are < 100 chars).
MAX_QUERY_CHARS: int = 2048

# Whether to eagerly initialize SymSpell + WordNet at service startup
# (True) or lazily on first request (False). Lazy = faster cold start;
# eager = first request is faster.
EAGER_INIT: bool = True


# ─────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────


def user_log_path(user_id: str) -> Path:
    """Return the on-disk path to a user-log file.

    A safe filename: alpha-numeric, dashes, underscores, dots. Anything
    else is collapsed to '_' to prevent path traversal.
    """
    safe = "".join(c if c.isalnum() or c in ("-", "_", ".") else "_" for c in user_id)
    if not safe:
        safe = "anonymous"
    return USER_LOG_DIR / f"{safe}.jsonl"


def stage_label(stage: str) -> str:
    """Stable label for a pipeline stage (used in ``RefineResponse.stages``)."""
    return stage.strip().lower()
