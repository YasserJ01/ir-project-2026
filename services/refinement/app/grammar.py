"""Grammar correction for the refinement service.

Thin wrapper around ``language_tool_python.LanguageTool``. **The grammar
stage is OFF by default** (see ``config.GRAMMAR_ENABLED_DEFAULT``) for
three reasons:

1. **First-call cost is real**: ``language_tool_python`` starts a Java
   subprocess and downloads a ~200 MB ``.jar`` from the LanguageTool
   CDN on first use.
2. **Cold start latency**: the JVM warm-up adds 3-10 s to the first
   /refine call.
3. **Our hardware**: 4 Mbps downstream means the .jar takes 5-8
   minutes to download.

The wrapper survives that gracefully:
- ``build_grammar_corrector()`` is a no-op if ``GRAMMAR_ENABLED_DEFAULT=False``
  -- the function returns ``None`` and the pipeline's grammar stage
  becomes a pass-through.
- The ``.jar`` is cached under ``data/grammar/`` (set via env var
  ``LTP_JAR_DIR``) so subsequent runs are JVM-warm-up only.
- If the user opts in but the download fails, the pipeline returns
  the input unchanged and logs a warning (it does NOT raise).

Grammar correction is **stage 1** of the pipeline -- it runs before
spell correction, so spell-check sees the cleaned text. (This matches
the guide's §4.3 ordering.)
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from typing import Protocol

from services.refinement.app.config import (
    GRAMMAR_AUTO_DOWNLOAD,
    GRAMMAR_ENABLED_DEFAULT,
    GRAMMAR_LANGUAGE,
    LT_DATA_DIR,
)

__all__ = [
    "GrammarCorrector",
    "build_grammar_corrector",
    "is_grammar_enabled",
]

logger = logging.getLogger(__name__)


class _GrammarBackend(Protocol):
    """Subset of ``LanguageTool`` we use -- lets us stub it in tests."""

    def check(self, text: str) -> list: ...
    def close(self) -> None: ...


def is_grammar_enabled() -> bool:
    """True iff the grammar stage is on for this service run."""
    return GRAMMAR_ENABLED_DEFAULT


def _make_grammar_corrector() -> _GrammarBackend | None:
    """Build a ``LanguageTool`` instance, downloading the .jar if needed.

    Returns ``None`` if grammar is disabled by config. Raises a clear
    ``RuntimeError`` if the underlying library fails to initialize.
    """
    if not GRAMMAR_ENABLED_DEFAULT:
        return None

    # ``language_tool_python`` reads ``LTP_JAR_DIR`` to know where to
    # cache the .jar. We set it before importing the library.
    LT_DATA_DIR.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("LTP_JAR_DIR", str(LT_DATA_DIR))

    try:
        import language_tool_python  # type: ignore[import-untyped]
    except ImportError as exc:  # pragma: no cover - dep installed
        raise RuntimeError(
            "language_tool_python is not installed. "
            "Run `pip install language-tool-python` (or `make install`)."
        ) from exc

    try:
        # ``download_if_missing=False`` so the wrapper can decide; we
        # want a clear error message rather than a 200 MB surprise.
        tool: _GrammarBackend = language_tool_python.LanguageTool(  # type: ignore[call-arg]
            language=GRAMMAR_LANGUAGE,
            download_if_missing=GRAMMAR_AUTO_DOWNLOAD,
        )
        return tool
    except Exception as exc:  # pragma: no cover - depends on env
        # Don't fail the whole service. The pipeline will fall back to
        # a no-op for the grammar stage.
        logger.warning("Failed to initialize LanguageTool (%s); grammar stage will be no-op.", exc)
        return None


@lru_cache(maxsize=1)
def build_grammar_corrector() -> _GrammarBackend | None:
    """Return a process-wide ``LanguageTool`` instance, or None if disabled."""
    return _make_grammar_corrector()


class GrammarCorrector:
    """High-level wrapper with a ``correct(text)`` method.

    Safe to instantiate even when grammar is disabled -- ``correct()``
    will just return the input unchanged.
    """

    def __init__(self, backend: _GrammarBackend | None = None) -> None:
        # ``backend=None`` means: try to build, but tolerate failure.
        if backend is None:
            backend = build_grammar_corrector()
        self._backend = backend

    @property
    def enabled(self) -> bool:
        return self._backend is not None

    def correct(self, text: str) -> str:
        """Return a grammar-corrected version of ``text`` (input unchanged on failure)."""
        if not text or self._backend is None:
            return text
        try:
            matches = self._backend.check(text)
        except Exception as exc:  # pragma: no cover - depends on backend health
            logger.warning("LanguageTool.check failed (%s); returning input.", exc)
            return text
        if not matches:
            return text
        # ``LanguageTool`` returns matches in left-to-right order; iterate
        # in reverse and apply replacements in-place on a mutable buffer.
        # This is the canonical pattern from language_tool_python's docs.
        corrected = text
        for match in reversed(matches):
            if not match.replacements:
                continue
            replacement = match.replacements[0]
            corrected = (
                corrected[: match.offset]
                + replacement
                + corrected[match.offset + match.errorLength :]
            )
        return corrected

    def close(self) -> None:
        """Release the Java subprocess (no-op if not enabled)."""
        if self._backend is None:
            return
        try:
            self._backend.close()
        except Exception:  # pragma: no cover - cleanup is best-effort
            pass
