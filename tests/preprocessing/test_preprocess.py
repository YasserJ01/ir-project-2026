"""Tests for ``shared.ir_common.preprocess`` — the Phase 1 contract."""

from __future__ import annotations

from shared.ir_common.preprocess import (
    drop_non_alpha,
    drop_short,
    preprocess,
    preprocess_batch,
    preprocess_cached,
    remove_stopwords,
    stem_tokens,
    strip_html,
)

# ─────────────────────────────────────────────────────────────────────────
# Individual pipeline steps
# ─────────────────────────────────────────────────────────────────────────


def test_strip_html_removes_tags() -> None:
    # Each <...> is replaced with a single space; the tokenizer collapses
    # the resulting whitespace. We just assert that no angle-brackets remain
    # and the inner text is present.
    out = strip_html("<p>Hello <b>world</b>!</p>")
    assert "<" not in out and ">" not in out
    assert "Hello" in out and "world" in out and "!" in out


def test_strip_html_keeps_inner_text() -> None:
    out = strip_html("<a href='x'>click</a> here")
    assert "<" not in out
    assert "click" in out and "here" in out


def test_stem_tokens_porter_deterministic() -> None:
    # Porter: running -> run, runs -> run, foxes -> fox
    assert stem_tokens(["running", "runs", "foxes", "university"]) == [
        "run",
        "run",
        "fox",
        "univers",
    ]


def test_remove_stopwords_drops_english_stopwords() -> None:
    out = remove_stopwords(["the", "quick", "brown", "fox", "is", "on", "a"])
    assert "the" not in out and "is" not in out and "a" not in out
    assert "quick" in out and "brown" in out and "fox" in out


def test_drop_short_default_min_length_2() -> None:
    # Default drops single-character tokens (a, I, ., ,).
    assert drop_short(["a", "I", "be", "see"]) == ["be", "see"]


def test_drop_non_alpha_drops_pure_punctuation() -> None:
    assert drop_non_alpha(["...", "--", "...", "co2", "xbox1", ""]) == ["co2", "xbox1"]


# ─────────────────────────────────────────────────────────────────────────
# The full pipeline (preprocess)
# ─────────────────────────────────────────────────────────────────────────


def test_preprocess_lowercases() -> None:
    assert preprocess("Hello WORLD") == ["hello", "world"]


def test_preprocess_returns_list_of_str() -> None:
    out = preprocess("The cats are sleeping on the mat.")
    assert isinstance(out, list)
    assert all(isinstance(t, str) for t in out)


def test_preprocess_stems_tokens() -> None:
    # 'running' and 'runs' should collapse to the same stem.
    out = preprocess("running runs")
    assert "run" in out
    assert out.count("run") == 2


def test_preprocess_full_sentence_doctest_example() -> None:
    # Matches the example in the docstring of ``preprocess``.
    assert preprocess("The quick brown foxes were running fast.") == [
        "quick",
        "brown",
        "fox",
        "run",
        "fast",
    ]


# ─────────────────────────────────────────────────────────────────────────
# Edge cases
# ─────────────────────────────────────────────────────────────────────────


def test_preprocess_empty_string_returns_empty_list() -> None:
    assert preprocess("") == []


def test_preprocess_punctuation_only_returns_empty_list() -> None:
    assert preprocess("!!! ??? ...") == []


def test_preprocess_strips_html_before_tokenizing() -> None:
    # Make sure the <p> tag is gone from the output, not just visible to humans.
    out = preprocess("<p>Hello, World!</p>")
    # "p" alone is removed by drop_short(); "hello" and "world" survive.
    assert "hello" in out and "world" in out
    # No stray tag fragments
    assert all("<" not in t and ">" not in t for t in out)


def test_preprocess_unicode_normalizes_ligature() -> None:
    # NFKC turns the 'fi' ligature (\ufb01) into the two letters 'fi'.
    # After lowercasing + tokenizing, we get "office" (assuming a single \ufb01 in "ofﬁce").
    out = preprocess("ﬁle")  # noqa: RUF001 — this test is *about* the ligature
    assert out == ["file"]


def test_preprocess_batch_streams_lazily() -> None:
    gen = preprocess_batch(["foo bar", "", "baz qux"])
    assert next(gen) == ["foo", "bar"]
    assert next(gen) == []
    assert next(gen) == ["baz", "qux"]


def test_preprocess_cached_returns_tuple_and_is_repeatable() -> None:
    a = preprocess_cached("Hello World")
    b = preprocess_cached("Hello World")
    assert isinstance(a, tuple)
    assert a == b
    assert a == ("hello", "world")


# ─────────────────────────────────────────────────────────────────────────
# Contract: only one definition of ``preprocess`` ships from the project
# ─────────────────────────────────────────────────────────────────────────


def test_preprocess_is_the_canonical_function() -> None:
    # If anyone ever adds a second ``preprocess`` in services/* or scripts/*,
    # this import should still resolve to the shared one (it does — that's the
    # whole point of the single source of truth). The real regression guard
    # lives in the smoke-test grep in ``docs/PHASE_1.md §8``.
    from shared.ir_common import preprocess as reexported

    assert reexported is preprocess
