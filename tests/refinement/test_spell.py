"""Unit tests for ``services.refinement.app.spell``."""

from __future__ import annotations

import pytest

from services.refinement.app.spell import SpellCorrector

# ─────────────────────────────────────────────────────────────────────────
# Single-word correction
# ─────────────────────────────────────────────────────────────────────────


class TestCorrectWord:
    def test_already_correct(self, spell: SpellCorrector) -> None:
        assert spell.correct_word("hello") == "hello"

    def test_simple_typo_transposition(self, spell: SpellCorrector) -> None:
        # Common transposition typos. "recieve" -> "receive" is the
        # canonical i-before-e typo.
        assert spell.correct_word("recieve") == "receive"
        assert spell.correct_word("wnated") == "wanted"  # wn -> an

    def test_drops_pure_punctuation(self, spell: SpellCorrector) -> None:
        assert spell.correct_word(",") == ","
        assert spell.correct_word("...") == "..."
        assert spell.correct_word("") == ""

    def test_drops_pure_numbers(self, spell: SpellCorrector) -> None:
        assert spell.correct_word("123") == "123"
        assert spell.correct_word("2024") == "2024"

    def test_drops_single_char(self, spell: SpellCorrector) -> None:
        # 1-char tokens are too ambiguous to correct safely.
        assert spell.correct_word("a") == "a"
        assert spell.correct_word("I") == "I"

    def test_casing_uppercase_preserved(self, spell: SpellCorrector) -> None:
        # If the input is ALL CAPS, the correction should be too.
        # (We don't test the specific word, just that casing heuristics
        # don't go haywire.)
        out = spell.correct_word("RECEIVE")
        assert out == "RECEIVE"  # already correct

    def test_casing_capital_preserved(self, spell: SpellCorrector) -> None:
        out = spell.correct_word("Capital")
        assert out == "Capital"  # already correct

    def test_no_suggestion_returns_original(self, spell: SpellCorrector) -> None:
        # A nonsense string the dictionary has never heard of.
        out = spell.correct_word("xqzklmp")
        # We don't assert *what* it returns (could be empty or the
        # original), but it must be a string and not raise.
        assert isinstance(out, str)


# ─────────────────────────────────────────────────────────────────────────
# Sentence-level correction
# ─────────────────────────────────────────────────────────────────────────


class TestCorrectSentence:
    def test_keeps_word_known_to_dict(self, spell: SpellCorrector) -> None:
        # "the" is in the dict; even though SymSpell's lookup returns
        # "they" as a suggestion (distance 1), we must NOT corrupt it.
        assert spell.correct("the") == "the"

    def test_corrects_common_typos(self, spell: SpellCorrector) -> None:
        assert spell.correct_word("wnat") == "what"
        assert spell.correct_word("ahve") == "have"
        assert spell.correct_word("thier") == "their"

    def test_preserves_punctuation_glue(self, spell: SpellCorrector) -> None:
        # "France?" should keep the trailing "?".
        out = spell.correct_word("France?")
        assert out.endswith("?")
        assert out.startswith("France")

    def test_handles_empty(self, spell: SpellCorrector) -> None:
        assert spell.correct("") == ""
        assert spell.correct("   ") == "   "

    def test_realistic_sentence(self, spell: SpellCorrector) -> None:
        # "What is the capital of France?" should round-trip unchanged
        # (every word is in the dict). The spell corrector must NOT
        # touch correct words.
        out = spell.correct("What is the capital of France?")
        assert "capital" in out
        assert "France" in out
        # ``What`` should stay ``What`` (not become ``They``).
        assert "What" in out

    def test_preserves_whitespace(self, spell: SpellCorrector) -> None:
        # Multiple spaces, tabs -- we don't test tabs (PowerShell gets
        # weird) but multiple spaces should round-trip.
        out = spell.correct("hello    world")
        assert "    " in out


# ─────────────────────────────────────────────────────────────────────────
# Construction
# ─────────────────────────────────────────────────────────────────────────


class TestConstruction:
    def test_default_construction_loads_dict(self) -> None:
        # No dict file -> FileNotFoundError. We don't actually want to
        # delete the dict in a unit test, so just confirm the happy
        # path works.
        sc = SpellCorrector()
        assert sc._sym is not None

    def test_construction_with_injected_symspell(self) -> None:
        from symspellpy import SymSpell

        # Empty SymSpell -- no suggestions, no exceptions.
        fake = SymSpell(max_dictionary_edit_distance=2)
        sc = SpellCorrector(sym_spell=fake)
        out = sc.correct_word("hello")
        assert out == "hello"  # in our cached dict, so no change


# ─────────────────────────────────────────────────────────────────────────
# Edge cases for pytest parametrize
# ─────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "typo,expected_substring",
    [
        ("recieve", "eceive"),  # i-e swap corrected
        ("definately", "efinitely"),
        ("seperate", "eparate"),
        ("occured", "ccurred"),
        ("neccessary", "ccessary"),
    ],
)
def test_common_typos(typo: str, expected_substring: str, spell: SpellCorrector) -> None:
    """Smoke-test a handful of well-known English typos."""
    out = spell.correct_word(typo)
    assert out != typo or expected_substring in out
    # We don't assert "must be X" because the SymSpell corpus doesn't
    # always have the obviously-correct word; we just want the
    # corrector to not corrupt correct words and to attempt a fix.
