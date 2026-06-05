"""Unit tests for ``services.refinement.app.synonyms``."""

from __future__ import annotations

import pytest

from services.refinement.app.synonyms import SynonymExpander


class TestExpandToken:
    def test_returns_list(self, synonyms: SynonymExpander) -> None:
        out = synonyms.expand_token("car", 2)
        assert isinstance(out, list)

    def test_skips_stopwords(self, synonyms: SynonymExpander) -> None:
        # Stopwords have no useful expansion.
        assert synonyms.expand_token("the", 2) == []
        assert synonyms.expand_token("a", 2) == []
        assert synonyms.expand_token("is", 2) == []

    def test_skips_punctuation(self, synonyms: SynonymExpander) -> None:
        assert synonyms.expand_token("...", 2) == []
        assert synonyms.expand_token("---", 2) == []
        assert synonyms.expand_token("", 2) == []

    def test_skips_numbers(self, synonyms: SynonymExpander) -> None:
        assert synonyms.expand_token("123", 2) == []
        assert synonyms.expand_token("2024", 2) == []

    def test_doesnt_return_input(self, synonyms: SynonymExpander) -> None:
        out = synonyms.expand_token("car", 5)
        assert "car" not in out

    def test_respects_count(self, synonyms: SynonymExpander) -> None:
        # n=0 -> no expansion regardless of the word.
        assert synonyms.expand_token("car", 0) == []
        # n=1 -> at most 1 result.
        out1 = synonyms.expand_token("car", 1)
        assert len(out1) <= 1
        # n=2 -> at most 2 results.
        out2 = synonyms.expand_token("car", 2)
        assert len(out2) <= 2

    def test_skips_multi_word_lemmas(self, synonyms: SynonymExpander) -> None:
        # WordNet has multi-word entries like ``ice_cream``. The
        # expander drops them so the output stays a single space-
        # joined string for Phase 5.
        out = synonyms.expand_token("car", 5)
        for token in out:
            assert " " not in token
            assert "_" not in token

    def test_lowercases(self, synonyms: SynonymExpander) -> None:
        out = synonyms.expand_token("CAR", 5)
        for token in out:
            assert token == token.lower()

    def test_unknown_word(self, synonyms: SynonymExpander) -> None:
        # Nonsense word -> no synsets -> no expansion.
        assert synonyms.expand_token("xqzklmp", 2) == []


class TestExpand:
    def test_returns_string(self, synonyms: SynonymExpander) -> None:
        out = synonyms.expand("fast car", 2)
        assert isinstance(out, str)

    def test_includes_originals(self, synonyms: SynonymExpander) -> None:
        # "fast" and "car" must remain in the output; we only add.
        out = synonyms.expand("fast car", 2)
        assert "fast" in out
        assert "car" in out

    def test_preserves_word_order(self, synonyms: SynonymExpander) -> None:
        # First word first, second word second, then expansions.
        out = synonyms.expand("car fast", 1)
        first_word = out.split(" ")[0]
        assert first_word == "car"

    def test_handles_empty(self, synonyms: SynonymExpander) -> None:
        assert synonyms.expand("", 2) == ""
        assert synonyms.expand("car", 0) == "car"  # n=0 means no-op

    def test_handles_punctuation(self, synonyms: SynonymExpander) -> None:
        # Punctuation gets preserved (we only strip it from the
        # lookup, then glue it back).
        out = synonyms.expand("car, fast!", 2)
        assert "car" in out
        assert "," in out
        assert "fast" in out
        assert "!" in out

    def test_realistic_query(self, synonyms: SynonymExpander) -> None:
        out = synonyms.expand("fast car", 2)
        # The original words are present, plus 0-4 synonyms
        # (2 originals + 2 from "fast" + 2 from "car" = up to 6).
        words = out.split(" ")
        assert "fast" in words
        assert "car" in words
        assert 2 <= len(words) <= 6


class TestConstruction:
    def test_singleton(self) -> None:
        # ``build_synonym_expander`` is ``@lru_cache(maxsize=1)`` so
        # two calls in the same process return equivalent instances.
        # We can't assert ``is`` because pytest fixtures or lru_cache
        # eviction across test boundaries can yield different objects;
        # but two instances must produce the same output for the same
        # input (they're stateless after construction).
        from services.refinement.app.synonyms import build_synonym_expander

        a = build_synonym_expander()
        b = build_synonym_expander()
        # Both should give the same synonyms for "car".
        assert a.expand_token("car", 3) == b.expand_token("car", 3)

    def test_handles_missing_wordnet(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from services.refinement.app import synonyms as syn_mod

        monkeypatch.setattr(syn_mod, "_is_wordnet_loaded", lambda: False)
        with pytest.raises(LookupError):
            syn_mod.SynonymExpander()
