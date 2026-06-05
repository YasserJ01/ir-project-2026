"""Unit tests for ``services.refinement.app.grammar``."""

from __future__ import annotations

from services.refinement.app.grammar import GrammarCorrector, is_grammar_enabled


class TestGrammarCorrector:
    def test_disabled_by_default(self, grammar: GrammarCorrector) -> None:
        # The default config has ``GRAMMAR_ENABLED_DEFAULT=False``,
        # so the corrector is a no-op even when grammar is "off".
        assert grammar.enabled is False
        assert is_grammar_enabled() is False

    def test_correct_returns_input_when_disabled(self, grammar: GrammarCorrector) -> None:
        # The whole point of the disabled corrector is to be a
        # pass-through -- so any input goes back unchanged.
        assert grammar.correct("i has a apple") == "i has a apple"
        assert grammar.correct("") == ""

    def test_close_when_disabled(self, grammar: GrammarCorrector) -> None:
        # Must not raise even when the backend is None.
        grammar.close()


class TestGrammarEnabled:
    def test_injected_backend_overrides_disabled(self) -> None:
        """A backend injected at construction is honored even when
        ``GRAMMAR_ENABLED_DEFAULT`` is False. (This is how tests can
        opt in to the grammar stage without flipping the global.)
        """

        class _FakeBackend:
            def check(self, text: str) -> list:
                # Pretend LanguageTool found one match.
                class _Match:
                    offset = 0
                    errorLength = 1
                    replacements = ["I"]

                return [_Match()]

            def close(self) -> None:
                pass

        gc = GrammarCorrector(backend=_FakeBackend())  # type: ignore[arg-type]
        assert gc.enabled is True
        assert gc.correct("i has a apple") == "I has a apple"

    def test_injected_backend_no_matches(self) -> None:
        class _NoMatchBackend:
            def check(self, text: str) -> list:
                return []

            def close(self) -> None:
                pass

        gc = GrammarCorrector(backend=_NoMatchBackend())  # type: ignore[arg-type]
        assert gc.correct("good grammar here") == "good grammar here"

    def test_injected_backend_failure_falls_back(self) -> None:
        class _BrokenBackend:
            def check(self, text: str) -> list:
                raise RuntimeError("Java crashed")

            def close(self) -> None:
                pass

        gc = GrammarCorrector(backend=_BrokenBackend())  # type: ignore[arg-type]
        # Pipeline should silently return input, not raise.
        assert gc.correct("text") == "text"
