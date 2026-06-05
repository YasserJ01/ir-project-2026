"""The 4-stage query-refinement pipeline.

Order (per guide 4.3):

    1. Grammar correction    (``grammar.GrammarCorrector``)
    2. Spell correction      (``spell.SpellCorrector``)
    3. Synonym expansion     (``synonyms.SynonymExpander``)
    4. Personalization       (``personalization.build_weight_map``)
    5. Tokenize              (``shared/ir_common/preprocess.preprocess``)

The pipeline is **pure** -- no global state, no I/O side effects. The
caller (``service.py``) owns latency timing and builds the
``RefineResponse``.

The pipeline accepts ``RefineRequest`` and returns a small intermediate
``PipelineResult`` with all the fields the response needs.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

from services.refinement.app.grammar import GrammarCorrector
from services.refinement.app.personalization import (
    build_weight_map,
    weight_map_summary,
)
from services.refinement.app.spell import SpellCorrector
from services.refinement.app.synonyms import SynonymExpander
from shared.ir_common.preprocess import preprocess
from shared.ir_common.schemas import RefinedToken, RefineRequest

__all__ = ["PipelineResult", "RefinementPipeline", "build_pipeline"]

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Intermediate result of the pipeline. The service maps this 1:1 to ``RefineResponse``."""

    original_query: str
    refined_query: str = ""
    expanded_query: str = ""
    tokens: list[str] = field(default_factory=list)
    weighted_tokens: list[RefinedToken] = field(default_factory=list)
    stages: dict[str, str] = field(default_factory=dict)
    user_id: str = ""


class RefinementPipeline:
    """Glue object that owns the four stage instances + orchestrates them."""

    def __init__(
        self,
        grammar: GrammarCorrector | None = None,
        spell: SpellCorrector | None = None,
        synonyms: SynonymExpander | None = None,
    ) -> None:
        # Grammar defaults to "no grammar corrector" (it's off by config).
        self.grammar = grammar if grammar is not None else GrammarCorrector()
        # Spell + synonyms are mandatory -- we always have them.
        self.spell = spell if spell is not None else SpellCorrector()
        self.synonyms = synonyms if synonyms is not None else SynonymExpander()

    def _run_grammar(self, request: RefineRequest) -> tuple[str, str]:
        if not request.enable_grammar or not self.grammar.enabled:
            return request.query, ""
        corrected = self.grammar.correct(request.query)
        if corrected == request.query:
            return request.query, ""
        return corrected, corrected

    def _run_spell(self, request: RefineRequest, text: str) -> tuple[str, str]:
        if not request.enable_spell:
            return text, ""
        corrected = self.spell.correct(text)
        if corrected == text:
            return text, ""
        return corrected, corrected

    def _run_synonyms(self, request: RefineRequest, text: str) -> tuple[str, str]:
        if not request.enable_synonyms or request.synonym_count <= 0:
            return text, ""
        expanded = self.synonyms.expand(text, n=request.synonym_count)
        if expanded == text:
            return text, ""
        return expanded, expanded

    def _run_personalization(
        self, request: RefineRequest, tokens: list[str]
    ) -> tuple[list[RefinedToken], str]:
        """Apply the user-history weight map to the tokenized output.

        Also flags any token that *was* in the user-history as
        ``added_by="personalization"`` -- this is informational, the
        weight itself is what Phase 5 will use for boosting.
        """
        if not request.enable_personalization:
            return [RefinedToken(token=t, weight=1.0, added_by="original") for t in tokens], ""

        weight_map = build_weight_map(request.user_id)
        if not weight_map:
            return [RefinedToken(token=t, weight=1.0, added_by="original") for t in tokens], ""

        out: list[RefinedToken] = []
        for t in tokens:
            w = weight_map.get(t, 1.0)
            boosted = w > 1.0
            out.append(
                RefinedToken(
                    token=t,
                    weight=w,
                    added_by="personalization" if boosted else "original",
                )
            )
        return out, weight_map_summary(weight_map)

    def run(self, request: RefineRequest) -> PipelineResult:
        """Execute the 5-stage pipeline. Returns a ``PipelineResult`` for the service."""
        result = PipelineResult(original_query=request.query, user_id=request.user_id)

        # Stage 1: grammar.
        after_grammar, grammar_trace = self._run_grammar(request)
        result.stages["grammar"] = grammar_trace

        # Stage 2: spell. The result of grammar+spell is the
        # ``refined_query`` exposed to callers -- it's the cleaned
        # text BEFORE synonym expansion.
        after_spell, spell_trace = self._run_spell(request, after_grammar)
        result.stages["spell"] = spell_trace
        result.refined_query = after_spell

        # Stage 3: synonyms. ``expanded_query`` is the final
        # text the tokenizer sees.
        after_synonyms, syn_trace = self._run_synonyms(request, after_spell)
        result.stages["synonyms"] = syn_trace
        result.expanded_query = after_synonyms

        # Stage 4: tokenize via the shared preprocessing pipeline.
        result.tokens = preprocess(after_synonyms)

        # Stage 5: personalization weight map.
        result.weighted_tokens, pers_trace = self._run_personalization(request, result.tokens)
        result.stages["personalization"] = pers_trace

        return result


def build_pipeline() -> RefinementPipeline:
    """Factory used by the service. Eagerly builds the four stage instances."""
    grammar = GrammarCorrector()  # no-op if GRAMMAR_ENABLED_DEFAULT=False
    spell = SpellCorrector()
    synonyms = SynonymExpander()
    return RefinementPipeline(grammar=grammar, spell=spell, synonyms=synonyms)


def measure_latency_ms(start: float) -> int:
    """Helper for the service. Returns integer ms since ``start``."""
    return int((time.perf_counter() - start) * 1000)
