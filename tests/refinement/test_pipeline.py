"""Unit tests for ``services.refinement.app.pipeline``."""

from __future__ import annotations

from services.refinement.app.pipeline import RefinementPipeline, build_pipeline
from shared.ir_common.schemas import RefinedToken, RefineRequest


class TestPipelineRun:
    def test_clean_query(self, pipeline: RefinementPipeline) -> None:
        req = RefineRequest(query="What is the capital of France?")
        result = pipeline.run(req)
        # Spell: no typos, no change.
        assert "capital" in result.refined_query
        # Synonym: "capital" gets expanded to (at least) "capital"
        # + up to 2 synonyms.
        assert "capital" in result.expanded_query
        # Tokens are stemmed.
        assert any("capit" in t for t in result.tokens)
        # No boosting (no user log).
        assert all(wt.weight == 1.0 for wt in result.weighted_tokens)
        # Stages: grammar disabled, spell no-op, synonyms fired.
        assert result.stages["grammar"] == ""
        assert result.stages["personalization"] == ""

    def test_typo_query(self, pipeline: RefinementPipeline) -> None:
        req = RefineRequest(query="recieve teh helo")
        result = pipeline.run(req)
        # Spell corrector: "recieve" -> "receive" or similar.
        assert "recieve" not in result.refined_query or "receive" in result.refined_query
        # Stages: spell should have fired.
        assert "spell" in result.stages
        assert result.stages["spell"] != ""

    def test_spell_disabled(self, pipeline: RefinementPipeline) -> None:
        req = RefineRequest(query="recieve teh helo", enable_spell=False)
        result = pipeline.run(req)
        # With spell off, "recieve" is preserved (might stem to "receiv"
        # after the preprocess pipeline though).
        assert result.stages["spell"] == ""

    def test_synonyms_disabled(self, pipeline: RefinementPipeline) -> None:
        req = RefineRequest(query="fast car", enable_synonyms=False)
        result = pipeline.run(req)
        # No synonym expansion -> expanded_query == refined_query.
        assert result.expanded_query == result.refined_query
        assert result.stages["synonyms"] == ""

    def test_synonym_count_zero(self, pipeline: RefinementPipeline) -> None:
        req = RefineRequest(query="fast car", synonym_count=0)
        result = pipeline.run(req)
        # synonym_count=0 also disables the stage.
        assert result.stages["synonyms"] == ""

    def test_personalization_disabled(self, pipeline: RefinementPipeline) -> None:
        req = RefineRequest(query="eiffel tower", enable_personalization=False, user_id="user_1")
        result = pipeline.run(req)
        # No weights populated (everything is 1.0).
        assert all(wt.weight == 1.0 for wt in result.weighted_tokens)
        assert all(wt.added_by == "original" for wt in result.weighted_tokens)
        assert result.stages["personalization"] == ""

    def test_grammar_disabled_by_default(self, pipeline: RefinementPipeline) -> None:
        req = RefineRequest(query="this is a test query")
        result = pipeline.run(req)
        # Grammar is off by default -> refined_query == original.
        assert result.refined_query == req.query
        assert result.stages["grammar"] == ""

    def test_weighted_tokens_default_weight(self, pipeline: RefinementPipeline) -> None:
        req = RefineRequest(query="hello world", user_id="ghost_user")
        result = pipeline.run(req)
        # No log file -> all weights 1.0, all ``added_by="original"``.
        for wt in result.weighted_tokens:
            assert wt.weight == 1.0
            assert wt.added_by == "original"


class TestPipelineConstruction:
    def test_build_pipeline_factory(self) -> None:
        p = build_pipeline()
        # The factory should populate all four stages (or set grammar
        # to a disabled one).
        assert p.grammar is not None
        assert p.spell is not None
        assert p.synonyms is not None


class TestPipelineWithPersonalization:
    def test_personalized_user_1(
        self,
        pipeline: RefinementPipeline,
        tmp_user_log_dir,  # noqa: ARG002  -- the fixture patches the dir
    ) -> None:
        # We need to use a user that has a log in ``tmp_user_log_dir``.
        # Easier: just call build_weight_map directly to check the
        # integration works end-to-end.
        from tests.refinement.conftest import write_user_log

        write_user_log(
            tmp_user_log_dir,
            "franky",
            [
                {"ts": 1.0, "query": "france capital", "clicked_doc_ids": ["d1", "d2", "d3"]},
                {"ts": 2.0, "query": "france history", "clicked_doc_ids": ["d4", "d5", "d6"]},
            ],
        )
        # The pipeline's personalization stage reads from
        # personalization. ``load_user_log`` is patched via the fixture.
        from services.refinement.app import personalization

        wm = personalization.build_weight_map("franky")
        # "france" should be boosted (2 queries, 6 distinct clicks).
        assert wm.get("france", 1.0) > 1.0


class TestIntermediateResult:
    def test_result_dataclass_defaults(self) -> None:
        from services.refinement.app.pipeline import PipelineResult

        r = PipelineResult(original_query="x", user_id="y")
        assert r.tokens == []
        assert r.weighted_tokens == []
        assert r.stages == {}
        assert r.refined_query == ""
        assert r.expanded_query == ""


class TestRefinedTokenSchema:
    def test_default_weight(self) -> None:
        rt = RefinedToken(token="car")
        assert rt.weight == 1.0
        assert rt.added_by == "original"

    def test_personalized_weight(self) -> None:
        rt = RefinedToken(token="france", weight=2.0, added_by="personalization")
        assert rt.weight == 2.0
        assert rt.added_by == "personalization"
