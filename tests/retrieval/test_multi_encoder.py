"""Unit tests for ``services/retrieval/app/multi_encoder.py``.

The multi-encoder runner depends on a search function (encoding +
FAISS lookup) and on the 2nd-encoder FAISS index being on disk. We
mock both:

  * :func:`fake_search_l6` and :func:`fake_search_l12` return canned
    results so the two encoders are easy to tell apart.
  * The on-disk check is monkeypatched via the conftest fixture
    ``monkeypatch`` -- we set
    ``services.retrieval.app.config.has_second_encoder_index`` to a
    ``lambda ds: True`` so the test doesn't need a real FAISS file.
"""

from __future__ import annotations

import pytest

from services.retrieval.app import config as config_mod
from services.retrieval.app import multi_encoder as me_mod
from services.retrieval.app.multi_encoder import (
    DEFAULT_ENCODER_1,
    DEFAULT_ENCODER_2,
    MultiEncoderError,
    MultiEncoderRunner,
    _short_name,
    build_default_multi_encoder_search,
)
from shared.ir_common.schemas import (
    MultiEncoderSearchRequest,
)

# ─────────────────────────────────────────────────────────────────────────
# Fakes
# ─────────────────────────────────────────────────────────────────────────


async def fake_search_l6(
    query_text: str, dataset_id: str, model_name: str, k: int
) -> tuple[list[float], list[str]]:
    """A fake L6 search. Uses overlap-by-tokens, biased toward L6 results."""
    if "fox" in query_text.lower():
        return [2.0, 1.5], ["d5", "d1"]  # L6 prefers d5
    return [1.0], ["d5"]


async def fake_search_l12(
    query_text: str, dataset_id: str, model_name: str, k: int
) -> tuple[list[float], list[str]]:
    """A fake L12 search. Different ranking to make the fusion test meaningful."""
    if "fox" in query_text.lower():
        return [1.8, 1.2], ["d1", "d5"]  # L12 prefers d1
    return [0.8], ["d1"]


def make_runner(
    has_index: bool = True,
) -> MultiEncoderRunner:
    """Build a runner with the 2-encoder fakes."""
    return MultiEncoderRunner(search_fn=_two_encoder_search(has_index))


def _two_encoder_search(
    has_index: bool,
) -> me_mod.MultiEncoderSearchFn:
    """Return a search fn that dispatches L6 vs L12."""

    async def _fn(
        query_text: str, dataset_id: str, model_name: str, k: int
    ) -> tuple[list[float], list[str]]:
        if not has_index:
            # The production code checks has_second_encoder_index
            # before reaching here, so this branch shouldn't fire.
            raise MultiEncoderError("no index")
        if model_name == DEFAULT_ENCODER_1:  # L6
            return await fake_search_l6(query_text, dataset_id, model_name, k)
        if model_name == DEFAULT_ENCODER_2:  # L12
            return await fake_search_l12(query_text, dataset_id, model_name, k)
        raise ValueError(f"unknown model {model_name}")

    return _fn


# ─────────────────────────────────────────────────────────────────────────
# _short_name
# ─────────────────────────────────────────────────────────────────────────


def test_short_name_l6() -> None:
    assert _short_name("sentence-transformers/all-MiniLM-L6-v2") == "l6"


def test_short_name_l12() -> None:
    assert _short_name("sentence-transformers/all-MiniLM-L12-v2") == "l12"


def test_short_name_unknown_unchanged() -> None:
    # An unrecognised model goes through unchanged.
    assert (
        _short_name("sentence-transformers/all-mpnet-base-v2")
        == "sentence-transformers/all-mpnet-base-v2"
    )


# ─────────────────────────────────────────────────────────────────────────
# Runner tests (with the 2-encoder fake + monkeypatched index check)
# ─────────────────────────────────────────────────────────────────────────


@pytest.fixture
def patched_index_exists(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pretend the 2nd-encoder FAISS index is on disk."""
    monkeypatch.setattr(config_mod, "has_second_encoder_index", lambda ds: True)


@pytest.fixture
def patched_index_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pretend the 2nd-encoder FAISS index is NOT on disk."""
    monkeypatch.setattr(config_mod, "has_second_encoder_index", lambda ds: False)


async def test_runner_uses_default_encoders(
    patched_index_exists: None,
) -> None:
    runner = make_runner()
    req = MultiEncoderSearchRequest(query="fox", k=2)
    resp = await runner.search("touche2020", req)
    # Both encoders were used (the timings dict has both keys).
    assert "l6" in resp.per_retriever_latency_ms
    assert "l12" in resp.per_retriever_latency_ms
    # The stages show which encoder produced which.
    assert "l6" in resp.stages
    assert "l12" in resp.stages


async def test_runner_rejects_missing_index(
    patched_index_missing: None,
) -> None:
    runner = make_runner(has_index=False)
    req = MultiEncoderSearchRequest(query="fox", k=2)
    with pytest.raises(MultiEncoderError, match="not built"):
        await runner.search("touche2020", req)


async def test_runner_rejects_same_encoder_twice(
    patched_index_exists: None,
) -> None:
    runner = make_runner()
    req = MultiEncoderSearchRequest(
        query="fox", k=2, encoder_1=DEFAULT_ENCODER_1, encoder_2=DEFAULT_ENCODER_1
    )
    with pytest.raises(MultiEncoderError, match="must be different"):
        await runner.search("touche2020", req)


async def test_runner_fuses_with_rrf(
    patched_index_exists: None,
) -> None:
    runner = make_runner()
    req = MultiEncoderSearchRequest(query="fox", k=2, fusion="rrf")
    resp = await runner.search("touche2020", req)
    # The fusion combines L6 (d5 first) and L12 (d1 first).
    # RRF: d5 = 1/(k+1) + 1/(k+2), d1 = 1/(k+2) + 1/(k+1) -> tied.
    # Tie-break: doc_id ascending -> d1, d5.
    assert [h.doc_id for h in resp.results] == ["d1", "d5"]
    # individual_scores has both encoder keys.
    for h in resp.results:
        assert "l6" in h.individual_scores
        assert "l12" in h.individual_scores


async def test_runner_fuses_with_combsum(
    patched_index_exists: None,
) -> None:
    runner = make_runner()
    req = MultiEncoderSearchRequest(query="fox", k=2, fusion="combsum")
    resp = await runner.search("touche2020", req)
    # CombSUM: each doc is max in one encoder, so both = 1.0; tied.
    assert [h.doc_id for h in resp.results] == ["d1", "d5"]


async def test_runner_fuses_with_combmnz(
    patched_index_exists: None,
) -> None:
    runner = make_runner()
    req = MultiEncoderSearchRequest(query="fox", k=2, fusion="combmnz")
    resp = await runner.search("touche2020", req)
    # CombMNZ: same as CombSUM here (count_nonzero = 2 for both docs).
    assert [h.doc_id for h in resp.results] == ["d1", "d5"]


async def test_runner_truncates_to_k(
    patched_index_exists: None,
) -> None:
    runner = make_runner()
    req = MultiEncoderSearchRequest(query="fox", k=1, fusion="rrf")
    resp = await runner.search("touche2020", req)
    assert len(resp.results) == 1
    # With only k=1 slot, the doc that's rank-1 in BOTH retrievers
    # wins. Neither d5 nor d1 is rank-1 in both (d5 is rank-1 in L6,
    # d1 is rank-1 in L12) -- they tie. doc_id ascending -> d1.
    assert resp.results[0].doc_id == "d1"


async def test_runner_custom_encoders(
    patched_index_exists: None,
) -> None:
    """The encoder_1 / encoder_2 override fields are honoured."""
    runner = make_runner()

    async def _fake_custom(
        query_text: str, dataset_id: str, model_name: str, k: int
    ) -> tuple[list[float], list[str]]:
        if model_name == "custom/encoder-A":
            return [1.0], ["dA"]
        if model_name == "custom/encoder-B":
            return [1.0], ["dB"]
        raise ValueError(model_name)

    runner._search = _fake_custom  # type: ignore[attr-defined]
    req = MultiEncoderSearchRequest(
        query="anything",
        k=1,
        encoder_1="custom/encoder-A",
        encoder_2="custom/encoder-B",
    )
    resp = await runner.search("touche2020", req)
    # The short names for non-default models are the full model names.
    assert "custom/encoder-A" in resp.stages
    assert "custom/encoder-B" in resp.stages
    # k=1 -> top-1 result, tied at 1.0, doc_id ascending -> dA wins.
    assert [h.doc_id for h in resp.results] == ["dA"]


# ─────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────


def test_build_default_multi_encoder_search_returns_callable() -> None:
    fn = build_default_multi_encoder_search()
    # The production closure is a coroutine function.
    import inspect

    assert inspect.iscoroutinefunction(fn)
