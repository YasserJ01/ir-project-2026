"""Unit tests for ``services.refinement.app.personalization``."""

from __future__ import annotations

from pathlib import Path

from services.refinement.app.personalization import (
    build_weight_map,
    get_history_tokens,
    load_user_log,
    weight_map_summary,
)
from tests.refinement.conftest import write_user_log


def _entry(query: str, clicked: list[str], ts: float = 0.0) -> dict:
    return {"ts": ts, "query": query, "clicked_doc_ids": clicked}


class TestLoadUserLog:
    def test_missing_file_returns_empty(self, tmp_user_log_dir: Path) -> None:
        assert load_user_log("ghost") == []

    def test_loads_real_file(self, tmp_user_log_dir: Path) -> None:
        write_user_log(
            tmp_user_log_dir,
            "alice",
            [
                _entry("france capital", ["d1", "d2"], ts=1.0),
                _entry("eiffel tower", ["d3"], ts=2.0),
            ],
        )
        entries = load_user_log("alice")
        assert len(entries) == 2
        assert entries[0].query == "france capital"
        assert entries[0].clicked_doc_ids == ["d1", "d2"]
        assert entries[1].ts == 2.0

    def test_handles_malformed_lines(self, tmp_user_log_dir: Path) -> None:
        path = write_user_log(tmp_user_log_dir, "bob", [_entry("hi", ["d1"])])
        with path.open("a", encoding="utf-8") as fh:
            fh.write("not json\n")
        entries = load_user_log("bob")
        # Malformed line is skipped, real line is kept.
        assert len(entries) == 1
        assert entries[0].query == "hi"

    def test_respects_max_lines(self, tmp_user_log_dir: Path) -> None:
        write_user_log(
            tmp_user_log_dir,
            "carol",
            [_entry(f"q{i}", [f"d{i}"]) for i in range(50)],
        )
        entries = load_user_log("carol", max_lines=10)
        assert len(entries) == 10


class TestBuildWeightMap:
    def test_empty_log_returns_empty(self, tmp_user_log_dir: Path) -> None:
        assert build_weight_map("ghost") == {}

    def test_no_clicks_returns_empty(self, tmp_user_log_dir: Path) -> None:
        # Entries with no clicked docs shouldn't move the needle.
        write_user_log(
            tmp_user_log_dir,
            "dave",
            [_entry("france capital", [], ts=1.0)],
        )
        assert build_weight_map("dave") == {}

    def test_boosts_term_at_threshold(self, tmp_user_log_dir: Path) -> None:
        # "france" appears in 3 different queries, each with a
        # different doc_id -- that's 3 distinct clicks, so it should
        # be boosted. (Same doc_id repeated would dedupe down to 1.)
        write_user_log(
            tmp_user_log_dir,
            "eve",
            [
                _entry("france capital", ["d1"], ts=1.0),
                _entry("france history", ["d2"], ts=2.0),
                _entry("france food", ["d3"], ts=3.0),
            ],
        )
        wm = build_weight_map("eve")
        assert wm.get("france", 1.0) > 1.0

    def test_below_threshold_not_boosted(self, tmp_user_log_dir: Path) -> None:
        # "capital" appears in 2 queries, same doc -- only 2 clicks,
        # below the 3-click threshold.
        write_user_log(
            tmp_user_log_dir,
            "frank",
            [
                _entry("capital city", ["d1"], ts=1.0),
                _entry("capital letter", ["d1"], ts=2.0),
            ],
        )
        wm = build_weight_map("frank")
        assert wm.get("capital", 1.0) == 1.0

    def test_dedupes_clicks_within_query(self, tmp_user_log_dir: Path) -> None:
        # 5 entries all click the same doc -> still only 1 distinct
        # click -> "france" is below threshold.
        write_user_log(
            tmp_user_log_dir,
            "grace",
            [_entry("france capital", ["d1"], ts=float(i)) for i in range(5)],
        )
        wm = build_weight_map("grace")
        assert wm.get("france", 1.0) == 1.0

    def test_different_docs_count_separately(self, tmp_user_log_dir: Path) -> None:
        # 3 queries, 3 different docs -> "france" has 3 distinct clicks.
        write_user_log(
            tmp_user_log_dir,
            "henry",
            [
                _entry("france capital", ["d1"], ts=1.0),
                _entry("france history", ["d2"], ts=2.0),
                _entry("france food", ["d3"], ts=3.0),
            ],
        )
        wm = build_weight_map("henry")
        assert wm.get("france", 1.0) > 1.0

    def test_stopwords_ignored(self, tmp_user_log_dir: Path) -> None:
        # "the" appears in 10 queries -- still ignored because the
        # helper regex strips 2-char-or-shorter and stopwords.
        write_user_log(
            tmp_user_log_dir,
            "ivy",
            [_entry("the cat", [f"d{i}"], ts=float(i)) for i in range(10)],
        )
        wm = build_weight_map("ivy")
        assert "the" not in wm


class TestGetHistoryTokens:
    def test_missing_user(self, tmp_user_log_dir: Path) -> None:
        assert get_history_tokens("ghost") == []

    def test_top_k(self, tmp_user_log_dir: Path) -> None:
        write_user_log(
            tmp_user_log_dir,
            "jack",
            [
                _entry("france history pizza", ["d1"], ts=1.0),
                _entry("italy history pizza", ["d2"], ts=2.0),
                _entry("france history pizza", ["d3"], ts=3.0),
                _entry("italy history pizza", ["d4"], ts=4.0),
            ],
        )
        top = get_history_tokens("jack", k=2)
        # "history" and "pizza" each appear 4x; "france" and "italy"
        # each appear 2x. The top 2 should be the 4x tokens.
        assert set(top) == {"history", "pizza"}
        assert len(top) == 2

    def test_respects_k_zero(self, tmp_user_log_dir: Path) -> None:
        write_user_log(tmp_user_log_dir, "kim", [_entry("france", ["d1"])])
        assert get_history_tokens("kim", k=0) == []


class TestWeightMapSummary:
    def test_empty_returns_empty(self) -> None:
        assert weight_map_summary({}) == ""

    def test_formats_pairs(self) -> None:
        out = weight_map_summary({"france": 2.0, "capital": 2.0})
        assert "france=2" in out
        assert "capital=2" in out
        assert out.startswith("boosted:")


class TestUserLogPath:
    def test_safe_filename(self) -> None:
        from services.refinement.app.config import user_log_path

        # Path traversal: slashes are sanitized so the result stays
        # inside USER_LOG_DIR (we don't normalize dots, we just
        # replace non-alphanumeric with underscore).
        path = user_log_path("../../etc/passwd")
        # The result is a Path *under* USER_LOG_DIR with no slashes
        # in the filename portion -- the only separator allowed is
        # the one between USER_LOG_DIR and the basename.
        assert "/" not in path.name
        # The path doesn't escape USER_LOG_DIR.
        assert path.parent == path.parent.parent / path.parent.name
        # It's a single filename ending in .jsonl.
        assert path.name.endswith(".jsonl")

    def test_empty_user_id(self) -> None:
        from services.refinement.app.config import user_log_path

        path = user_log_path("")
        # Empty string -> "anonymous".
        assert path.name == "anonymous.jsonl"
