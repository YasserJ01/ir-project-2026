"""Test fixtures for the query-refinement service.

We don't load the real sentence-transformer here (the refinement
service doesn't use one) but we do load:

* SymSpell + the dictionary (``build_spell_corrector``)
* WordNet (``build_synonym_expander``)
* Grammar corrector (``build_grammar_corrector`` -- returns None
  because ``GRAMMAR_ENABLED_DEFAULT=False``)

The ``tmp_user_log_dir`` fixture points personalization at a
``tmp_path`` so tests can write fake JSONL logs without polluting
``data/user_logs/``.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.refinement.app import config as cfg  # noqa: E402
from services.refinement.app import service as service_mod  # noqa: E402
from services.refinement.app.grammar import GrammarCorrector  # noqa: E402
from services.refinement.app.pipeline import RefinementPipeline  # noqa: E402
from services.refinement.app.spell import SpellCorrector  # noqa: E402
from services.refinement.app.synonyms import SynonymExpander  # noqa: E402


@pytest.fixture(scope="session")
def spell() -> SpellCorrector:
    return SpellCorrector()


@pytest.fixture(scope="session")
def synonyms() -> SynonymExpander:
    return SynonymExpander()


@pytest.fixture(scope="session")
def grammar() -> GrammarCorrector:
    """Grammar corrector -- may be a no-op if the .jar wasn't downloaded."""
    return GrammarCorrector()


@pytest.fixture(scope="session")
def pipeline(
    spell: SpellCorrector, synonyms: SynonymExpander, grammar: GrammarCorrector
) -> RefinementPipeline:
    return RefinementPipeline(grammar=grammar, spell=spell, synonyms=synonyms)


@pytest.fixture
def tmp_user_log_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """Redirect the user-log directory to a tmp dir for the duration of one test."""
    monkeypatch.setattr(cfg, "USER_LOG_DIR", tmp_path)
    # ``user_log_path`` closes over the module's USER_LOG_DIR via
    # ``Path / f"{safe}.jsonl"`` so re-importing it is the safest
    # way to make tests see the new dir.
    from services.refinement.app import personalization

    monkeypatch.setattr(personalization, "USER_LOG_DIR", tmp_path)
    yield tmp_path


@pytest.fixture
def client_factory(tmp_user_log_dir: Path) -> Iterator[TestClient]:  # noqa: ARG001
    """Yield a FastAPI ``TestClient``. The personalization module is
    already pointed at ``tmp_user_log_dir`` by the upstream fixture
    so ``/refine`` with ``user_id`` works against any log we write.
    """
    with TestClient(service_mod.app) as c:
        yield c


def write_user_log(log_dir: Path, user_id: str, entries: list[dict]) -> Path:
    """Helper: write a JSONL user-log file and return its path."""
    log_dir.mkdir(parents=True, exist_ok=True)
    safe = "".join(c if c.isalnum() or c in ("-", "_", ".") else "_" for c in user_id)
    path = log_dir / f"{safe}.jsonl"
    with path.open("w", encoding="utf-8") as fh:
        for entry in entries:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return path
