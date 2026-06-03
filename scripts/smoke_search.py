"""Smoke test the indexing service end-to-end.

Phase 2 of the IR project. Loads each dataset's indexes from disk and
runs a few hand-picked queries against each of the three retrievers,
printing the top-3 docs. Used to eyeball the indexes before committing
and as a sanity check after a fresh build.

CLI:
    python scripts/smoke_search.py                           # all datasets
    python scripts/smoke_search.py --datasets touche2020
    python scripts/smoke_search.py --k 5                    # top-5 instead of 3
    python scripts/smoke_search.py --query "machine learning"  # add a custom query

The hand-picked queries are designed to be domain-relevant for each
corpus. For touche2020 (argument retrieval on debate topics), we use
queries about climate change and capital punishment. For nq (open-domain
QA on Wikipedia passages), we use factual questions.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

# Force UTF-8 on Windows.
try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
except Exception:
    pass

# Make `shared` and `services` importable when run as a script.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.indexing.app import bm25 as bm25_mod  # noqa: E402
from services.indexing.app import tfidf as tfidf_mod  # noqa: E402
from services.indexing.app.config import DATASETS, index_dir  # noqa: E402
from shared.ir_common.preprocess import preprocess  # noqa: E402

# Hand-picked queries per dataset. The first three are designed to be
# domain-relevant; the rest are bonuses.
DEFAULT_QUERIES: dict[str, list[str]] = {
    "touche2020": [
        "Should abortion be legalized?",  # debate topic
        "Is climate change caused by humans?",  # debate topic
        "Should the death penalty be abolished?",  # debate topic
    ],
    "nq": [
        "when was the declaration of independence signed",
        "what is the largest planet in the solar system",
        "how many continents are there in the world",
    ],
}

# We need a tiny way to render a doc's text from its doc_id. The
# retrieval services don't store the text; we read the original
# ``docs.jsonl`` on demand (which is gitignored but on disk). For
# brevity we only show the first 200 chars of each hit's text.


def _load_doc_text(dataset_id: str, doc_id: str) -> str:
    """Read the text for a doc_id from ``data/processed/<ds>/docs.jsonl``.

    Slow (linear scan) but only used for eyeballing 3-9 hits, not in any
    production path.
    """
    docs_path = Path(ROOT) / "data" / "processed" / dataset_id / "docs.jsonl"
    if not docs_path.exists():
        return ""
    target = doc_id
    with docs_path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row["id"] == target:
                return str(row["text"][:200]).replace("\n", " ")
    return ""


def smoke(
    dataset_id: str,
    queries: list[str],
    k: int,
) -> None:
    d = index_dir(dataset_id)
    if not (d / tfidf_mod.MATRIX_FILENAME).exists():
        print(f"[smoke] {dataset_id} has no TF-IDF index. Run `make build-indexes` first.")
        return

    print(f"\n========== {dataset_id} ==========")
    print(f"  index dir: {d}")

    # Load all three retrievers (LRU caching is the service's job;
    # the smoke script just loads each once).
    tfidf = tfidf_mod.TfidfRetriever.load(d)
    bm25 = bm25_mod.BM25Retriever.load(d)
    if tfidf.matrix is None:
        raise RuntimeError(f"TF-IDF matrix missing in {d}")
    if tfidf.vectorizer is None:
        raise RuntimeError(f"TF-IDF vectorizer missing in {d}")
    print(
        f"  tfidf: {len(tfidf.vectorizer.vocabulary_):,} vocab, " f"{tfidf.matrix.shape[0]:,} docs"
    )
    print(f"  bm25:  {len(bm25.vocab):,} vocab, {len(bm25.doc_ids):,} docs")

    for q_text in queries:
        print(f"\n--- query: {q_text!r} ---")
        tokens = preprocess(q_text)
        if not tokens:
            print("  (no tokens after preprocessing)")
            continue
        print(f"  tokens ({len(tokens)}): {tokens[:15]}{' ...' if len(tokens) > 15 else ''}")

        for model_name in ("bm25", "tfidf"):
            t0 = time.perf_counter()
            hits_list: list[Any]
            if model_name == "bm25":
                bm25_hits, _ = bm25.search(tokens, k=k)
                hits_list = list(bm25_hits)
            else:
                hits_list = list(tfidf.search(tokens, k=k))
            elapsed_ms = (time.perf_counter() - t0) * 1000
            print(f"\n  [{model_name}] top-{k} ({elapsed_ms:.1f} ms):")
            for h in hits_list[:k]:
                if h is None:
                    continue
                snippet = _load_doc_text(dataset_id, h.doc_id)
                print(f"    rank={h.rank} doc_id={h.doc_id} score={h.score:.4f}")
                if snippet:
                    print(f'      "{snippet}"')
                else:
                    print(f"      (no text available for {h.doc_id})")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        help=f"Datasets to test (default: {list(DATASETS)}).",
    )
    parser.add_argument(
        "--query",
        action="append",
        default=None,
        help=("Add a custom query (in addition to the default ones). " "Repeat the flag for more."),
    )
    parser.add_argument(
        "--k",
        type=int,
        default=3,
        help="Number of results to display per query per model (default 3).",
    )
    args = parser.parse_args()

    datasets = args.datasets or list(DATASETS)
    for ds in datasets:
        queries = list(DEFAULT_QUERIES.get(ds, []))
        if args.query:
            queries.extend(args.query)
        if not queries:
            print(f"[smoke] no queries defined for {ds}; pass --query")
            continue
        smoke(ds, queries, args.k)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
