"""Build the three classical indexes for one or more datasets.

Phase 2 of the IR project. Reads ``data/processed/<dataset_id>/tokens.jsonl``
(Phase 1 output) and writes:

  - inverted.pkl                -- the InvertedIndex (cap-tunable)
  - tfidf_vectorizer.pkl        -- the fitted TfidfVectorizer
  - tfidf_matrix.npz            -- the sparse TF-IDF matrix
  - doc_ids.json (TF-IDF)       -- row-index -> doc_id mapping
  - bm25.pkl                    -- the default (k1=1.5, b=0.75) BM25 instance
  - bm25_token_ids.pkl          -- corpus in token-ID form (for re-tuning)
  - bm25_vocab.json             -- term -> id mapping
  - doc_ids.json (BM25)         -- row-index -> doc_id mapping
  - build_meta.json             -- stats and timing

CLI:
    python scripts/build_indexes.py                          # build both
    python scripts/build_indexes.py --datasets touche2020
    python scripts/build_indexes.py --min-df 5 --max-df-ratio 0.3
    python scripts/build_indexes.py --no-progress            # quieter

By default we cap the InvertedIndex at ``min_df=2, max_df_ratio=0.5``
(see ``services/indexing.app.config.DEFAULT_*``). On a 16 GB machine the
uncapped dict-of-dicts is 8-10 GB and will OOM the build; the cap
brings it down to 4-5 GB.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Force UTF-8 on Windows.
try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
except Exception:
    pass

# Allow `python scripts/build_indexes.py` from the repo root.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.indexing.app import bm25 as bm25_mod  # noqa: E402
from services.indexing.app import inverted_index as inverted_index_mod  # noqa: E402
from services.indexing.app import tfidf as tfidf_mod  # noqa: E402
from services.indexing.app.config import (  # noqa: E402
    DATASETS,
    DEFAULT_MAX_DF_RATIO,
    DEFAULT_MIN_DF,
    INDEX_ROOT,
    index_dir,
)
from services.indexing.app.corpus import stream_tokens  # noqa: E402


def build_one(
    dataset_id: str,
    min_df: int,
    max_df_ratio: float,
    bm25_method: str,
    show_progress: bool,
) -> dict:
    """Build all three indexes for ``dataset_id``. Returns a stats dict."""
    out_dir = index_dir(dataset_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    print(f"\n=== {dataset_id} ===", flush=True)
    print(f"  output:  {out_dir}")
    print(f"  min_df={min_df}, max_df_ratio={max_df_ratio}, bm25_method={bm25_method}")

    # 1. Stream tokens.jsonl into memory.
    #    The build script needs the full corpus to fit sklearn's
    #    TfidfVectorizer and bm25s's BM25.peak RAM ~4 GB on 16 GB.
    t0 = time.time()
    doc_ids: list[str] = []
    corpus: list[list[str]] = []
    iter_tokens = stream_tokens(dataset_id)
    if show_progress:
        from tqdm import tqdm

        iter_tokens = tqdm(iter_tokens, desc="read tokens.jsonl", unit="doc")
    for doc_id, tokens in iter_tokens:
        doc_ids.append(doc_id)
        corpus.append(tokens)
    read_elapsed = time.time() - t0
    print(f"  [1/4] read {len(doc_ids):,} docs in {read_elapsed:.1f}s")

    # 2. InvertedIndex.
    t0 = time.time()
    idx = inverted_index_mod.InvertedIndex(min_df=min_df, max_df_ratio=max_df_ratio)
    idx.build(zip(doc_ids, corpus, strict=True))
    idx.save(out_dir / inverted_index_mod.INDEX_FILENAME)
    idx_elapsed = time.time() - t0
    print(
        f"  [2/4] inverted_index: vocab={len(idx.inverted_index):,}, "
        f"docs={idx.total_docs:,}, avg_dl={idx.avg_doc_length:.1f} "
        f"({idx_elapsed:.1f}s)"
    )

    # 3. TF-IDF.
    t0 = time.time()
    tfidf = tfidf_mod.TfidfRetriever()
    tfidf.build(corpus, doc_ids)
    tfidf.save(out_dir)
    tfidf_elapsed = time.time() - t0
    tfidf_vocab = len(tfidf.vectorizer.vocabulary_) if tfidf.vectorizer else 0
    tfidf_nnz = int(tfidf.matrix.nnz) if tfidf.matrix is not None else 0
    print(f"  [3/4] tfidf: vocab={tfidf_vocab:,}, nnz={tfidf_nnz:,} " f"({tfidf_elapsed:.1f}s)")

    # 4. BM25.
    t0 = time.time()
    bm25 = bm25_mod.BM25Retriever()
    bm25.build(
        corpus,
        doc_ids,
        method=bm25_method,
        show_progress=show_progress,
    )
    bm25.save(out_dir)
    bm25_elapsed = time.time() - t0
    default_bm = bm25._default_bm
    if default_bm is None:
        raise RuntimeError("BM25 build produced no default_bm")
    print(
        f"  [4/4] bm25: vocab={len(bm25.vocab):,}, "
        f"default k1={default_bm.k1}, b={default_bm.b} "
        f"({bm25_elapsed:.1f}s)"
    )

    # 5. Persist build_meta.json.
    total_elapsed = time.time() - started
    size_mb = 0.0
    for f in out_dir.iterdir():
        size_mb += f.stat().st_size / (1024 * 1024)
    meta = {
        "dataset_id": dataset_id,
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "elapsed_seconds": round(total_elapsed, 2),
        "total_docs": len(doc_ids),
        "inverted_vocab_post_cap": len(idx.inverted_index),
        "tfidf_vocab": tfidf_vocab,
        "tfidf_nnz": tfidf_nnz,
        "bm25_vocab": len(bm25.vocab),
        "min_df": min_df,
        "max_df_ratio": max_df_ratio,
        "bm25_method": bm25_method,
        "size_mb": round(size_mb, 1),
    }
    (out_dir / "build_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"  total: {total_elapsed:.1f}s, on-disk size: {size_mb:.1f} MB")
    return meta


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        help=f"Datasets to build (default: {list(DATASETS)}).",
    )
    parser.add_argument(
        "--min-df",
        type=int,
        default=DEFAULT_MIN_DF,
        help=f"Min doc-freq for InvertedIndex (default {DEFAULT_MIN_DF}).",
    )
    parser.add_argument(
        "--max-df-ratio",
        type=float,
        default=DEFAULT_MAX_DF_RATIO,
        help=f"Max doc-freq ratio for InvertedIndex (default {DEFAULT_MAX_DF_RATIO}).",
    )
    parser.add_argument(
        "--bm25-method",
        default="lucene",
        choices=["lucene", "atire", "robertson", "bm25l", "bm25plus"],
        help="BM25 variant (default 'lucene' = BM25Okapi equivalent).",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress bars (for CI / non-TTY).",
    )
    args = parser.parse_args()

    if args.datasets:
        ids = args.datasets
    else:
        ids = list(DATASETS)

    # Sanity: each dataset must have a tokens.jsonl.
    from services.indexing.app.config import tokens_path

    for ds in ids:
        if not tokens_path(ds).exists():
            print(f"[build] tokens.jsonl for '{ds}' not found at {tokens_path(ds)}")
            print(f"[build] run `make ingest-{ds[0]}` and `make tokenize` first.")
            return 2

    print(f"[build] Building indexes for {len(ids)} dataset(s): {ids}")
    print(f"[build] Output root: {INDEX_ROOT}")
    print(f"[build] min_df={args.min_df}, max_df_ratio={args.max_df_ratio}")
    print(f"[build] bm25_method={args.bm25_method}")

    show_progress = not args.no_progress
    all_meta: list[dict] = []
    overall_started = time.time()
    for ds in ids:
        meta = build_one(
            ds,
            min_df=args.min_df,
            max_df_ratio=args.max_df_ratio,
            bm25_method=args.bm25_method,
            show_progress=show_progress,
        )
        all_meta.append(meta)
    overall_elapsed = time.time() - overall_started

    print("\n=== Summary ===")
    for m in all_meta:
        print(
            f"  {m['dataset_id']:>12}  "
            f"docs={m['total_docs']:>9,}  "
            f"inverted_vocab={m['inverted_vocab_post_cap']:>7,}  "
            f"tfidf_vocab={m['tfidf_vocab']:>7,}  "
            f"bm25_vocab={m['bm25_vocab']:>7,}  "
            f"{m['elapsed_seconds']:>7.1f}s  "
            f"{m['size_mb']:>6.1f} MB"
        )
    print(f"  total: {overall_elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
