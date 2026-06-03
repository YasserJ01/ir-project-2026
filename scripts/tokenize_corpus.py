"""Tokenize every ``docs.jsonl`` under ``data/processed/`` → ``tokens.jsonl``.

For each dataset, reads ``docs.jsonl`` line by line, runs the shared
``preprocess()`` function, and writes ``tokens.jsonl`` with
``{"id": ..., "tokens": [...]}``. Reports per-dataset statistics.

This is the persistent tokenized corpus that Phase 2 (BM25/TF-IDF) and
Phase 3 (embeddings) consume. Keeping it on disk means we tokenize once
and re-use for many experiments.

By default we use a ``multiprocessing.Pool`` with one worker per CPU
core (cap 8). The shared NLTK resources are loaded inside each worker
the first time it is used; output order is preserved via imap.

Usage:
    python scripts/tokenize_corpus.py
    python scripts/tokenize_corpus.py --datasets msmarco_passage
    python scripts/tokenize_corpus.py --workers 1   # debug
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import statistics
import sys
import time
from pathlib import Path

# Force UTF-8 stdout/stderr on Windows.
try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
except Exception:
    pass

from tqdm import tqdm

from shared.ir_common.preprocess import preprocess

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data" / "processed"


def _tokenize_line(line: str) -> tuple[str, list[str]]:
    """Worker entry: parse one JSONL line, preprocess text, return (id, tokens)."""
    row = json.loads(line)
    return row["id"], preprocess(row["text"])


def _tokenize_chunk(chunk: list[str]) -> list[tuple[str, list[str]]]:
    """Process a list of lines in a single worker (better IPC amortization)."""
    return [_tokenize_line(line) for line in chunk]


def _chunkify(items: list[str], chunk_size: int) -> list[list[str]]:
    return [items[i : i + chunk_size] for i in range(0, len(items), chunk_size)]


def tokenize_one(dataset_id: str, workers: int) -> dict:
    docs_path = DATA_ROOT / dataset_id / "docs.jsonl"
    if not docs_path.exists():
        print(f"[tokenize] {docs_path} not found - run ingest first.", file=sys.stderr)
        return {"dataset_id": dataset_id, "status": "missing"}

    out_path = docs_path.parent / "tokens.jsonl"
    print(f"[tokenize] {docs_path} -> {out_path}")
    print(f"[tokenize] workers = {workers}")

    # Read all lines once (the source files fit in RAM; ~1 GB total).
    started = time.time()
    with docs_path.open("r", encoding="utf-8") as src:
        lines = src.readlines()
    print(f"[tokenize] read {len(lines):,} lines in {time.time()-started:.1f}s")

    counts: list[int] = []
    started = time.time()
    with out_path.open("w", encoding="utf-8") as dst:
        if workers <= 1:
            # Serial fallback (simpler, easier to debug).
            for line in tqdm(lines, desc=dataset_id, unit="doc"):
                doc_id, tokens = _tokenize_line(line)
                dst.write(json.dumps({"id": doc_id, "tokens": tokens}, ensure_ascii=False) + "\n")
                counts.append(len(tokens))
        else:
            # Parallel: chunk lines, imap_unordered for throughput, but write
            # back in original order by keeping a positional index per chunk.
            chunk_size = max(200, len(lines) // (workers * 50))
            chunks = _chunkify(lines, chunk_size)
            with mp.Pool(processes=workers) as pool:
                for result in tqdm(
                    pool.imap(_tokenize_chunk, chunks),
                    total=len(chunks),
                    desc=f"{dataset_id} ({workers}w)",
                    unit="chunk",
                ):
                    for doc_id, tokens in result:
                        dst.write(
                            json.dumps({"id": doc_id, "tokens": tokens}, ensure_ascii=False) + "\n"
                        )
                        counts.append(len(tokens))

    elapsed = time.time() - started
    total_tokens = sum(counts)
    stats = {
        "dataset_id": dataset_id,
        "status": "ok",
        "docs": len(counts),
        "total_tokens": total_tokens,
        "mean_tokens_per_doc": round(statistics.fmean(counts), 2),
        "median_tokens_per_doc": int(statistics.median(counts)),
        "min_tokens_per_doc": min(counts),
        "max_tokens_per_doc": max(counts),
        "elapsed_seconds": round(elapsed, 2),
        "tokens_per_sec": int(total_tokens / elapsed) if elapsed > 0 else 0,
        "workers": workers,
    }
    meta = docs_path.parent / "tokenize_meta.json"
    meta.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return stats


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        help="Dataset ids. Default: every subdir of data/processed with a docs.jsonl.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(8, os.cpu_count() or 1),
        help="Number of worker processes (default: min(8, cpu_count))",
    )
    args = parser.parse_args()

    if args.datasets:
        ids = args.datasets
    else:
        if not DATA_ROOT.exists():
            print(f"[tokenize] {DATA_ROOT} not found.", file=sys.stderr)
            return 2
        ids = sorted(p.name for p in DATA_ROOT.iterdir() if (p / "docs.jsonl").exists())
    if not ids:
        print("[tokenize] No docs.jsonl files found.", file=sys.stderr)
        return 2

    print(f"[tokenize] Tokenizing {len(ids)} dataset(s): {ids}\n")
    results = [tokenize_one(ds_id, args.workers) for ds_id in ids]

    print("\n[tokenize] === Summary ===")
    for r in results:
        if r.get("status") != "ok":
            print(f"  {r['dataset_id']}: {r['status']}")
            continue
        print(
            f"  {r['dataset_id']}: {r['docs']:,} docs, "
            f"{r['total_tokens']:,} tokens, "
            f"mean {r['mean_tokens_per_doc']} / doc, "
            f"{r['tokens_per_sec']:,} tok/s ({r['workers']} workers, {r['elapsed_seconds']}s)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
