"""Ingest Dataset B — ``beir/nq`` — into JSONL.

Phase 1 of the IR project. Streams BEIR Natural Questions documents from
the ``ir_datasets`` cache and writes ``{"id", "text"}`` JSONL to
``data/processed/nq/docs.jsonl``.

The full corpus has 2,681,468 documents (Wikipedia passages paired with
Google's Natural Questions). We cap at 500,000 for a fair comparison
with Dataset A and to keep BM25/FAISS indexing tractable.

Usage:
    python scripts/ingest_dataset_b.py
    python scripts/ingest_dataset_b.py --limit 100   # smoke test
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Force UTF-8 stdout/stderr on Windows.
try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except Exception:
    pass

import ir_datasets
from tqdm import tqdm

DATASET_ID = "nq"
IR_DATASET_NAME = "beir/nq"
DEFAULT_CAP = 500_000  # fair-comparison cap matching Dataset A

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "processed" / DATASET_ID
DOCS_PATH = OUT_DIR / "docs.jsonl"
META_PATH = OUT_DIR / "sample_meta.json"


def ingest(limit: int) -> int:
    """Stream NQ documents, write JSONL, return number of docs written."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[ingest] Loading ir_datasets '{IR_DATASET_NAME}' …", flush=True)
    ds = ir_datasets.load(IR_DATASET_NAME)

    written = 0
    skipped_empty = 0
    started = time.time()

    total_hint = min(limit, ds.docs_count()) if hasattr(ds, "docs_count") else limit
    with (
        DOCS_PATH.open("w", encoding="utf-8") as f,
        tqdm(total=total_hint, desc=DATASET_ID, unit="doc") as bar,
    ):
        for doc in ds.docs_iter():
            text = (doc.text or "").strip()
            if not text:
                skipped_empty += 1
                continue
            f.write(json.dumps({"id": doc.doc_id, "text": text}, ensure_ascii=False) + "\n")
            written += 1
            bar.update(1)
            if written >= limit:
                break

    elapsed = time.time() - started
    META_PATH.write_text(
        json.dumps(
            {
                "dataset": IR_DATASET_NAME,
                "dataset_id": DATASET_ID,
                "stored_docs": written,
                "skipped_empty": skipped_empty,
                "cap": limit,
                "ingested_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "elapsed_seconds": round(elapsed, 2),
                "ir_datasets_version": ir_datasets.__version__,
                "schema": {"id": "str", "text": "str"},
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_CAP,
        help=f"Maximum number of docs to write (default {DEFAULT_CAP:,})",
    )
    args = parser.parse_args()
    if args.limit < 1:
        print("--limit must be ≥ 1", file=sys.stderr)
        return 2

    print(f"[ingest] {IR_DATASET_NAME} → {DOCS_PATH}")
    print(f"[ingest] cap = {args.limit:,}")
    n = ingest(args.limit)
    print(f"[ingest] ✅ Wrote {n:,} docs to {DOCS_PATH}")
    print(f"[ingest] Metadata: {META_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
