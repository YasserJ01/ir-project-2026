"""Ingest Dataset A — ``beir/webis-touche2020`` — into JSONL.

Phase 1 of the IR project. Streams documents from the ``ir_datasets`` cache
(``~/.ir_datasets/``) and writes one ``{"id", "text"}`` per line to
``data/processed/touche2020/docs.jsonl``.

The full corpus has 382,545 docs from the BEIR Webis-Touche 2020 collection
(argument retrieval, debate topics). We cap at 500K — but the full corpus
fits inside the cap, so this just means "take all of it".

Usage:
    python scripts/ingest_dataset_a.py                # full corpus
    python scripts/ingest_dataset_a.py --limit 100    # smoke test
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Force UTF-8 stdout/stderr on Windows so progress output (and the
# em-dash/arrow chars below) survive PowerShell's default cp1252 codec.
try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except Exception:
    pass

import ir_datasets
from tqdm import tqdm

DATASET_ID = "touche2020"
IR_DATASET_NAME = "beir/webis-touche2020"
DEFAULT_CAP = 500_000  # cap > corpus size, so effectively the full corpus

# Project root = parent of this script's parent
ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "processed" / DATASET_ID
DOCS_PATH = OUT_DIR / "docs.jsonl"
META_PATH = OUT_DIR / "sample_meta.json"


def ingest(limit: int) -> int:
    """Stream documents, write JSONL, return number of docs written."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[ingest] Loading ir_datasets '{IR_DATASET_NAME}' …", flush=True)
    ds = ir_datasets.load(IR_DATASET_NAME)

    written = 0
    skipped_empty = 0
    started = time.time()

    with (
        DOCS_PATH.open("w", encoding="utf-8") as f,
        tqdm(total=min(limit, ds.docs_count()), desc=DATASET_ID, unit="doc") as bar,
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
