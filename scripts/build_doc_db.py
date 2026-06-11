"""Build SQLite document databases from processed ``docs.jsonl`` files.

Reads ``data/processed/{dataset_id}/docs.jsonl`` and writes
``data/dbs/{dataset_id}.db`` with schema::

    CREATE TABLE documents (doc_id TEXT PRIMARY KEY, text TEXT)

Usage::

    py -3.12 -m scripts.build_doc_db
    py -3.12 -m scripts.build_doc_db --dataset touche2020
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from shared.ir_common.doc_db import create_db, insert_docs, db_path

DATASETS = ("touche2020", "nq")
PROCESSED_ROOT = Path(__file__).resolve().parents[1] / "data" / "processed"
BATCH_SIZE = 1000


def build(dataset_id: str) -> None:
    docs_path = PROCESSED_ROOT / dataset_id / "docs.jsonl"
    if not docs_path.exists():
        print(f"ERROR: {docs_path} not found. Run Phase 1 ingestion first.", file=sys.stderr)
        raise SystemExit(1)

    dest = db_path(dataset_id)
    t0 = time.perf_counter()
    conn = create_db(dataset_id)
    batch: list[tuple[str, str]] = []
    total = 0

    with open(docs_path, encoding="utf-8") as f:
        for line in f:
            doc = json.loads(line)
            batch.append((doc["id"], doc["text"]))
            total += 1
            if len(batch) >= BATCH_SIZE:
                insert_docs(conn, batch)
                batch.clear()
                print(f"  {dataset_id}: {total} docs inserted...")

    if batch:
        insert_docs(conn, batch)

    conn.close()
    elapsed = time.perf_counter() - t0
    size_mb = dest.stat().st_size / (1024 * 1024)
    print(f"  ✅ {dataset_id}: {total} docs → {dest} ({size_mb:.1f} MB) in {elapsed:.1f}s")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build SQLite document databases.")
    parser.add_argument(
        "--dataset",
        choices=DATASETS,
        default=None,
        help="Build only one dataset (default: both).",
    )
    args = parser.parse_args()

    targets = [args.dataset] if args.dataset else DATASETS
    for ds in targets:
        build(ds)

    print("Done.")


if __name__ == "__main__":
    main()
