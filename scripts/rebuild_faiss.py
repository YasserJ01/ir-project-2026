"""Rebuild FAISS dense indexes for one or both datasets.

This is the "Vector Store hardening" entry point (Phase 8, §8.1).
It wraps :mod:`scripts.build_dense_indexes` and adds convenience
flags:

* ``--force`` is **on by default** (no need to pass it).
* ``--ivf`` builds ``IndexIVFFlat`` instead of ``IndexFlatIP``.

Examples
--------
    # Rebuild both datasets with default settings.
    python scripts/rebuild_faiss.py

    # Rebuild only nq with IndexIVFFlat.
    python scripts/rebuild_faiss.py --datasets nq --ivf
"""

from __future__ import annotations

import os
import sys

# Set torch threads BEFORE any heavy import.
os.environ.setdefault("OMP_NUM_THREADS", "6")
os.environ.setdefault("MKL_NUM_THREADS", "6")

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse  # noqa: E402

from shared.ir_common.schemas import DATASET_IDS  # noqa: E402


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Rebuild FAISS dense indexes (Vector Store hardening)."
    )
    p.add_argument(
        "--datasets",
        nargs="+",
        choices=DATASET_IDS,
        default=list(DATASET_IDS),
        help="Which datasets to build (default: all).",
    )
    p.add_argument(
        "--model",
        default=None,
        help="Sentence-transformers model name (default: from config).",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Encode batch size (default: auto GPU/CPU).",
    )
    p.add_argument(
        "--ivf",
        action="store_true",
        help="Build IndexIVFFlat instead of IndexFlatIP.",
    )
    p.add_argument(
        "--nlist",
        type=int,
        default=4096,
        help="IVF centroid count (default: 4096, only used with --ivf).",
    )
    p.add_argument(
        "--max-docs",
        type=int,
        default=0,
        help="Cap at first N docs per dataset (0 = all).",
    )
    p.add_argument(
        "--no-progress",
        action="store_true",
        help="Silence the sentence-transformers progress bar.",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()

    if args.ivf:
        os.environ["FAISS_INDEX_TYPE"] = "IndexIVFFlat"
        if args.nlist:
            os.environ["FAISS_IVF_NLIST"] = str(args.nlist)

    from scripts.build_dense_indexes import _build_one

    summaries = []
    for ds in args.datasets:
        model = args.model or "sentence-transformers/all-MiniLM-L6-v2"
        batch = args.batch_size or 0  # 0 = let config decide

        if batch == 0:
            from services.retrieval.app.config import (
                DEFAULT_BATCH_SIZE,
                DEFAULT_BATCH_SIZE_GPU,
                EMBED_DEVICE,
            )

            batch = DEFAULT_BATCH_SIZE_GPU if EMBED_DEVICE == "cuda" else DEFAULT_BATCH_SIZE

        print(f"\n=== Rebuilding {ds} (FAISS_INDEX_TYPE={os.environ.get('FAISS_INDEX_TYPE', 'IndexFlatIP')}) ===")
        try:
            meta = _build_one(
                dataset_id=ds,
                model_name=model,
                batch_size=batch,
                show_progress=not args.no_progress,
                force=True,
                max_docs=args.max_docs,
            )
            summaries.append(meta)
        except Exception as exc:
            print(f"[rebuild] {ds} FAILED: {exc}")
            summaries.append({"dataset_id": ds, "status": "error", "error": str(exc)})

    print("\n=== Rebuild Summary ===")
    for s in summaries:
        status = s.get("status", "?")
        ds = s.get("dataset_id", "?")
        n = s.get("num_vectors", 0)
        sec = s.get("elapsed_seconds", 0.0)
        mb = s.get("size_mb", 0.0)
        if status == "ok":
            print(f"  {ds:12s}  OK  vectors={n:>9,}  {sec:>7.1f}s  {mb:>7.1f} MB")
        else:
            err = s.get("error", status)
            print(f"  {ds:12s}  FAIL  {err}")

    if all(s.get("status") == "error" for s in summaries):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
