"""Build the 2nd-encoder (L12) FAISS indexes for the IR project (Phase 5).

This is the L12 analog of ``build_dense_indexes.py``. Both datasets
(L12 indexes) are encoded with ``all-MiniLM-L12-v2`` and written to:

  * ``data/indexes/<ds>/faiss_l12.index``     -- FAISS IndexFlatIP, 384-dim
  * ``data/indexes/<ds>/embeddings_l12.npy``  -- raw float32 vectors
  * ``data/indexes/<ds>/doc_ids.json``        -- SHARED with the L6 index
  * ``data/indexes/<ds>/build_meta_l12.json`` -- per-dataset build metadata

The L6 files (``faiss.index``, ``embeddings.npy``, ``build_meta.json``)
are NOT touched. The two indexes share the corpus order, so the
doc_ids.json can be shared.

This script runs synchronously (no BackgroundTasks) and prints per-step
progress. Use ``launch_dense_2.py`` to start it as a detached
subprocess (so a 3+ hour build survives the opencode shell tool's
120 s timeout).

Examples
--------
    # Build both datasets with L12.
    python scripts/build_dense_2.py

    # Build just one dataset, force rebuild even if present.
    python scripts/build_dense_2.py --datasets nq --force

    # Encode only the first 50,000 docs of each dataset (smoke test).
    python scripts/build_dense_2.py --max-docs 50000
"""

from __future__ import annotations

# Set torch threads BEFORE any heavy import. On Windows the default is
# often 1, which makes CPU encoding ~10x slower than it needs to be.
import os

os.environ.setdefault("OMP_NUM_THREADS", "6")
os.environ.setdefault("MKL_NUM_THREADS", "6")

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

# Allow `python scripts/build_dense_2.py` from the repo root.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.retrieval.app import embedder as embedder_mod  # noqa: E402
from services.retrieval.app import vector_store as vector_store_mod  # noqa: E402
from services.retrieval.app.config import (  # noqa: E402
    DATASETS,
    DEFAULT_BATCH_SIZE,
    DEFAULT_BATCH_SIZE_GPU,
    EMBED_DEVICE,
    SECOND_ENCODER_EMBEDDINGS_FILENAME,
    SECOND_ENCODER_INDEX_FILENAME,
    SECOND_ENCODER_NAME,
    USE_FP16,
    docs_path,
    index_dir,
)
from shared.ir_common.schemas import DATASET_IDS  # noqa: E402


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Build the 2nd-encoder (L12) FAISS indexes. " f"Default model: {SECOND_ENCODER_NAME}."
        )
    )
    p.add_argument(
        "--datasets",
        nargs="+",
        choices=DATASET_IDS,
        default=list(DATASETS),
        help="Which datasets to build (default: all).",
    )
    p.add_argument(
        "--model",
        default=SECOND_ENCODER_NAME,
        help=f"Hugging Face model name (default: {SECOND_ENCODER_NAME}).",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE_GPU if EMBED_DEVICE == "cuda" else DEFAULT_BATCH_SIZE,
        help=(
            f"Encode batch size. Default: {DEFAULT_BATCH_SIZE_GPU} on GPU, "
            f"{DEFAULT_BATCH_SIZE} on CPU. Override with this flag."
        ),
    )
    p.add_argument(
        "--no-progress",
        action="store_true",
        help="Silence the sentence-transformers progress bar.",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Rebuild even if the L12 FAISS index already exists.",
    )
    p.add_argument(
        "--max-docs",
        type=int,
        default=0,
        help=(
            "Keep only the first N docs of each dataset (0 = all). "
            "Useful for fast end-to-end smoke tests on CPU-only machines."
        ),
    )
    return p.parse_args()


def _load_docs(dataset_id: str, max_docs: int = 0) -> tuple[list[str], list[str]]:
    """Stream ``docs.jsonl`` into two parallel lists.

    Same char-cap heuristic as build_dense_indexes.py -- MiniLM truncates
    to 256 WordPiece tokens at encode time, so we pre-truncate to
    ~1024 chars per doc to skip the tokeniser overhead on the long
    tail of long documents.
    """
    p = docs_path(dataset_id)
    if not p.exists():
        raise FileNotFoundError(
            f"docs.jsonl for '{dataset_id}' not found at {p}. " "Run `make ingest-{a,b}` first."
        )
    char_cap = int(os.environ.get("IR_BUILD_CHAR_CAP", "1024"))
    doc_ids: list[str] = []
    texts: list[str] = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            if max_docs and len(doc_ids) >= max_docs:
                break
            row = json.loads(line)
            doc_ids.append(str(row["id"]))
            t = str(row["text"])
            if len(t) > char_cap:
                t = t[:char_cap]
            texts.append(t)
    if not texts:
        raise ValueError(f"No documents found in {p}")
    return doc_ids, texts


def _build_one(
    dataset_id: str,
    model_name: str,
    batch_size: int,
    show_progress: bool,
    force: bool,
    max_docs: int,
) -> dict[str, object]:
    """Build the L12 dense index for a single dataset. Returns a metadata dict."""
    d = index_dir(dataset_id)
    faiss_path = d / SECOND_ENCODER_INDEX_FILENAME
    meta_path = d / "build_meta_l12.json"
    if faiss_path.exists() and not force:
        print(
            f"[build2] {dataset_id}: L12 FAISS index already exists at {faiss_path}. "
            "Use --force to rebuild.",
            flush=True,
        )
        if meta_path.exists():
            return dict[str, object](json.loads(meta_path.read_text(encoding="utf-8")))
        return {"dataset_id": dataset_id, "status": "exists"}

    started = time.time()
    print(f"\n=== {dataset_id} (L12) ===", flush=True)
    print(f"  output:  {d}", flush=True)
    print(f"  model:   {model_name}", flush=True)
    print(f"  batch:   {batch_size}", flush=True)
    if max_docs:
        print(f"  cap:     first {max_docs:,} docs only", flush=True)
    d.mkdir(parents=True, exist_ok=True)

    # 1. Load docs.
    t0 = time.time()
    doc_ids, texts = _load_docs(dataset_id, max_docs=max_docs)
    load_seconds = time.time() - t0
    print(f"  [1/4] load docs: {len(texts):,} docs ({load_seconds:.1f}s)", flush=True)

    # 2. Warm up the encoder.
    t0 = time.time()
    emb = embedder_mod.Embedder(default_model_name=model_name, batch_size=batch_size)
    emb.warm_up(model_name)
    warm_seconds = time.time() - t0
    print(
        f"  [2/4] warm up model: {warm_seconds:.1f}s "
        f"(device={emb.device}, fp16={emb.use_fp16})",
        flush=True,
    )

    # 3. Encode.
    t0 = time.time()
    vectors = emb.encode_documents(texts, model_name=model_name, show_progress=not show_progress)
    encode_seconds = time.time() - t0
    docs_per_sec = len(texts) / encode_seconds if encode_seconds > 0 else 0.0
    print(
        f"  [3/4] encode: {vectors.shape[0]:,} vectors x {vectors.shape[1]}-dim, "
        f"{encode_seconds:.1f}s ({docs_per_sec:,.0f} docs/s)",
        flush=True,
    )

    # 4. Build FAISS + save under L12 filenames.
    t0 = time.time()
    idx = vector_store_mod.DenseIndex()
    idx.add(vectors, doc_ids)
    idx.save(
        d,
        index_filename=SECOND_ENCODER_INDEX_FILENAME,
        embeddings_filename=SECOND_ENCODER_EMBEDDINGS_FILENAME,
    )
    save_seconds = time.time() - t0
    elapsed = time.time() - started

    # 5. Persist build_meta_l12.json.
    size_mb = 0.0
    for name in (
        SECOND_ENCODER_INDEX_FILENAME,
        SECOND_ENCODER_EMBEDDINGS_FILENAME,
        vector_store_mod.DOC_IDS_FILENAME,
    ):
        p = d / name
        if p.exists():
            size_mb += p.stat().st_size / (1024 * 1024)

    meta: dict[str, object] = {
        "dataset_id": dataset_id,
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "status": "ok",
        "model_name": model_name,
        "index_type": "IndexFlatIP",
        "num_vectors": int(vectors.shape[0]),
        "embedding_dim": int(vectors.shape[1]),
        "elapsed_seconds": round(elapsed, 2),
        "load_seconds": round(load_seconds, 2),
        "warm_seconds": round(warm_seconds, 2),
        "encode_seconds": round(encode_seconds, 2),
        "save_seconds": round(save_seconds, 2),
        "docs_per_sec": round(docs_per_sec, 1),
        "batch_size": batch_size,
        "max_docs": max_docs,
        "char_cap": int(os.environ.get("IR_BUILD_CHAR_CAP", "1024")),
        "size_mb": round(size_mb, 1),
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(
        f"  [4/4] save faiss_l12 + embeddings_l12 + meta: "
        f"{size_mb:.1f} MB on disk ({save_seconds:.1f}s)",
        flush=True,
    )
    print(f"  total: {elapsed:.1f}s", flush=True)
    return meta


def main() -> int:
    args = _parse_args()
    print(
        f"[build2] Building L12 dense indexes for {len(args.datasets)} dataset(s): "
        f"{args.datasets}",
        flush=True,
    )
    print(f"[build2] Model: {args.model}", flush=True)
    print(f"[build2] Device: {EMBED_DEVICE} (fp16={USE_FP16})", flush=True)
    print(f"[build2] Batch size: {args.batch_size}", flush=True)
    started = time.time()
    summaries: list[dict[str, object]] = []
    for ds in args.datasets:
        try:
            meta = _build_one(
                dataset_id=ds,
                model_name=args.model,
                batch_size=args.batch_size,
                show_progress=not args.no_progress,
                force=args.force,
                max_docs=args.max_docs,
            )
            summaries.append(meta)
        except Exception as exc:  # noqa: BLE001
            print(f"[build2] {ds} FAILED: {exc}", flush=True)
            summaries.append(
                {
                    "dataset_id": ds,
                    "status": "error",
                    "error": str(exc),
                    "elapsed_seconds": 0.0,
                }
            )
    total = time.time() - started
    print("\n=== Summary (L12) ===", flush=True)
    for s in summaries:
        ds = s.get("dataset_id", "?")
        n = s.get("num_vectors", 0)
        d_ = s.get("embedding_dim", 0)
        sec = s.get("elapsed_seconds", 0.0)
        mb = s.get("size_mb", 0.0)
        print(
            f"    {ds:12s}  vectors={n:>9,}  dim={d_:>3}  " f"{sec:>7.1f}s   {mb:>7.1f} MB",
            flush=True,
        )
    print(f"  total: {total:.1f}s", flush=True)
    if all(s.get("status") == "error" for s in summaries):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
