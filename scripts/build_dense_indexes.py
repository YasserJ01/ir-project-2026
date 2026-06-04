"""Build FAISS dense indexes for one or more datasets.

CLI wrapper around :class:`services.retrieval.app.service._do_build`
that runs synchronously (no BackgroundTasks) and writes a per-step
progress log. Use this when you want to build indexes from the
command line, e.g. in CI.

Examples
--------
    # Build both datasets with the default model + batch size.
    python scripts/build_dense_indexes.py

    # Build just nq with a custom model and larger batch.
    python scripts/build_dense_indexes.py --datasets nq \\
        --model sentence-transformers/all-mpnet-base-v2 \\
        --batch-size 128

    # Build silently (no progress bar).
    python scripts/build_dense_indexes.py --no-progress
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

# Allow `python scripts/build_dense_indexes.py` from the repo root.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.retrieval.app import embedder as embedder_mod  # noqa: E402
from services.retrieval.app import vector_store as vector_store_mod  # noqa: E402
from services.retrieval.app.config import (  # noqa: E402
    DATASETS,
    DEFAULT_BATCH_SIZE,
    DEFAULT_BATCH_SIZE_GPU,
    DEFAULT_MODEL_NAME,
    EMBED_DEVICE,
    USE_FP16,
    docs_path,
    index_dir,
)
from shared.ir_common.schemas import DATASET_IDS  # noqa: E402


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build FAISS dense indexes for the IR project.")
    p.add_argument(
        "--datasets",
        nargs="+",
        choices=DATASET_IDS,
        default=list(DATASETS),
        help="Which datasets to build (default: all).",
    )
    p.add_argument(
        "--model",
        default=DEFAULT_MODEL_NAME,
        help=f"Hugging Face model name (default: {DEFAULT_MODEL_NAME}).",
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
        help="Rebuild even if the FAISS index already exists.",
    )
    p.add_argument(
        "--max-docs",
        type=int,
        default=0,
        help=(
            "Keep only the first N docs of each dataset (0 = all). "
            "Useful for fast end-to-end smoke tests on CPU-only machines "
            "where encoding 882K docs takes 5+ hours. Try 50000 for a "
            "~40 min build that still gives meaningful Phase 9 numbers."
        ),
    )
    return p.parse_args()


def _load_docs(dataset_id: str, max_docs: int = 0) -> tuple[list[str], list[str]]:
    """Stream ``docs.jsonl`` into two parallel lists.

    The encoder (MiniLM-L6-v2) truncates to 256 WordPiece tokens at
    encode time; longer inputs are silently clipped to the head. We
    pre-truncate the text here to roughly the same character budget
    (256 tokens * ~4 chars/token ≈ 1024 chars) to skip the tokeniser
    overhead on the long tail of long documents. Quality is identical
    because MiniLM would have truncated anyway.

    ``max_docs > 0`` keeps only the first ``max_docs`` rows. Useful
    for fast end-to-end smoke tests on a 12-core / no-GPU box where
    encoding 882K docs is a 5+ hour job. Default 0 = no cap.
    """
    p = docs_path(dataset_id)
    if not p.exists():
        raise FileNotFoundError(
            f"docs.jsonl for '{dataset_id}' not found at {p}. " "Run `make ingest-{a,b}` first."
        )
    # ~256 WordPiece tokens at ~4 chars/token (English) is a safe cap.
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
    """Build the dense index for a single dataset. Returns a metadata dict."""
    d = index_dir(dataset_id)
    faiss_path = d / vector_store_mod.INDEX_FILENAME
    if faiss_path.exists() and not force:
        print(
            f"[build] {dataset_id}: FAISS index already exists at {faiss_path}. "
            "Use --force to rebuild."
        )
        # Read existing build_meta.json so the caller still gets a summary.
        meta_path = d / "build_meta.json"
        if meta_path.exists():
            return dict[str, object](json.loads(meta_path.read_text(encoding="utf-8")))
        return {"dataset_id": dataset_id, "status": "exists"}

    started = time.time()
    print(f"\n=== {dataset_id} ===", flush=True)
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
    print(
        f"  [1/4] load docs: {len(texts):,} docs ({load_seconds:.1f}s)",
        flush=True,
    )

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

    # 4. Build FAISS + save.
    t0 = time.time()
    idx = vector_store_mod.DenseIndex()
    idx.add(vectors, doc_ids)
    idx.save(d)
    save_seconds = time.time() - t0
    elapsed = time.time() - started

    # 5. Persist build_meta.json.
    size_mb = 0.0
    for name in (
        vector_store_mod.INDEX_FILENAME,
        vector_store_mod.EMBEDDINGS_FILENAME,
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
    (d / "build_meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(
        f"  [4/4] save faiss + npy: {size_mb:.1f} MB on disk ({save_seconds:.1f}s)",
        flush=True,
    )
    print(f"  total: {elapsed:.1f}s", flush=True)
    return meta


def main() -> int:
    args = _parse_args()
    print(
        f"[build] Building dense indexes for {len(args.datasets)} dataset(s): " f"{args.datasets}",
        flush=True,
    )
    print(f"[build] Model: {args.model}", flush=True)
    print(f"[build] Device: {EMBED_DEVICE} (fp16={USE_FP16})", flush=True)
    print(f"[build] Batch size: {args.batch_size}", flush=True)
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
            print(f"[build] {ds} FAILED: {exc}", flush=True)
            summaries.append(
                {
                    "dataset_id": ds,
                    "status": "error",
                    "error": str(exc),
                    "elapsed_seconds": 0.0,
                }
            )
    total = time.time() - started
    print("\n=== Summary ===", flush=True)
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
    # Exit 0 unless every build errored.
    if all(s.get("status") == "error" for s in summaries):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
