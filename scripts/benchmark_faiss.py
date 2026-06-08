"""Benchmark FAISS dense indexes: Flat vs IVF latency + recall.

Measures average query latency and recall@10 for both ``IndexFlatIP``
(exact) and ``IndexIVFFlat`` (approximate) on the same embeddings,
using the dataset's own test queries.

Usage
-----
    # Benchmark both datasets.
    python scripts/benchmark_faiss.py

    # Benchmark only nq with 100 samples.
    python scripts/benchmark_faiss.py --datasets nq --samples 100

    # Use a different IVF nprobe.
    python scripts/benchmark_faiss.py --nprobe 32
"""

from __future__ import annotations

import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "6")
os.environ.setdefault("MKL_NUM_THREADS", "6")

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse  # noqa: E402

import numpy as np  # noqa: E402

from services.retrieval.app.config import (  # noqa: E402
    DATASETS,
    DEFAULT_MODEL_NAME,
    EMBED_DEVICE,
    FAISS_IVF_NLIST,
    FAISS_IVF_NPROBE,
    USE_FP16,
    index_dir,
)
from services.retrieval.app.vector_store import DenseIndex  # noqa: E402

IR_DATASET_MAP = {
    "touche2020": "beir/webis-touche2020",
    "nq": "beir/nq",
}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Benchmark FAISS Flat vs IVF latency and recall."
    )
    p.add_argument(
        "--datasets",
        nargs="+",
        choices=list(DATASETS),
        default=list(DATASETS),
        help="Datasets to benchmark (default: all).",
    )
    p.add_argument(
        "--samples",
        type=int,
        default=0,
        help="Max test queries per dataset (0 = all).",
    )
    p.add_argument(
        "--nprobe",
        type=int,
        default=FAISS_IVF_NPROBE,
        help=f"IVF nprobe (default: {FAISS_IVF_NPROBE}).",
    )
    p.add_argument(
        "--nlist",
        type=int,
        default=FAISS_IVF_NLIST,
        help=f"IVF nlist (default: {FAISS_IVF_NLIST}).",
    )
    p.add_argument(
        "--k",
        type=int,
        default=10,
        help="Top-k for ranking (default: 10).",
    )
    p.add_argument(
        "--warmup",
        type=int,
        default=3,
        help="Warmup queries before timing (default: 3).",
    )
    return p.parse_args()


def _load_test_queries(dataset_id: str, max_samples: int) -> list[str]:
    import ir_datasets

    ir_id = IR_DATASET_MAP[dataset_id]
    ds = ir_datasets.load(ir_id)
    queries = []
    for q in ds.queries_iter():
        queries.append(str(q.text))
        if max_samples and len(queries) >= max_samples:
            break
    if not queries:
        print(f"  WARNING: no test queries found for {dataset_id}")
    return queries


def _load_embedder(model_name: str) -> object:
    from services.retrieval.app.embedder import Embedder

    emb = Embedder()
    emb.warm_up(model_name)
    return emb


def _encode_queries(
    embedder: object, queries: list[str], model_name: str
) -> np.ndarray:
    """Encode all queries and return a single (N, dim) float32 array."""
    all_vecs = []
    batch = 64
    for i in range(0, len(queries), batch):
        batch_q = queries[i : i + batch]
        vecs = embedder.encode(batch_q, model_name=model_name, show_progress=False)  # type: ignore[union-attr]
        all_vecs.append(vecs)
    return np.concatenate(all_vecs, axis=0).astype(np.float32)


def _build_ivf_index(flat_idx: DenseIndex, nlist: int, nprobe: int) -> DenseIndex:
    """Build a fresh IndexIVFFlat from the same vectors as *flat_idx*."""
    import faiss

    vectors = flat_idx.vectors
    if vectors is None:
        raise RuntimeError("flat_idx has no vectors loaded")

    dim = int(vectors.shape[1])
    quantizer = faiss.IndexFlatIP(dim)
    ivf = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
    n_train = min(vectors.shape[0], nlist * 30)
    ivf.train(np.ascontiguousarray(vectors[:n_train]))
    ivf.add(np.ascontiguousarray(vectors))
    ivf.nprobe = nprobe

    result = DenseIndex()
    result.vectors = vectors
    result.doc_ids = flat_idx.doc_ids
    result._index = ivf  # type: ignore[assignment]
    return result


def _run_benchmark(
    dataset_id: str,
    flat_idx: DenseIndex,
    ivf_idx: DenseIndex | None,
    queries: list[str],
    query_vectors: np.ndarray,
    k: int,
    warmup: int,
) -> dict[str, object]:
    """Run benchmark and return a stats dict."""

    def measure_latency(
        idx: DenseIndex, label: str
    ) -> tuple[float, float, float, float]:
        latencies: list[float] = []
        for _i, qv in enumerate(query_vectors):
            q = qv.copy()
            t0 = time.perf_counter()
            idx.search(q, k=k)
            latencies.append((time.perf_counter() - t0) * 1000)
        # Skip warmup from stats.
        timed = sorted(latencies[warmup:]) if warmup < len(latencies) else sorted(latencies)
        if not timed:
            return 0.0, 0.0, 0.0, 0.0
        avg = float(np.mean(timed))
        p50 = float(np.median(timed))
        p95 = timed[int(len(timed) * 0.95)]
        p99 = timed[int(len(timed) * 0.99)]
        return avg, p50, p95, p99

    def compute_recall(
        flat_results: list[list[int]],
        ivf_results: list[list[int]],
        k_val: int,
    ) -> float:
        """Recall@k: fraction of Flat top-k docs found in IVF top-k."""
        hits = 0
        total = 0
        for flat_ids, ivf_ids in zip(flat_results, ivf_results, strict=True):
            flat_set = set(int(x) for x in flat_ids[:k_val])
            ivf_set = set(int(x) for x in ivf_ids[:k_val])
            hits += len(flat_set & ivf_set)
            total += min(k_val, len(flat_set))
        return hits / total if total else 1.0

    # Flat results (ground truth).
    flat_results: list[list[int]] = []
    for qv in query_vectors:
        _, ids = flat_idx.search(qv.copy(), k=k)
        flat_results.append(ids.tolist())

    flat_avg, flat_p50, flat_p95, flat_p99 = measure_latency(flat_idx, "Flat")

    stats: dict[str, object] = {
        "dataset_id": dataset_id,
        "num_queries": len(queries),
        "num_vectors": flat_idx.size(),
        "embedding_dim": flat_idx.dim(),
        "index_flat": "IndexFlatIP",
        "flat_latency_ms_avg": round(flat_avg, 2),
        "flat_latency_ms_p50": round(flat_p50, 2),
        "flat_latency_ms_p95": round(flat_p95, 2),
        "flat_latency_ms_p99": round(flat_p99, 2),
    }

    if ivf_idx is not None:
        ivf_results: list[list[int]] = []
        for qv in query_vectors:
            _, ids = ivf_idx.search(qv.copy(), k=k)
            ivf_results.append(ids.tolist())

        ivf_avg, ivf_p50, ivf_p95, ivf_p99 = measure_latency(ivf_idx, "IVF")
        recall = compute_recall(flat_results, ivf_results, k)

        stats.update({
            "index_ivf": "IndexIVFFlat",
            "ivf_nlist": FAISS_IVF_NLIST,
            "ivf_nprobe": FAISS_IVF_NPROBE,
            "ivf_latency_ms_avg": round(ivf_avg, 2),
            "ivf_latency_ms_p50": round(ivf_p50, 2),
            "ivf_latency_ms_p95": round(ivf_p95, 2),
            "ivf_latency_ms_p99": round(ivf_p99, 2),
            f"recall_at_{k}": round(recall, 4),
            "latency_speedup": round(flat_avg / ivf_avg, 2) if ivf_avg > 0 else float("inf"),
        })

    return stats


def main() -> int:
    args = _parse_args()
    model_name = DEFAULT_MODEL_NAME
    print(f"Device: {EMBED_DEVICE} (fp16={USE_FP16})")
    print(f"Model:  {model_name}")
    print()

    embedder = _load_embedder(model_name)
    all_stats: list[dict[str, object]] = []

    for ds in args.datasets:
        print(f"{'='*60}")
        print(f"  Dataset: {ds}")
        print(f"{'='*60}")

        indir = index_dir(ds)
        if not (indir / "faiss.index").exists():
            print(f"  SKIP: FAISS index not found at {indir / 'faiss.index'}")
            print()
            continue

        print(f"  Loading Flat index from {indir}...")
        flat_idx = DenseIndex.load(indir)
        print(f"    vectors={flat_idx.size():,}, dim={flat_idx.dim()}")

        print(f"  Building IVF index (nlist={args.nlist}, nprobe={args.nprobe})...")
        t0 = time.perf_counter()
        ivf_idx = _build_ivf_index(flat_idx, args.nlist, args.nprobe)
        build_s = time.perf_counter() - t0
        print(f"    built in {build_s:.1f}s")

        print("  Loading test queries...")
        queries = _load_test_queries(ds, args.samples)
        if not queries:
            print(f"  SKIP: no queries for {ds}")
            print()
            continue
        print(f"    {len(queries)} queries loaded")

        print("  Encoding queries...")
        t0 = time.perf_counter()
        query_vectors = _encode_queries(embedder, queries, model_name)
        encode_s = time.perf_counter() - t0
        print(f"    encoded in {encode_s:.1f}s ({encode_s/len(queries):.2f}s/query)")

        print(f"  Running benchmark (k={args.k}, warmup={args.warmup})...")
        stats = _run_benchmark(
            ds, flat_idx, ivf_idx, queries, query_vectors, args.k, args.warmup
        )
        all_stats.append(stats)

        print()
        print("  ── Results ──")
        flat_avg = stats["flat_latency_ms_avg"]
        flat_p50 = stats["flat_latency_ms_p50"]
        flat_p95 = stats["flat_latency_ms_p95"]
        print(f"  Flat  : avg={flat_avg:>8.2f} ms  p50={flat_p50:>8.2f} ms  p95={flat_p95:>8.2f} ms")
        ivf_avg = stats.get("ivf_latency_ms_avg", 0.0)
        ivf_p50 = stats.get("ivf_latency_ms_p50", 0.0)
        ivf_p95 = stats.get("ivf_latency_ms_p95", 0.0)
        recall_key = f"recall_at_{args.k}"
        recall = stats.get(recall_key, 1.0)
        speedup = stats.get("latency_speedup", 1.0)
        print(f"  IVF   : avg={ivf_avg:>8.2f} ms  p50={ivf_p50:>8.2f} ms  p95={ivf_p95:>8.2f} ms")
        print(f"  Recall@{args.k}: {recall:.4f}   Speedup: {speedup:.2f}x")
        print()

    print(f"{'='*60}")
    print("  Summary")
    print(f"{'='*60}")
    for s in all_stats:
        ds = s["dataset_id"]
        flat_avg = s["flat_latency_ms_avg"]
        ivf_avg = s.get("ivf_latency_ms_avg", 0.0)
        recall = s.get(f"recall_at_{args.k}", 1.0)
        speedup = s.get("latency_speedup", 1.0)
        print(f"  {ds:12s}  Flat {flat_avg:>8.2f} ms  IVF {ivf_avg:>8.2f} ms  "
              f"Recall@{args.k}={recall:.4f}  {speedup:.2f}x")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
