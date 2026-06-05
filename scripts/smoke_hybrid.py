"""Smoke test the Phase 5 hybrid / multi-encoder endpoints.

Spins up the FastAPI app in-process (no real service launch needed)
and exercises all 5 representations for each dataset, plus the
multi-encoder endpoint if the L12 index is present. Eyeballs the
top-3 hits per representation and prints a compact summary.

Unlike the other smoke scripts, this one uses ``httpx.AsyncClient``
against an ``ASGITransport`` -- no need to launch uvicorn.

Examples
--------
    # Default: both datasets, all reps, k=3.
    python scripts/smoke_hybrid.py

    # One dataset, top-5.
    python scripts/smoke_hybrid.py --datasets touche2020 --k 5

    # Skip multi-encoder (L12 index not yet built).
    python scripts/smoke_hybrid.py --no-multi-encoder
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

# Allow `python scripts/smoke_hybrid.py` from the repo root.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# httpx + ASGITransport -- no need to run a real uvicorn for a smoke test.
import httpx  # noqa: E402
from httpx import ASGITransport  # noqa: E402

from services.retrieval.app import config as config_mod  # noqa: E402
from services.retrieval.app import service as service_mod  # noqa: E402
from shared.ir_common.schemas import (  # noqa: E402
    MultiEncoderSearchRequest,
)

REPS: list[str] = [
    "tfidf",
    "bm25",
    "embedding",
    "hybrid_serial",
    "hybrid_parallel",
]

FUSIONS: list[str] = ["rrf", "combsum", "combmnz"]

DEFAULT_QUERIES: dict[str, list[str]] = {
    "touche2020": [
        "Should abortion be legalized?",
        "Is climate change caused by humans?",
    ],
    "nq": [
        "when was the declaration of independence signed",
        "what is the largest planet in the solar system",
    ],
}


def _load_doc_text(dataset_id: str, doc_id: str) -> str:
    """Read a snippet of the doc text from docs.jsonl (O(N) scan)."""
    docs_path = ROOT / "data" / "processed" / dataset_id / "docs.jsonl"
    if not docs_path.exists():
        return ""
    target = doc_id
    with docs_path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row["id"] == target:
                return str(row["text"][:200]).replace("\n", " ")
    return ""


def _print_hits(hits: list[dict[str, Any]], dataset_id: str) -> None:
    for h in hits:
        doc_id = h.get("doc_id", "?")
        score = h.get("score", 0.0)
        snippet = _load_doc_text(dataset_id, doc_id)
        ind = h.get("individual_scores", {})
        ind_str = ", ".join(f"{k}={v:.3f}" for k, v in ind.items())
        ind_p = f"  individual: {{{ind_str}}}" if ind_str else ""
        print(f"    rank={h.get('rank', '?')}  doc_id={doc_id}  score={score:.4f}{ind_p}")
        if snippet:
            print(f'      "{snippet}"')


async def _run_all(
    client: httpx.AsyncClient,
    datasets: list[str],
    k: int,
    do_multi_encoder: bool,
) -> None:
    for ds in datasets:
        if ds not in DEFAULT_QUERIES:
            print(f"[smoke] No default queries for {ds!r}; skipping.")
            continue
        for q in DEFAULT_QUERIES[ds]:
            print(f"\n========== {ds} :: {q!r} ==========")
            for rep in REPS:
                body: dict[str, Any] = {
                    "query": q,
                    "k": k,
                    "representation": rep,
                    "fusion": "rrf",
                }
                if rep in ("hybrid_serial", "hybrid_parallel"):
                    body["candidate_k"] = max(20, k * 4)
                t0 = time.perf_counter()
                r = await client.post(f"/hybrid/{ds}/search", json=body)
                ms = (time.perf_counter() - t0) * 1000
                if r.status_code != 200:
                    print(
                        f"  [{rep:18s}] HTTP {r.status_code} ({ms:.0f} ms) "
                        f"{r.json().get('detail', '')}"
                    )
                    continue
                payload = r.json()
                hits = payload.get("results", [])
                print(
                    f"  [{rep:18s}] {len(hits):>2d} hits  "
                    f"latency={payload.get('latency_ms', 0):>4d}ms  "
                    f"rtt={ms:>4.0f}ms"
                )
                _print_hits(hits[:3], ds)

            if not do_multi_encoder:
                continue
            if not config_mod.has_second_encoder_index(ds):
                print("  [multi_encoder     ] L12 index not built; skipping")
                continue
            for fusion in FUSIONS:
                body = MultiEncoderSearchRequest(query=q, k=k, fusion=fusion).model_dump()
                t0 = time.perf_counter()
                r = await client.post(f"/multi-encoder/{ds}/search", json=body)
                ms = (time.perf_counter() - t0) * 1000
                if r.status_code != 200:
                    print(
                        f"  [multi-encoder {fusion:7s}] HTTP {r.status_code} "
                        f"({ms:.0f} ms) {r.json().get('detail', '')}"
                    )
                    continue
                payload = r.json()
                hits = payload.get("results", [])
                print(
                    f"  [multi-encoder {fusion:7s}] {len(hits):>2d} hits  "
                    f"latency={payload.get('latency_ms', 0):>4d}ms  "
                    f"rtt={ms:>4.0f}ms"
                )
                _print_hits(hits[:3], ds)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Smoke test the Phase 5 hybrid / multi-encoder endpoints."
    )
    p.add_argument(
        "--datasets",
        nargs="+",
        default=["touche2020", "nq"],
        choices=["touche2020", "nq"],
        help="Which datasets to smoke (default: both).",
    )
    p.add_argument("--k", type=int, default=3, help="Top-k per query (default 3).")
    p.add_argument(
        "--no-multi-encoder",
        action="store_true",
        help="Skip the /multi-encoder endpoint even if L12 is built.",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    # Reset singletons between smoke runs.
    service_mod._EMBEDDER = None
    service_mod._FAISS_CACHE_2.clear()
    service_mod._DENSE_CLOSURE = None
    service_mod._ORCHESTRATOR = None
    service_mod._MULTI_ENCODER_CLOSURE = None

    import asyncio

    async def _go() -> None:
        transport = ASGITransport(app=service_mod.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            await _run_all(client, args.datasets, args.k, not args.no_multi_encoder)

    asyncio.run(_go())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
