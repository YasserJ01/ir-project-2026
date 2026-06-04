"""Smoke test the dense retrieval service.

Hand-picked queries against both datasets. Prints the top-k hits with
a snippet from the doc text for eyeball verification. Mirrors the
shape of :mod:`scripts.smoke_search` (Phase 2) so the output is
comparable.

Examples
--------
    # Top-3 (default).
    python scripts/smoke_dense.py

    # Top-5.
    python scripts/smoke_dense.py --k 5
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

# Allow `python scripts/smoke_dense.py` from the repo root.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.retrieval.app import embedder as embedder_mod  # noqa: E402
from services.retrieval.app import vector_store as vector_store_mod  # noqa: E402
from services.retrieval.app.config import (  # noqa: E402
    DEFAULT_MODEL_NAME,
    index_dir,
)

DEFAULT_QUERIES: dict[str, list[str]] = {
    "touche2020": [
        "Should abortion be legalized?",
        "Is climate change caused by humans?",
        "Should the death penalty be abolished?",
    ],
    "nq": [
        "when was the declaration of independence signed",
        "what is the largest planet in the solar system",
        "how many continents are there in the world",
    ],
}


def _load_doc_text(dataset_id: str, doc_id: str) -> str:
    """Read a snippet of the doc text from docs.jsonl (O(N) scan)."""
    docs_path = Path(ROOT) / "data" / "processed" / dataset_id / "docs.jsonl"
    if not docs_path.exists():
        return ""
    target = doc_id
    with docs_path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row["id"] == target:
                return str(row["text"][:200]).replace("\n", " ")
    return ""


def _smoke(dataset_id: str, queries: list[str], k: int, model_name: str) -> None:
    d = index_dir(dataset_id)
    if not (d / vector_store_mod.INDEX_FILENAME).exists():
        print(f"[smoke] {dataset_id} has no FAISS index. Run `make build-dense` first.")
        return

    print(f"\n========== {dataset_id} ==========")
    print(f"  index dir: {d}")
    print(f"  model:     {model_name}")

    # Lazy-load the model + index.
    emb = embedder_mod.Embedder(default_model_name=model_name)
    emb.warm_up(model_name)
    idx = vector_store_mod.DenseIndex.load(d)
    print(f"  faiss:     {idx.size():,} vectors x {idx.dim()}-dim")

    for q_text in queries:
        print(f"\n--- query: {q_text!r} ---")
        t0 = time.perf_counter()
        q_vec = emb.encode_query(q_text, model_name=model_name)
        scores, ids = idx.search(q_vec, k)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        print(f"  top-{k} ({elapsed_ms:.1f} ms):")
        for rank, (s, i) in enumerate(zip(scores, ids, strict=True), start=1):
            doc_id = idx.doc_ids[int(i)]
            snippet = _load_doc_text(dataset_id, doc_id)
            print(f"    rank={rank} doc_id={doc_id} score={s:.4f}")
            if snippet:
                print(f'      "{snippet}"')
            else:
                print(f"      (no text available for {doc_id})")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Smoke test the dense retrieval index.")
    p.add_argument(
        "--datasets",
        nargs="+",
        default=["touche2020", "nq"],
        choices=["touche2020", "nq"],
        help="Which datasets to smoke (default: both).",
    )
    p.add_argument("--k", type=int, default=3, help="Top-k per query (default 3).")
    p.add_argument(
        "--model",
        default=DEFAULT_MODEL_NAME,
        help=f"Model name (default: {DEFAULT_MODEL_NAME}).",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    for ds in args.datasets:
        queries = DEFAULT_QUERIES.get(ds, [])
        if not queries:
            print(f"[smoke] No default queries for {ds!r}; skipping.")
            continue
        _smoke(ds, queries, args.k, args.model)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
