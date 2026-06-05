"""Report the build status of the L12 (2nd-encoder) FAISS indexes.

Reads ``build_meta_l12.json`` for each known dataset and prints a
one-line summary per dataset. Useful as a polling target while the
build runs in the background via ``launch_dense_2.py``.

Examples
--------
    # Print current status.
    python scripts/check_dense_2_status.py

    # Tail the log + status in a single command.
    python scripts/check_dense_2_status.py --watch 30
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Allow `python scripts/check_dense_2_status.py` from the repo root.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.retrieval.app.config import (  # noqa: E402
    DATASETS,
    SECOND_ENCODER_EMBEDDINGS_FILENAME,
    SECOND_ENCODER_INDEX_FILENAME,
    index_dir,
)
from shared.ir_common.schemas import DATASET_IDS  # noqa: E402


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Report L12 (2nd-encoder) build status per dataset.")
    p.add_argument(
        "--datasets",
        nargs="+",
        choices=DATASET_IDS,
        default=list(DATASETS),
        help="Which datasets to check (default: all).",
    )
    p.add_argument(
        "--watch",
        type=int,
        default=0,
        metavar="SECS",
        help="Re-print every SECS seconds until both datasets are 'ok'.",
    )
    return p.parse_args()


def _status_for(dataset_id: str) -> dict[str, object]:
    """Return a small dict describing the L12 build state for ``dataset_id``."""
    d = index_dir(dataset_id)
    faiss_path = d / SECOND_ENCODER_INDEX_FILENAME
    emb_path = d / SECOND_ENCODER_EMBEDDINGS_FILENAME
    meta_path = d / "build_meta_l12.json"
    out: dict[str, object] = {"dataset_id": dataset_id, "state": "missing"}
    if meta_path.exists():
        try:
            m = json.loads(meta_path.read_text(encoding="utf-8"))
            out["state"] = m.get("status", "unknown")
            out["built_at"] = m.get("built_at", "")
            out["num_vectors"] = m.get("num_vectors", 0)
            out["elapsed_seconds"] = m.get("elapsed_seconds", 0.0)
            out["size_mb"] = m.get("size_mb", 0.0)
            out["model_name"] = m.get("model_name", "")
        except (json.JSONDecodeError, OSError):
            out["state"] = "corrupt_meta"
    elif faiss_path.exists() and not meta_path.exists():
        # FAISS written but build_meta not yet -- still in progress.
        out["state"] = "in_progress"
        if faiss_path.exists():
            out["size_mb"] = round(faiss_path.stat().st_size / (1024 * 1024), 1)
        if emb_path.exists():
            out["emb_size_mb"] = round(emb_path.stat().st_size / (1024 * 1024), 1)
    return out


def _print_table(rows: list[dict[str, object]]) -> None:
    print(f"{'dataset':12s}  {'state':12s}  {'vectors':>10s}  {'elapsed':>9s}  {'size_mb':>8s}")
    print("-" * 60)
    for r in rows:
        ds = str(r.get("dataset_id", "?"))
        state = str(r.get("state", "?"))
        n = r.get("num_vectors", 0)
        sec = r.get("elapsed_seconds", 0.0)
        mb = r.get("size_mb", 0.0)
        n_str = f"{n:>10,}" if isinstance(n, int) else f"{str(n):>10s}"
        sec_str = f"{sec:>8.1f}s" if isinstance(sec, (int, float)) else f"{str(sec):>9s}"
        mb_str = f"{mb:>7.1f}" if isinstance(mb, (int, float)) else f"{str(mb):>8s}"
        print(f"{ds:12s}  {state:12s}  {n_str}  {sec_str}  {mb_str} MB")


def _all_ok(rows: list[dict[str, object]]) -> bool:
    return all(r.get("state") == "ok" for r in rows)


def main() -> int:
    args = _parse_args()
    while True:
        rows = [_status_for(ds) for ds in args.datasets]
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"\n[{ts}] L12 build status:")
        _print_table(rows)
        if not args.watch:
            return 0
        if _all_ok(rows):
            print("\nAll datasets: 'ok'. Multi-encoder is live.")
            return 0
        print(f"\nSleeping {args.watch}s... (Ctrl-C to stop)")
        try:
            time.sleep(args.watch)
        except KeyboardInterrupt:
            return 0


if __name__ == "__main__":
    raise SystemExit(main())
