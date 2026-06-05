"""Download the 2nd sentence-transformer encoder (Phase 5: all-MiniLM-L12-v2).

The 1st encoder (``all-MiniLM-L6-v2``, 90 MB) is already on disk from
Phase 3. The 2nd encoder is needed for the multi-encoder search path:

  * ``sentence-transformers/all-MiniLM-L12-v2`` -- 120 MB, 384-dim,
    6 layers (vs L6's 6 layers? actually L6 = 6 layers, L12 = 12 layers).
    The deeper model is slower per query but tends to score higher on
    BEIR / MS MARCO style benchmarks.

The download itself is just a pre-cache so the build script doesn't
fail at "model not found". It's a thin wrapper around
``sentence_transformers.SentenceTransformer`` (which delegates to
``huggingface_hub``).

Examples
--------
    python scripts/download_second_model.py
    python scripts/download_second_model.py --show-path
"""

from __future__ import annotations

# Set torch threads BEFORE any heavy import. On Windows the default is
# often 1, which makes CPU encoding ~10x slower than it needs to be.
import os

os.environ.setdefault("OMP_NUM_THREADS", "6")
os.environ.setdefault("MKL_NUM_THREADS", "6")

import argparse
import sys
from pathlib import Path

# Force UTF-8 on Windows.
try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
except Exception:
    pass

# Allow `python scripts/download_second_model.py` from the repo root.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.retrieval.app.config import SECOND_ENCODER_NAME  # noqa: E402


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Download the 2nd sentence-transformer model (Phase 5: L12)."
    )
    p.add_argument(
        "--model",
        default=SECOND_ENCODER_NAME,
        help=f"Hugging Face model name (default: {SECOND_ENCODER_NAME}).",
    )
    p.add_argument(
        "--show-path",
        action="store_true",
        help="Just print the resolved on-disk path and exit.",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    if args.show_path:
        # Resolve via huggingface_hub so the user sees the exact path
        # even if the model isn't downloaded yet.
        from huggingface_hub import snapshot_download

        try:
            path = snapshot_download(args.model, local_files_only=True)
        except Exception:  # noqa: BLE001
            # Not downloaded yet -- show the hub cache root instead.
            from huggingface_hub.constants import HF_HUB_CACHE

            path = str(Path(HF_HUB_CACHE) / ("models--" + args.model.replace("/", "--")))
        print(path)
        return 0

    print(f"[download] Fetching {args.model!r} via sentence-transformers...", flush=True)
    from sentence_transformers import SentenceTransformer

    t0 = __import__("time").time()
    st = SentenceTransformer(args.model)
    elapsed = __import__("time").time() - t0
    dim = int(
        getattr(st, "get_embedding_dimension", lambda: st.get_sentence_embedding_dimension())()
    )
    print(
        f"[download] Done in {elapsed:.1f}s. dim={dim}  " f"path={Path(st.cache_folder).resolve()}",
        flush=True,
    )
    print(
        "[download] You can now run `make build-dense-2` to encode both datasets "
        "with this encoder.",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
