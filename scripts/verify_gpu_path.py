#!/usr/bin/env python3
"""Verify the GPU path is ready: torch+cu121 + CUDA available + sentence-transformers works.

Run this AFTER `make install-torch-gpu`. Exits 0 on success, non-zero on failure.
"""

import sys


def main() -> int:
    print("[1/5] importing torch...")
    import torch

    print(f"  torch={torch.__version__} cuda={torch.version.cuda}")
    if not torch.cuda.is_available():
        print("  ERROR: torch.cuda.is_available() is False")
        return 1
    print(f"  cuda available: {torch.cuda.get_device_name(0)}")
    print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    print("[2/5] importing sentence_transformers...")
    from sentence_transformers import SentenceTransformer

    print("  ok")

    print("[3/5] loading model (small, cached)...")
    import os

    from services.retrieval.app.config import model_cache_dir

    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    cache = str(model_cache_dir(model_name))
    cache_folder = cache if os.path.isdir(cache) else None
    st = SentenceTransformer(model_name, device="cuda", cache_folder=cache_folder)
    print(f"  model loaded, dim={st.get_sentence_embedding_dimension()}")

    print("[4/5] encoding 8 sentences (warmup)...")
    vecs = st.encode(
        ["the quick brown fox", "jumps over the lazy dog", "hello world"] * 3,
        batch_size=8,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    print(f"  shape={vecs.shape} dtype={vecs.dtype}")

    print("[5/5] measuring throughput (1000 short docs)...")
    import time

    docs = ["the quick brown fox jumps over the lazy dog"] * 1000
    t0 = time.time()
    vecs = st.encode(
        docs,
        batch_size=512,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    elapsed = time.time() - t0
    print(f"  1000 docs in {elapsed:.2f}s = {1000/elapsed:.0f} docs/sec")

    print("\nGPU PATH READY")
    return 0


if __name__ == "__main__":
    sys.exit(main())
