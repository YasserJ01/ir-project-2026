"""Download TinyLlama files to a local directory using direct HTTP for model weights.
Avoids huggingface_hub's unreliable large-file download on slow connections.
"""
import hashlib
import logging
import os
import shutil
import sys
import time
import traceback
from pathlib import Path

import requests

LOCAL_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "models" / "tinyllama"
LOG = Path(__file__).resolve().parent.parent.parent / "data" / "download_tinyllama.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(LOG), logging.StreamHandler()],
)
logger = logging.getLogger("download_tinyllama")
sys.excepthook = lambda tp, val, tb: logger.critical("Unhandled", exc_info=(tp, val, tb))

HF_BASE = "https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0/resolve/main"

SMALL_FILES = [
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "generation_config.json",
]

BIG_FILE = "model.safetensors"  # ~1 GB


def _download_small_files() -> None:
    """Download small config/tokenizer files using huggingface_hub (fast, reliable)."""
    from huggingface_hub import hf_hub_download

    MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    for fname in SMALL_FILES:
        dest = LOCAL_DIR / fname
        if dest.exists() and dest.stat().st_size > 0:
            logger.info("  %s already exists, skipping", fname)
            continue
        logger.info("Downloading %s ...", fname)
        t1 = time.perf_counter()
        path = hf_hub_download(repo_id=MODEL_ID, filename=fname, resume_download=True)
        elapsed = time.perf_counter() - t1
        shutil.copy2(path, dest)
        logger.info("  %s: %.1f MB in %.1f s", fname, dest.stat().st_size / (1024 * 1024), elapsed)


def _download_big_file() -> None:
    """Download model.safetensors with direct HTTP + streaming + progress logging."""
    dest = LOCAL_DIR / BIG_FILE
    if dest.exists() and dest.stat().st_size > 0:
        logger.info("  %s already exists (%.1f MB), verifying hash...", BIG_FILE, dest.stat().st_size / (1024 * 1024))
        if _verify(dest):
            logger.info("  %s hash OK, skipping", BIG_FILE)
            return
        logger.warning("  %s hash mismatch, re-downloading", BIG_FILE)
        dest.unlink()

    url = f"{HF_BASE}/{BIG_FILE}"
    tmp = dest.with_suffix(".tmp")

    logger.info("Downloading %s from %s ...", BIG_FILE, url)

    # HEAD request to get content-length
    try:
        head = requests.head(url, timeout=30, allow_redirects=True)
        total = int(head.headers.get("content-length", 0))
        logger.info("  Remote size: %.1f MB", total / (1024 * 1024))
    except Exception as e:
        logger.warning("  HEAD failed (%s), proceeding without size", e)
        total = 0

    # Streaming GET
    resp = requests.get(url, stream=True, timeout=(30, 600))
    resp.raise_for_status()

    if total == 0:
        total = int(resp.headers.get("content-length", 0))

    downloaded = 0
    t0 = time.perf_counter()
    last_report = 0
    last_report_time = t0

    with open(tmp, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                now = time.perf_counter()
                # Report every 10 MB or every 30 s
                if downloaded - last_report >= 10 * 1024 * 1024 or (now - last_report_time) >= 30:
                    pct = f"{100 * downloaded / total:.1f}%" if total else f"{downloaded / (1024*1024):.1f} MB"
                    elapsed = now - t0
                    speed = downloaded / (1024 * 1024 * elapsed) if elapsed > 0 else 0
                    logger.info("  %s of %.1f MB at %.1f MB/s", pct, total / (1024 * 1024), speed)
                    last_report = downloaded
                    last_report_time = now

    elapsed = time.perf_counter() - t0
    speed = downloaded / (1024 * 1024 * elapsed) if elapsed > 0 else 0
    logger.info("  Downloaded %.1f MB in %.1f s (%.1f MB/s)", downloaded / (1024 * 1024), elapsed, speed)

    # Move tmp to final location
    tmp.rename(dest)

    # Verify hash
    if not _verify(dest):
        logger.critical("  HASH MISMATCH after download! File may be corrupt.")
        dest.unlink(missing_ok=True)
        sys.exit(1)

    logger.info("  %s: download + verification OK", BIG_FILE)


def _verify(path: Path) -> bool:
    """Verify the sha256 of model.safetensors matches the expected value."""
    # We compute sha256 and compare against expected from HuggingFace
    logger.info("  Computing sha256 of %s ...", path.name)
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(65536)
            if not chunk:
                break
            h.update(chunk)
    computed = h.hexdigest()

    # Get expected hash from HF API
    expected = _get_expected_hash(path.name)
    if expected and computed != expected:
        logger.error("  sha256 mismatch: computed=%s expected=%s", computed, expected)
        return False
    logger.info("  sha256: %s", computed)
    return True


def _get_expected_hash(filename: str) -> str | None:
    """Fetch expected sha256 from HuggingFace model API."""
    import json

    url = f"https://huggingface.co/api/models/TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        for sibling in data.get("siblings", []):
            if sibling["rfilename"] == filename:
                return sibling.get("sha256")
    except Exception as e:
        logger.warning("  Could not fetch expected hash: %s", e)
    return None


def main() -> None:
    LOCAL_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Target directory: %s", LOCAL_DIR)
    logger.info("Log file: %s", LOG)

    t0 = time.perf_counter()

    _download_small_files()
    _download_big_file()

    total = time.perf_counter() - t0
    total_mb = sum(f.stat().st_size for f in LOCAL_DIR.iterdir() if f.is_file()) / (1024 * 1024)
    logger.info("All files downloaded: %.1f MB total in %.1f min", total_mb, total / 60)

    # Verify by loading the model
    logger.info("Loading model from local path to verify...")
    import torch
    from transformers import pipeline as hf_pipeline

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    t1 = time.perf_counter()
    _ = hf_pipeline("text-generation", model=str(LOCAL_DIR), device=device, torch_dtype=dtype)
    loaded = time.perf_counter() - t1
    logger.info("Model loaded + verified in %.1f s (device=%s)", loaded, device)
    logger.info("ALL DONE. Local model is ready at %s", LOCAL_DIR)


if __name__ == "__main__":
    main()
