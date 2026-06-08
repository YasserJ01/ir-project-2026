"""Download TinyLlama 1.1B GGUF (Q4_K_M) from TheBloke via direct HTTP streaming.

Output: data/models/tinyllama/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf (~700 MB).
"""
import hashlib
import logging
import os
import sys
import time
from pathlib import Path

import requests

GGUF_FILENAME = "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
HF_REPO = "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF"
HF_BASE = f"https://huggingface.co/{HF_REPO}/resolve/main"

LOCAL_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "models" / "tinyllama"
LOG = Path(__file__).resolve().parent.parent.parent / "data" / "download_tinyllama_gguf.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(LOG, encoding="utf-8"), logging.StreamHandler()],
)
logger = logging.getLogger("download_tinyllama_gguf")
sys.excepthook = lambda tp, val, tb: logger.critical("Unhandled", exc_info=(tp, val, tb))


def _get_expected_hash(filename: str) -> str | None:
    url = f"https://huggingface.co/api/models/{HF_REPO}"
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        for sibling in data.get("siblings", []):
            if sibling["rfilename"] == filename:
                return sibling.get("sha256")
    except Exception as e:
        logger.warning("Could not fetch expected hash: %s", e)
    return None


def _verify(path: Path, expected: str | None) -> bool:
    logger.info("Computing sha256 of %s ...", path.name)
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(65536)
            if not chunk:
                break
            h.update(chunk)
    computed = h.hexdigest()
    if expected and computed != expected:
        logger.error("sha256 mismatch: computed=%s expected=%s", computed, expected)
        return False
    logger.info("sha256: %s", computed)
    return True


def main() -> None:
    LOCAL_DIR.mkdir(parents=True, exist_ok=True)
    dest = LOCAL_DIR / GGUF_FILENAME

    if dest.exists() and dest.stat().st_size > 0:
        logger.info("%s already exists (%.1f MB), verifying hash...", GGUF_FILENAME, dest.stat().st_size / (1024 * 1024))
        expected = _get_expected_hash(GGUF_FILENAME)
        if _verify(dest, expected):
            logger.info("Hash OK, skipping download")
            return
        logger.warning("Hash mismatch, re-downloading")
        dest.unlink()

    url = f"{HF_BASE}/{GGUF_FILENAME}"
    tmp = dest.with_suffix(".tmp")

    logger.info("Downloading %s from %s ...", GGUF_FILENAME, url)

    try:
        head = requests.head(url, timeout=30, allow_redirects=True)
        total = int(head.headers.get("content-length", 0))
        logger.info("Remote size: %.1f MB", total / (1024 * 1024))
    except Exception as e:
        logger.warning("HEAD failed (%s), proceeding without size", e)
        total = 0

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
                if downloaded - last_report >= 10 * 1024 * 1024 or (now - last_report_time) >= 30:
                    pct = f"{100 * downloaded / total:.1f}%" if total else f"{downloaded / (1024*1024):.1f} MB"
                    elapsed = now - t0
                    speed = downloaded / (1024 * 1024 * elapsed) if elapsed > 0 else 0
                    logger.info("  %s of %.1f MB at %.1f MB/s", pct, total / (1024 * 1024), speed)
                    last_report = downloaded
                    last_report_time = now

    elapsed = time.perf_counter() - t0
    speed = downloaded / (1024 * 1024 * elapsed) if elapsed > 0 else 0
    logger.info("Downloaded %.1f MB in %.1f s (%.1f MB/s)", downloaded / (1024 * 1024), elapsed, speed)

    tmp.rename(dest)

    expected = _get_expected_hash(GGUF_FILENAME)
    if not _verify(dest, expected):
        logger.critical("HASH MISMATCH after download! File may be corrupt.")
        dest.unlink(missing_ok=True)
        sys.exit(1)

    logger.info("%s: download + verification OK", GGUF_FILENAME)
    logger.info("Total size: %.1f MB", dest.stat().st_size / (1024 * 1024))


if __name__ == "__main__":
    main()
