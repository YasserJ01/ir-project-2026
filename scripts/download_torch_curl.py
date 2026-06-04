#!/usr/bin/env python3
"""Download torch+cu121 with curl + auto-resume. Survives flaky networks.

Uses curl's ``-C -`` flag to resume from where a previous attempt left off.
PyTorch's CDN (AmazonS3 + CloudFront) supports HTTP range requests, so
this is safe to interrupt and restart.

The wheel is saved to data/downloads/. On success a DONE marker is written.
"""

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("F:/IR project")
WHEEL_DIR = ROOT / "data" / "downloads"
WHEEL_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = WHEEL_DIR / "download_curl.log"
DONE_MARKER = WHEEL_DIR / "DONE"
WHEEL_NAME = "torch-2.5.1+cu121-cp312-cp312-win_amd64.whl"
WHEEL_PATH = WHEEL_DIR / WHEEL_NAME
URL = "https://download.pytorch.org/whl/cu121/" + WHEEL_NAME.replace("+", "%2B")
CURL = Path(r"C:\Windows\System32\curl.exe")
EXPECTED_SIZE = 2_449_331_371
MIN_SIZE_BYTES = 2_400_000_000  # tolerate a few KB of HTTP/encoding overhead
MAX_ATTEMPTS = 30
SLEEP_BETWEEN = 15


def log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def curl_size() -> int:
    return WHEEL_PATH.stat().st_size if WHEEL_PATH.exists() else 0


def is_complete() -> bool:
    return WHEEL_PATH.exists() and WHEEL_PATH.stat().st_size >= MIN_SIZE_BYTES


def attempt() -> bool:
    have = curl_size()
    if have >= MIN_SIZE_BYTES:
        log(f"wheel already complete: {have/1e6:.1f} MB")
        return True
    if have:
        log(f"resuming from {have/1e6:.1f} MB (curl -C -)")
    else:
        log("starting fresh download")
    rc = subprocess.call(
        [
            str(CURL),
            "-L",  # follow redirects
            "-C",
            "-",  # auto-resume
            "--retry",
            "5",  # curl-internal retries on transient errors
            "--retry-delay",
            "3",
            "--retry-all-errors",
            "-o",
            str(WHEEL_PATH),
            URL,
        ],
        cwd=str(ROOT),
    )
    have = curl_size()
    log(f"curl exit={rc}, wheel={have/1e6:.1f} MB")
    if rc == 0 and have >= MIN_SIZE_BYTES:
        return True
    return False


def main() -> int:
    log("=== torch+cu121 curl-resume download started ===")
    log(f"url: {URL}")
    log(f"target: {WHEEL_PATH}")
    log(f"max attempts: {MAX_ATTEMPTS}, sleep: {SLEEP_BETWEEN}s")
    for i in range(1, MAX_ATTEMPTS + 1):
        log(f"[attempt {i}/{MAX_ATTEMPTS}]")
        if attempt():
            log("SUCCESS")
            DONE_MARKER.write_text("OK", encoding="utf-8")
            return 0
        if i < MAX_ATTEMPTS:
            log(f"sleeping {SLEEP_BETWEEN}s before retry...")
            time.sleep(SLEEP_BETWEEN)
    log(f"GAVE UP after {MAX_ATTEMPTS} attempts")
    DONE_MARKER.write_text("FAIL", encoding="utf-8")
    return 1


if __name__ == "__main__":
    sys.exit(main())
