#!/usr/bin/env python3
"""Download torch+cu121 with retry. Runs forever until success or max attempts."""

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("F:/IR project")
VENV_PY = ROOT / ".venv" / "Scripts" / "python.exe"
WHEEL_DIR = ROOT / "data" / "downloads"
WHEEL_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = WHEEL_DIR / "download.log"
PIP_OUT = WHEEL_DIR / "pip.out.log"
PIP_ERR = WHEEL_DIR / "pip.err.log"
DONE_MARKER = WHEEL_DIR / "DONE"
MIN_SIZE_BYTES = 500 * 1024 * 1024
MAX_ATTEMPTS = 30
SLEEP_BETWEEN = 30

URL = "https://download.pytorch.org/whl/cu121"
PKG = "torch==2.5.1+cu121"


def log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def find_wheel() -> Path | None:
    for p in WHEEL_DIR.glob("torch-*.whl"):
        if "any.whl" in p.name:
            continue
        return p
    return None


def attempt() -> bool:
    existing = find_wheel()
    if existing and existing.stat().st_size >= MIN_SIZE_BYTES:
        log(f"wheel already present: {existing.name} ({existing.stat().st_size/1e6:.1f} MB)")
        return True
    if existing:
        log(f"removing partial wheel: {existing.name} ({existing.stat().st_size/1e6:.1f} MB)")
        existing.unlink()

    log("running: pip download")
    with PIP_OUT.open("w", encoding="utf-8") as out, PIP_ERR.open("w", encoding="utf-8") as err:
        rc = subprocess.call(
            [
                str(VENV_PY),
                "-m",
                "pip",
                "download",
                PKG,
                "--index-url",
                URL,
                "--dest",
                str(WHEEL_DIR),
                "--no-deps",
                "--disable-pip-version-check",
                "--no-cache-dir",
            ],
            stdout=out,
            stderr=err,
            cwd=str(ROOT),
        )

    wheel = find_wheel()
    size = wheel.stat().st_size if wheel else 0
    log(f"pip exit={rc}, wheel={wheel.name if wheel else 'NONE'} ({size/1e6:.1f} MB)")
    if rc == 0 and wheel and size >= MIN_SIZE_BYTES:
        return True
    return False


def main() -> int:
    log("=== torch+cu121 download started ===")
    log(f"wheel dir: {WHEEL_DIR}")
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
