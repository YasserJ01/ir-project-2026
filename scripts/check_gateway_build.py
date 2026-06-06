"""Check the gateway image build status (Phase 6).

Reads the last few lines of the build log and checks whether the
image exists in the local Docker registry.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    log_path = repo_root / "data" / "build_gateway_image.log"
    err_log_path = repo_root / "data" / "build_gateway_image.err.log"

    if log_path.exists():
        print(f"=== {log_path} (last 30 lines) ===")
        text = log_path.read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines()
        for line in lines[-30:]:
            print(line)
    else:
        print(f"No log file at {log_path} yet.")

    print()
    if err_log_path.exists():
        size = err_log_path.stat().st_size
        print(f"=== {err_log_path} (size: {size} bytes) ===")
        if size > 0:
            text = err_log_path.read_text(encoding="utf-8", errors="replace")
            for line in text.splitlines()[-20:]:
                print(line)

    print()
    print("=== Docker image status ===")
    if shutil.which("docker") is None:
        print("docker not on PATH")
        return 1
    r = subprocess.run(
        ["docker", "images", "--format", "{{.Repository}}:{{.Tag}}  {{.Size}}  {{.CreatedAt}}"],
        capture_output=True, text=True,
    )
    for line in r.stdout.splitlines():
        if "ir-project" in line.lower() or "gateway" in line.lower():
            print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
