"""Detached launcher for the gateway Docker image build (Phase 6).

The 2.4 GB torch wheel is the bottleneck (~80 min on 4 Mbps). We
launch in a detached subprocess so the build survives the opencode
120s shell timeout. Logs go to data/build_gateway_image.log.

Run:
    python scripts/launch_gateway_build.py
    # then check status with:
    Get-Content data\build_gateway_image.log -Tail 20
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    log_path = repo_root / "data" / "build_gateway_image.log"
    err_log_path = repo_root / "data" / "build_gateway_image.err.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    creationflags = 0
    if sys.platform == "win32":
        # Detach from the current process group + no window.
        DETACHED_PROCESS = 0x00000008
        CREATE_NO_WINDOW = 0x08000000
        CREATE_NEW_PROCESS_GROUP = 0x00000200
        creationflags = DETACHED_PROCESS | CREATE_NO_WINDOW | CREATE_NEW_PROCESS_GROUP

    cmd = ["docker", "compose", "-f", "docker-compose.yml", "build", "gateway"]
    print(f"Launching: {' '.join(cmd)}")
    print(f"  stdout -> {log_path}")
    print(f"  stderr -> {err_log_path}")

    with open(log_path, "ab", buffering=0) as out, open(err_log_path, "ab", buffering=0) as err:
        subprocess.Popen(
            cmd,
            stdout=out,
            stderr=err,
            cwd=str(repo_root),
            close_fds=True,
            creationflags=creationflags,
        )
    print("Launched. Check the log files for progress.")
    print(f"  Get-Content {log_path} -Tail 30")
    return 0


if __name__ == "__main__":
    sys.exit(main())
