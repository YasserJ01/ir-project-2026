"""Detached launcher for building the 4 remaining backend service images.

The 4 services that need building (after gateway + ui are already done):
- preprocessing
- indexing
- retrieval (CPU, since compose is CPU)
- refinement

Each backend service needs ~2-3 min on this machine (torch wheel is
already cached from the gateway build), but the shell tool has a
120s timeout. Launching detached survives the timeout.

Logs go to data/build_backend_4.log.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    log_path = repo_root / "data" / "build_backend_4.log"
    err_log_path = repo_root / "data" / "build_backend_4.err.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    creationflags = 0
    if sys.platform == "win32":
        DETACHED_PROCESS = 0x00000008
        CREATE_NO_WINDOW = 0x08000000
        CREATE_NEW_PROCESS_GROUP = 0x00000200
        creationflags = DETACHED_PROCESS | CREATE_NO_WINDOW | CREATE_NEW_PROCESS_GROUP

    services = ["preprocessing", "indexing", "retrieval", "refinement"]
    cmd = [
        "docker", "compose", "-f", "docker-compose.yml",
        "build", "--parallel", *services,
    ]
    print(f"Launching: {' '.join(cmd)}")
    print(f"  stdout -> {log_path}")
    print(f"  stderr -> {err_log_path}")

    # Truncate old logs.
    for p in (log_path, err_log_path):
        if p.exists():
            p.unlink()

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
