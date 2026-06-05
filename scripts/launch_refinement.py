"""Launch the refinement service in the background and exit immediately.

Survives the opencode 120s shell timeout by detaching from the
parent process group. Stops cleanly when the parent PowerShell
process is killed (close-fds=True means a new console group).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG_PATH = ROOT / "data" / "refinement_service.log"

CREATE_NEW_PROCESS_GROUP = 0x00000200
DETACHED_PROCESS = 0x00000008
CREATE_NO_WINDOW = 0x08000000


def main() -> int:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    log_fh = open(LOG_PATH, "ab", buffering=0)
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "services.refinement.app.service:app",
            "--port",
            "8004",
            "--host",
            "0.0.0.0",
        ],
        cwd=str(ROOT),
        stdout=log_fh,
        stderr=log_fh,
        stdin=subprocess.DEVNULL,
        creationflags=DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP | CREATE_NO_WINDOW,
        close_fds=True,
    )
    print(f"OK  Launched refinement service PID={proc.pid}, log -> {LOG_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
