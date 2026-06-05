"""Launch the L12 (2nd-encoder) FAISS index build in the background and exit.

The build encodes 882,544 documents across both datasets with
``all-MiniLM-L12-v2`` and writes ~2.6 GB of FAISS + numpy data to
disk. Wall time on the GTX 1650 Max-Q is ~3.7 hours; on CPU-only it
would be 15+ hours. Either way, the build outlasts the opencode
120-second shell timeout, so this launcher detaches the build process
from the parent console.

Log goes to ``data/build_dense_2.log`` and stderr is also captured to
``data/build_dense_2.err.log`` so the user can tail the live progress
in a separate PowerShell window:

    Get-Content -Path F:\\IR\\project\\data\\build_dense_2.log -Wait
    Get-Content -Path F:\\IR\\project\\data\\build_dense_2.err.log -Wait

The build is idempotent -- if you stop it mid-way and re-launch, the
``build_meta_l12.json`` sentinel will not yet exist for the
in-progress dataset, so the build will resume from the top of that
dataset. To force a clean rebuild, delete ``build_meta_l12.json`` and
the L12 files first.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG_PATH = ROOT / "data" / "build_dense_2.log"
ERR_PATH = ROOT / "data" / "build_dense_2.err.log"

CREATE_NEW_PROCESS_GROUP = 0x00000200
DETACHED_PROCESS = 0x00000008
CREATE_NO_WINDOW = 0x08000000


def main() -> int:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    log_fh = open(LOG_PATH, "ab", buffering=0)
    err_fh = open(ERR_PATH, "ab", buffering=0)
    proc = subprocess.Popen(
        [
            sys.executable,
            "-u",
            "scripts/build_dense_2.py",
        ],
        cwd=str(ROOT),
        stdout=log_fh,
        stderr=err_fh,
        stdin=subprocess.DEVNULL,
        creationflags=DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP | CREATE_NO_WINDOW,
        close_fds=True,
    )
    print(f"OK  Launched L12 build PID={proc.pid}")
    print(f"    stdout -> {LOG_PATH}")
    print(f"    stderr -> {ERR_PATH}")
    print()
    print("Tail the build in another PowerShell with:")
    print(f"  Get-Content -Path '{LOG_PATH}' -Wait")
    print()
    print("Check progress with:")
    print("  python scripts/check_dense_2_status.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
