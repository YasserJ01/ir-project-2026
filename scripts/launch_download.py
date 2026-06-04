#!/usr/bin/env python3
"""Launch the torch download as a fully detached process and exit immediately.

The child survives shell timeouts / parent termination because we use
DETACHED_PROCESS on Windows, which creates a new process group with no
controlling terminal.
"""

import subprocess
from pathlib import Path

DETACHED_PROCESS = 0x00000008
CREATE_NO_WINDOW = 0x08000000

ROOT = Path("F:/IR project")
PY = ROOT / ".venv" / "Scripts" / "python.exe"
SCRIPT = ROOT / "scripts" / "download_torch_gpu.py"
LOG = ROOT / "data" / "downloads" / "launcher.log"

LOG.parent.mkdir(parents=True, exist_ok=True)

with LOG.open("a", encoding="utf-8") as f:
    f.write(f"[launcher] starting {SCRIPT}\n")
    proc = subprocess.Popen(
        [str(PY), str(SCRIPT)],
        creationflags=DETACHED_PROCESS | CREATE_NO_WINDOW,
        cwd=str(ROOT),
        close_fds=True,
    )
    f.write(f"[launcher] child pid={proc.pid}\n")
    f.write("[launcher] exiting\n")
