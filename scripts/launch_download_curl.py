#!/usr/bin/env python3
"""Launch the curl download as a fully detached process. Exits immediately."""

import subprocess
from pathlib import Path

DETACHED_PROCESS = 0x00000008
CREATE_NO_WINDOW = 0x08000000

ROOT = Path("F:/IR project")
PY = ROOT / ".venv" / "Scripts" / "python.exe"
SCRIPT = ROOT / "scripts" / "download_torch_curl.py"
LOG = ROOT / "data" / "downloads" / "launcher_curl.log"

LOG.parent.mkdir(parents=True, exist_ok=True)

with LOG.open("a", encoding="utf-8") as f:
    f.write(f"[launcher_curl] starting {SCRIPT}\n")
    proc = subprocess.Popen(
        [str(PY), str(SCRIPT)],
        creationflags=DETACHED_PROCESS | CREATE_NO_WINDOW,
        cwd=str(ROOT),
        close_fds=True,
    )
    f.write(f"[launcher_curl] child pid={proc.pid}\n")
    f.write("[launcher_curl] exiting\n")
