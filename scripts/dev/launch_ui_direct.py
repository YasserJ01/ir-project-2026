"""Run vite directly (bypassing npm.cmd) to see console output.

This script is for debugging only — normal operation uses npm.cmd.
"""
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(r"F:\IR project")
UI = REPO / "services" / "ui"
LOG = REPO / "data" / "dev_ui_direct.log"

env = os.environ.copy()
env["PATH"] = (
    r"C:\Program Files\nodejs;C:\Program Files\Git\cmd;"
    + env.get("PATH", "")
)
env["NODE_ENV"] = "development"

# Run vite directly via node (not npm.cmd). Detached so it survives
# the opencode 120s shell timeout.
flags = (
    subprocess.DETACHED_PROCESS
    | subprocess.CREATE_NO_WINDOW
    | subprocess.CREATE_NEW_PROCESS_GROUP
)

vite_js = UI / "node_modules" / "vite" / "bin" / "vite.js"
if not vite_js.exists():
    sys.exit("vite.js not found; run: cd services/ui && npm install")

with open(LOG, "wb") as out:
    proc = subprocess.Popen(
        [r"C:\Program Files\nodejs\node.exe", str(vite_js),
         "--port", "5173", "--strictPort", "--host", "127.0.0.1"],
        cwd=str(UI),
        env=env,
        stdout=out,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        creationflags=flags,
        close_fds=True,
    )

print(f"vite (direct) launched: pid={proc.pid}, log={LOG}")
