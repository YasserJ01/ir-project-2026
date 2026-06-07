"""Detached launcher for the API gateway (uvicorn :8000).

Depends on indexing(:8002), retrieval(:8003), refinement(:8004), and
preprocessing(:8001) being reachable. Without --reload so the import
time stays fast on slow disks.
"""
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(r"F:\IR project")
VENV_PY = REPO / ".venv" / "Scripts" / "python.exe"
LOG = REPO / "data" / "dev_gateway.log"
ERR = REPO / "data" / "dev_gateway.err.log"
PID = REPO / "data" / "dev_gateway.pid"

env = os.environ.copy()
env["PATH"] = r"C:\Program Files\Git\cmd;C:\Windows\System32;" + env.get("PATH", "")
env["PYTHONPATH"] = str(REPO) + os.pathsep + env.get("PYTHONPATH", "")

flags = (
    subprocess.DETACHED_PROCESS
    | subprocess.CREATE_NO_WINDOW
    | subprocess.CREATE_NEW_PROCESS_GROUP
)

with open(LOG, "wb") as out, open(ERR, "wb") as err:
    proc = subprocess.Popen(
        [str(VENV_PY), "-m", "uvicorn",
         "services.gateway.app.main:app",
         "--host", "0.0.0.0", "--port", "8000"],
        cwd=str(REPO),
        env=env,
        stdout=out,
        stderr=err,
        stdin=subprocess.DEVNULL,
        creationflags=flags,
        close_fds=True,
    )

PID.write_text(str(proc.pid))
print(f"gateway launched: pid={proc.pid}, log={LOG}")
