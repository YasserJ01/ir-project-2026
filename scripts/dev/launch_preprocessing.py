"""Detached launcher for the preprocessing service (uvicorn :8001).

Not currently called by the UI flow, but the gateway /health endpoint
probes it, so start it for a clean "ok" status.
"""
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(r"F:\IR project")
VENV_PY = REPO / ".venv" / "Scripts" / "python.exe"
LOG = REPO / "data" / "dev_preprocessing.log"
ERR = REPO / "data" / "dev_preprocessing.err.log"
PID = REPO / "data" / "dev_preprocessing.pid"

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
         "services.preprocessing.app.pipeline:app",
         "--host", "0.0.0.0", "--port", "8001"],
        cwd=str(REPO),
        env=env,
        stdout=out,
        stderr=err,
        stdin=subprocess.DEVNULL,
        creationflags=flags,
        close_fds=True,
    )

PID.write_text(str(proc.pid))
print(f"preprocessing launched: pid={proc.pid}, log={LOG}")
