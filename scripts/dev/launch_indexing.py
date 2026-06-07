"""Detached launcher for the indexing service (uvicorn :8002).

Spawns uvicorn as a fully detached process so it survives the opencode
120s shell timeout. Logs to data/dev_indexing.{log,err.log}. PID is
written to data/dev_indexing.pid for stop_all.py.
"""
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(r"F:\IR project")
VENV_PY = REPO / ".venv" / "Scripts" / "python.exe"
LOG = REPO / "data" / "dev_indexing.log"
ERR = REPO / "data" / "dev_indexing.err.log"
PID = REPO / "data" / "dev_indexing.pid"

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
         "services.indexing.app.service:app",
         "--host", "0.0.0.0", "--port", "8002"],
        cwd=str(REPO),
        env=env,
        stdout=out,
        stderr=err,
        stdin=subprocess.DEVNULL,
        creationflags=flags,
        close_fds=True,
    )

PID.write_text(str(proc.pid))
print(f"indexing launched: pid={proc.pid}, log={LOG}")
