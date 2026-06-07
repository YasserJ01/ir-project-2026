"""Detached launcher for the dense-retrieval service (uvicorn :8003).

NOTE: First search call will take 10-30s while the MiniLM L6+L12
models and FAISS indexes load into memory. Subsequent calls are
60-140 ms.
"""
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(r"F:\IR project")
VENV_PY = REPO / ".venv" / "Scripts" / "python.exe"
LOG = REPO / "data" / "dev_retrieval.log"
ERR = REPO / "data" / "dev_retrieval.err.log"
PID = REPO / "data" / "dev_retrieval.pid"

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
         "services.retrieval.app.service:app",
         "--host", "0.0.0.0", "--port", "8003"],
        cwd=str(REPO),
        env=env,
        stdout=out,
        stderr=err,
        stdin=subprocess.DEVNULL,
        creationflags=flags,
        close_fds=True,
    )

PID.write_text(str(proc.pid))
print(f"retrieval launched: pid={proc.pid}, log={LOG}")
