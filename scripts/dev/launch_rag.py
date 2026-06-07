"""Detached launcher for the RAG service (uvicorn :8005)."""
import os
import subprocess
from pathlib import Path

REPO = Path(r"F:\IR project")
VENV_PY = REPO / ".venv" / "Scripts" / "python.exe"
LOG = REPO / "data" / "dev_rag.log"
ERR = REPO / "data" / "dev_rag.err.log"
PID = REPO / "data" / "dev_rag.pid"

env = os.environ.copy()
env["PATH"] = r"C:\Program Files\Git\cmd;C:\Windows\System32;" + env.get("PATH", "")
env["PYTHONPATH"] = str(REPO) + os.pathsep + env.get("PYTHONPATH", "")
env["RAG_CLIENT_TIMEOUT"] = "180"

flags = (
    subprocess.DETACHED_PROCESS
    | subprocess.CREATE_NO_WINDOW
    | subprocess.CREATE_NEW_PROCESS_GROUP
)

with open(LOG, "wb") as out, open(ERR, "wb") as err:
    proc = subprocess.Popen(
        [str(VENV_PY), "-m", "uvicorn",
         "services.rag.app.service:app",
         "--host", "0.0.0.0", "--port", "8005"],
        cwd=str(REPO),
        env=env,
        stdout=out,
        stderr=err,
        stdin=subprocess.DEVNULL,
        creationflags=flags,
        close_fds=True,
    )

PID.write_text(str(proc.pid))
print(f"rag launched: pid={proc.pid}, log={LOG}")
