"""Detached launcher for the Vite dev server (port 5173).

Vite is configured (vite.config.ts) to proxy /api/* to
http://localhost:8000, so the React UI talks to the gateway
transparently. Open http://localhost:5173 in a browser.
"""
import os
import subprocess
from pathlib import Path

REPO = Path(r"F:\IR project")
UI = REPO / "services" / "ui"
LOG = REPO / "data" / "dev_ui.log"
ERR = REPO / "data" / "dev_ui.err.log"
PID = REPO / "data" / "dev_ui.pid"

(UI / "node_modules" / ".vite").mkdir(parents=True, exist_ok=True)

env = os.environ.copy()
env["PATH"] = (
    r"C:\Program Files\nodejs;C:\Program Files\Git\cmd;"
    + env.get("PATH", "")
)
env["NODE_ENV"] = "development"

flags = (
    subprocess.DETACHED_PROCESS
    | subprocess.CREATE_NO_WINDOW
    | subprocess.CREATE_NEW_PROCESS_GROUP
)

# Use npm.cmd + "run dev" so we get the script from package.json
# exactly as documented in the Makefile. node_modules/.bin/vite.cmd
# works too, but the npm script path is the canonical one.
npm = Path(r"C:\Program Files\nodejs\npm.cmd")
if not npm.exists():
    raise SystemExit("npm.cmd not found at C:\\Program Files\\nodejs\\")

with open(LOG, "wb") as out, open(ERR, "wb") as err:
    proc = subprocess.Popen(
        [str(npm), "run", "dev", "--",
         "--port", "5173", "--strictPort", "--host", "127.0.0.1"],
        cwd=str(UI),
        env=env,
        stdout=out,
        stderr=err,
        stdin=subprocess.DEVNULL,
        creationflags=flags,
        close_fds=True,
    )

PID.write_text(str(proc.pid))
print(f"vite launched: pid={proc.pid}, log={LOG}")
print("Open http://localhost:5173 in a browser.")
