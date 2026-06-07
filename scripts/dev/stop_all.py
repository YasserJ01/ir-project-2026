"""Stop all dev services launched by scripts/dev/launch_*.py.

Reads each data/dev_*.pid, kills the PID + its children via psutil
(uvicorn spawns the worker as a child on Windows), then removes the
PID file. Safe to re-run.
"""
import sys
from pathlib import Path

REPO = Path(r"F:\IR project")
try:
    import psutil
except ImportError:
    sys.exit("psutil not in venv. Run: .venv\\Scripts\\python -m pip install psutil")

KILLED = 0
for pid_file in sorted(REPO.glob("data/dev_*.pid")):
    pid = int(pid_file.read_text().strip())
    try:
        parent = psutil.Process(pid)
        name = parent.name()
    except psutil.NoSuchProcess:
        print(f"  {pid_file.stem}: pid={pid} already gone")
        pid_file.unlink(missing_ok=True)
        continue

    # uvicorn --reload-less mode is a single process; --reload forks a
    # child watcher. We kill the whole tree to be safe.
    children = parent.children(recursive=True)
    for c in children:
        try:
            c.terminate()
        except psutil.NoSuchProcess:
            pass
    try:
        parent.terminate()
    except psutil.NoSuchProcess:
        pass
    gone, alive = psutil.wait_procs([parent, *children], timeout=5)
    for p in alive:
        try:
            p.kill()
        except psutil.NoSuchProcess:
            pass
    KILLED += 1
    print(f"  {pid_file.stem}: killed pid={pid} ({name}) + {len(children)} child(ren)")
    pid_file.unlink(missing_ok=True)

if KILLED == 0:
    print("nothing to stop (no dev_*.pid files)")
else:
    print(f"stopped {KILLED} service(s)")
