"""Launch all 5 dev services in dependency order, poll /health for
each, and print a final summary. Survives the opencode 120s shell
timeout because each launcher script uses a fully-detached Popen.

Order:
  1. indexing     :8002  (needed for bm25/tfidf)
  2. retrieval    :8003  (needed for dense/hybrid/multi_encoder; slowest start ~10-30s)
  3. refinement   :8004  (needed for with_features mode)
  4. preprocessing:8001  (needed only for /health to be 'ok')
  5. gateway      :8000  (depends on all 4)
  6. ui           :5173  (Vite; depends on gateway being reachable)

Usage:
  py -3.12 scripts\\dev\\launch_all.py
  py -3.12 scripts\\dev\\stop_all.py    # to stop everything
"""
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

REPO = Path(r"F:\IR project")
PY = sys.executable

SERVICES = [
    ("preprocessing", "scripts/dev/launch_preprocessing.py", 8001),
    ("indexing",      "scripts/dev/launch_indexing.py",      8002),
    ("retrieval",     "scripts/dev/launch_retrieval.py",     8003),
    ("refinement",    "scripts/dev/launch_refinement.py",    8004),
    ("gateway",       "scripts/dev/launch_gateway.py",       8000),
    ("ui",            "scripts/dev/launch_ui.py",            5173),
]


def wait_healthy(name: str, port: int, timeout_s: float = 90.0) -> bool:
    url = f"http://localhost:{port}/health"
    deadline = time.monotonic() + timeout_s
    last_err = None
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as r:
                if r.status == 200:
                    print(f"  [ok]   {name:14s} :{port} responding in "
                          f"{timeout_s - (deadline - time.monotonic()):.1f}s")
                    return True
        except Exception as e:  # noqa: BLE001
            last_err = e
        time.sleep(0.5)
    print(f"  [FAIL] {name:14s} :{port} not reachable after {timeout_s}s: {last_err}")
    return False


def main() -> int:
    started = time.monotonic()
    for name, script, port in SERVICES:
        print(f"\n--- launching {name} (:{port}) ---")
        result = subprocess.run(
            [PY, script], cwd=str(REPO), capture_output=True, text=True
        )
        print(result.stdout.strip())
        if result.returncode != 0:
            print(f"  launcher failed: {result.stderr.strip()}")
            return result.returncode

    # Poll each port. Gateway and UI don't have /health, so we use
    # the index page as a proxy for "port is open".
    print("\n--- waiting for services to be reachable ---")
    ok = True
    for name, _, port in SERVICES:
        # gateway + ui have no /health; any 200/404 response means up.
        if name in ("gateway", "ui"):
            url = f"http://localhost:{port}/" if name == "gateway" \
                  else f"http://localhost:{port}/"
            try:
                with urllib.request.urlopen(url, timeout=2) as r:
                    print(f"  [ok]   {name:14s} :{port} responded {r.status}")
            except Exception as e:  # noqa: BLE001
                print(f"  [FAIL] {name:14s} :{port} not reachable: {e}")
                ok = False
        else:
            if not wait_healthy(name, port):
                ok = False

    elapsed = time.monotonic() - started
    print("\n" + "=" * 60)
    if ok:
        print(f"  ALL SERVICES UP in {elapsed:.1f}s")
        print()
        print("  Open the UI:  http://localhost:5173")
        print("  Gateway:      http://localhost:8000/api/health")
        print()
        print("  Try a search:  dataset=touche2020, query='eiffel tower',")
        print("                 representation=bm25, mode=basic, press Enter.")
        print()
        print("  To stop:       py -3.12 scripts/dev/stop_all.py")
    else:
        print(f"  SOME SERVICES FAILED to start in {elapsed:.1f}s")
        print("  Check logs in data/dev_*.{log,err.log}")
    print("=" * 60)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
