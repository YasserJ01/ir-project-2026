"""Quick build progress reporter (Phase 6).

Reports:
- elapsed time since build started
- per-service current step + most recent log line
- list of completed images
- per-service step (1-11) extracted from "#NN [service  X/11]" headers
"""
from __future__ import annotations

import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# Force UTF-8 on Windows so em-dashes / arrows in build logs don't crash.
try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
except Exception:
    pass


def parse_log(log: Path) -> dict[str, dict[str, str]]:
    """Return {service: {step, last_line, last_time}} for each service."""
    services = ["preprocessing", "indexing", "retrieval", "refinement"]
    out = {s: {"step": "(not started)", "last_line": "", "last_time": "0"} for s in services}
    if not log.exists():
        return out
    text = log.read_text(encoding="utf-8", errors="replace")
    # Pattern: #NN [service_name  X/11] COMMAND...
    header_re = re.compile(r"^#(\d+)\s*\[(\w+)\s+(\d+)/(\d+)\]\s*(.+?)$")
    for line in text.splitlines():
        m = header_re.match(line)
        if not m:
            continue
        step_num, svc, num, denom, cmd = m.groups()
        if svc in out:
            # Find the latest time stamp on this line, e.g. "52.3"
            t_match = re.search(r"(\d+(?:\.\d+)?)s\b", line.split(svc, 1)[0])
            t = t_match.group(1) if t_match else "0"
            out[svc] = {
                "step": f"{num}/{denom}",
                "last_line": cmd[:120],
                "last_time": t,
            }
    return out


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    # Pick the most recent backend build log (parallel or serial).
    candidates = [
        repo_root / "data" / "build_backend_4_serial.log",
        repo_root / "data" / "build_backend_4.log",
    ]
    log = next((p for p in candidates if p.exists()), candidates[0])
    err_log = log.with_suffix(".err.log")

    now = datetime.now()
    print(f"=== Build progress report — {now.strftime('%H:%M:%S')} ===\n")

    if not log.exists():
        print("No build log yet.")
        return 0

    log_mtime = datetime.fromtimestamp(log.stat().st_mtime)
    age = (now - log_mtime).total_seconds()
    print(f"Log last updated: {age:.0f} s ago ({log_mtime.strftime('%H:%M:%S')})")
    print(f"Log size: {log.stat().st_size:,} bytes\n")

    services = parse_log(log)
    print("Per-service progress:")
    print(f"  {'service':<14} {'step':<10} {'elapsed':<10} command")
    print(f"  {'-'*14} {'-'*10} {'-'*10} {'-'*60}")
    for svc, info in services.items():
        print(f"  {svc:<14} {info['step']:<10} {info['last_time']+'s':<10} {info['last_line'][:60]}")

    if err_log.exists() and err_log.stat().st_size > 0:
        print(f"\nerr.log: {err_log.stat().st_size} bytes")
        text = err_log.read_text(encoding="utf-8", errors="replace")
        for line in text.splitlines()[-5:]:
            print(f"  {line}")

    print("\n=== Built images so far ===")
    r = subprocess.run(
        ["docker", "images", "--format", "{{.Repository}}:{{.Tag}}  {{.Size}}  {{.CreatedAt}}"],
        capture_output=True, text=True,
    )
    for line in r.stdout.splitlines():
        if "ir-project" in line.lower():
            print(f"  {line}")

    print("\n=== Last 5 log lines ===")
    text = log.read_text(encoding="utf-8", errors="replace")
    for line in text.splitlines()[-5:]:
        print(f"  {line}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
