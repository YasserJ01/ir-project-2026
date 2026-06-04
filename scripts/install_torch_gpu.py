#!/usr/bin/env python3
"""Install the GPU torch wheel from data/downloads/ and verify the install.

Run AFTER the curl download completes. Idempotent: re-running is safe.
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path("F:/IR project")
VENV_PY = ROOT / ".venv" / "Scripts" / "python.exe"
WHEEL = ROOT / "data" / "downloads" / "torch-2.5.1+cu121-cp312-cp312-win_amd64.whl"


def run(cmd: list[str], **kw) -> int:
    print(f"\n>>> {' '.join(cmd)}", flush=True)
    return subprocess.call(cmd, **kw)


def main() -> int:
    if not WHEEL.exists():
        print(f"ERROR: wheel not found at {WHEEL}")
        print("Run scripts/download_torch_curl.py first (or wait for it to finish).")
        return 1

    size_mb = WHEEL.stat().st_size / 1e6
    print(f"wheel: {WHEEL.name} ({size_mb:.1f} MB)")

    # 1. Uninstall the CPU torch.
    if run([str(VENV_PY), "-m", "pip", "uninstall", "-y", "torch"]) != 0:
        print("ERROR: failed to uninstall CPU torch")
        return 1

    # 2. Install the GPU wheel locally (no deps; the other packages are
    #    already installed at the right versions).
    if (
        run(
            [
                str(VENV_PY),
                "-m",
                "pip",
                "install",
                str(WHEEL),
                "--no-deps",
                "--no-index",  # local wheel only, do not hit PyPI
            ]
        )
        != 0
    ):
        print("ERROR: failed to install GPU torch")
        return 1

    # 3. Verify.
    print("\n=== Verify ===", flush=True)
    return run(
        [
            str(VENV_PY),
            "-c",
            "import torch; print('torch=', torch.__version__, 'cuda=', torch.cuda.is_available(), 'device=', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')",
        ]
    )


if __name__ == "__main__":
    sys.exit(main())
