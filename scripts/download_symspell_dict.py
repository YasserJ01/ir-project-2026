#!/usr/bin/env python
"""Download the SymSpell English frequency dictionary.

The SymSpell library needs a word-frequency file. We use the official
``frequency_dictionary_en_82_765.txt`` (~1.3 MB, 82,765 entries) hosted
on Wolf Garbe's SymSpell GitHub repo.

This is a one-time download -- the dict is small enough to live in
the repo's ``data/dicts/`` directory (gitignored). The download is
idempotent: re-running just confirms the file is present and
reports its size.

Usage:
    python scripts/download_symspell_dict.py
"""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

DICT_URL = "https://raw.githubusercontent.com/wolfgarbe/SymSpell/master/SymSpell/frequency_dictionary_en_82_765.txt"

# Project root = parent of this script.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DICT_PATH = PROJECT_ROOT / "data" / "dicts" / "frequency_dictionary_en_82_765.txt"


def main() -> int:
    DICT_PATH.parent.mkdir(parents=True, exist_ok=True)
    if DICT_PATH.exists():
        size_kb = DICT_PATH.stat().st_size / 1024
        print(f"OK  Dictionary already present at {DICT_PATH} ({size_kb:.1f} KB)")
        return 0

    print(f"Downloading {DICT_URL} -> {DICT_PATH}")
    try:
        # ``urlopen`` over stdlib so we don't need a `requests` dep for
        # a 1.3 MB file. 30 s timeout -- on a 4 Mbps line, the worst
        # case is ~3 s, so this is plenty of slack.
        with urllib.request.urlopen(DICT_URL, timeout=30) as resp:
            data = resp.read()
    except Exception as exc:
        print(f"FAIL  Download failed: {exc}", file=sys.stderr)
        return 1

    DICT_PATH.write_bytes(data)
    size_kb = len(data) / 1024
    print(f"OK  Wrote {size_kb:.1f} KB to {DICT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
