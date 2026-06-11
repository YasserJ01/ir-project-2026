"""SQLite-based document store.

Replaces the runtime dependency on ``ir_datasets.docs_store()``.
Each dataset gets its own SQLite file at ``data/dbs/{dataset_id}.db``
with schema::

    CREATE TABLE documents (doc_id TEXT PRIMARY KEY, text TEXT)

Usage (build)::

    from shared.ir_common.doc_db import create_db, insert_docs
    conn = create_db("touche2020")
    insert_docs(conn, [("doc-1", "full text..."), ...])
    conn.close()

Usage (runtime)::

    from shared.ir_common.doc_db import get_doc
    text = get_doc("touche2020", "doc-1")
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
DB_DIR: Path = _PROJECT_ROOT / "data" / "dbs"


def db_path(dataset_id: str) -> Path:
    """Return the path to the SQLite file for a dataset."""
    return DB_DIR / f"{dataset_id}.db"


def create_db(dataset_id: str) -> sqlite3.Connection:
    """Create (or open) the SQLite database for *dataset_id* and ensure
    the ``documents`` table exists.

    The caller is responsible for closing the connection.
    """
    DB_DIR.mkdir(parents=True, exist_ok=True)
    path = db_path(dataset_id)
    conn = sqlite3.connect(str(path))
    conn.execute(
        "CREATE TABLE IF NOT EXISTS documents (doc_id TEXT PRIMARY KEY, text TEXT)"
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_doc_id ON documents(doc_id)")
    conn.commit()
    return conn


def insert_docs(conn: sqlite3.Connection, docs: list[tuple[str, str]]) -> None:
    """Batch-insert ``(doc_id, text)`` pairs.

    Uses ``INSERT OR REPLACE`` so re-running the migration is idempotent.
    """
    conn.executemany(
        "INSERT OR REPLACE INTO documents (doc_id, text) VALUES (?, ?)", docs
    )
    conn.commit()


def get_doc(dataset_id: str, doc_id: str) -> str | None:
    """Return the original document text, or ``None`` if not found."""
    path = db_path(dataset_id)
    if not path.exists():
        return None
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    try:
        row = conn.execute(
            "SELECT text FROM documents WHERE doc_id = ?", (doc_id,)
        ).fetchone()
        return row[0] if row else None
    finally:
        conn.close()
