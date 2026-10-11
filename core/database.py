"""Our own small tables: chat titles for the sidebar, and saved settings.

The messages themselves live in LangGraph's checkpoint tables (see graph.py), in the same SQLite file.
API keys are never stored here.
"""

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone

from config import DB_PATH


@contextmanager
def _connect():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def init_db() -> None:
    with _connect() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS chats (
                thread_id  TEXT PRIMARY KEY,
                title      TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS settings (
                key   TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            """
        )


def create_chat(thread_id: str, title: str) -> None:
    now = _now()
    with _connect() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO chats (thread_id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
            (thread_id, title, now, now),
        )


def touch_chat(thread_id: str) -> None:
    with _connect() as conn:
        conn.execute("UPDATE chats SET updated_at = ? WHERE thread_id = ?", (_now(), thread_id))


def list_chats() -> list[dict]:
    with _connect() as conn:
        rows = conn.execute("SELECT * FROM chats ORDER BY updated_at DESC").fetchall()
    return [dict(row) for row in rows]


def get_chat(thread_id: str) -> dict | None:
    with _connect() as conn:
        row = conn.execute("SELECT * FROM chats WHERE thread_id = ?", (thread_id,)).fetchone()
    return dict(row) if row else None


def delete_chat(thread_id: str) -> None:
    with _connect() as conn:
        conn.execute("DELETE FROM chats WHERE thread_id = ?", (thread_id,))


def load_settings() -> dict:
    with _connect() as conn:
        rows = conn.execute("SELECT key, value FROM settings").fetchall()
    return {row["key"]: json.loads(row["value"]) for row in rows}


def save_settings(settings: dict) -> None:
    with _connect() as conn:
        conn.executemany(
            "INSERT INTO settings (key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            [(key, json.dumps(value)) for key, value in settings.items()],
        )
