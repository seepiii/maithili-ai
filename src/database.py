import sqlite3
import os
from src.config import settings

def get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(os.path.join(settings.data_dir, "maithili.db"))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn

def init_db():
    os.makedirs(settings.data_dir, exist_ok=True)
    with get_db() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                query TEXT NOT NULL,
                response TEXT NOT NULL,
                retrieved_doc_ids TEXT DEFAULT '[]',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS corrections (
                id TEXT PRIMARY KEY,
                conversation_id TEXT NOT NULL REFERENCES conversations(id),
                wrong_response TEXT NOT NULL,
                correct_response TEXT NOT NULL,
                corrected_by TEXT NOT NULL,
                explanation TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS contributions (
                id TEXT PRIMARY KEY,
                text_maithili TEXT NOT NULL,
                text_english TEXT,
                text_transliteration TEXT,
                contributor TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );
        """)
        try:
            conn.execute("ALTER TABLE corrections ADD COLUMN explanation TEXT")
        except sqlite3.OperationalError:
            pass
