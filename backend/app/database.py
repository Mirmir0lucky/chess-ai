import sqlite3
from pathlib import Path
from contextlib import contextmanager

ROOT = Path(__file__).resolve().parents[2]
DB_PATH = ROOT / "data" / "arena.sqlite3"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

@contextmanager
def connection():
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()

def init_db():
    with connection() as c:
        c.executescript('''
        CREATE TABLE IF NOT EXISTS games (
          id INTEGER PRIMARY KEY AUTOINCREMENT, started_at TEXT NOT NULL, ended_at TEXT,
          result TEXT, termination TEXT, moves_count INTEGER DEFAULT 0, duration_seconds REAL DEFAULT 0,
          start_fen TEXT NOT NULL, pgn TEXT, white_version INTEGER DEFAULT 1, black_version INTEGER DEFAULT 1
        );
        CREATE TABLE IF NOT EXISTS training_runs (
          id INTEGER PRIMARY KEY AUTOINCREMENT, started_at TEXT NOT NULL, ended_at TEXT,
          status TEXT NOT NULL, episodes INTEGER DEFAULT 0, last_error TEXT
        );
        CREATE TABLE IF NOT EXISTS training_metrics (
          id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT NOT NULL, episode INTEGER,
          white_loss REAL, black_loss REAL, white_win_rate REAL, black_win_rate REAL, draw_rate REAL
        );
        CREATE TABLE IF NOT EXISTS checkpoints (
          id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT NOT NULL, color TEXT NOT NULL,
          version INTEGER NOT NULL, path TEXT NOT NULL, steps INTEGER DEFAULT 0
        );
        ''')
