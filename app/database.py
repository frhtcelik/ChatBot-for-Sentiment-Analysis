"""SQLite persistence for chat and batch analyses."""

from __future__ import annotations

import csv
import sqlite3
from pathlib import Path
from typing import Iterable, Optional

from app.sentiment import SentimentResult


DEFAULT_DB_PATH = Path("data/sentiment_chatbot.sqlite3")


def get_connection(db_path: Path | str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    return connection


def init_db(db_path: Path | str = DEFAULT_DB_PATH) -> None:
    with get_connection(db_path) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_message TEXT NOT NULL,
                bot_reply TEXT NOT NULL,
                label TEXT NOT NULL,
                score REAL NOT NULL,
                subjectivity REAL NOT NULL,
                confidence REAL NOT NULL,
                model_name TEXT NOT NULL DEFAULT 'hybrid-textblob-lexicon',
                source TEXT NOT NULL DEFAULT 'chat',
                feedback TEXT,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        # Migration for existing databases
        try:
            connection.execute("ALTER TABLE messages ADD COLUMN feedback TEXT")
        except sqlite3.OperationalError:
            # Column already exists
            pass

        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_messages_created_at
            ON messages(created_at)
            """
        )


def save_message(
    user_message: str,
    bot_reply: str,
    result: SentimentResult,
    source: str = "chat",
    db_path: Path | str = DEFAULT_DB_PATH,
) -> int:
    init_db(db_path)
    with get_connection(db_path) as connection:
        cursor = connection.execute(
            """
            INSERT INTO messages (
                user_message,
                bot_reply,
                label,
                score,
                subjectivity,
                confidence,
                source
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                user_message,
                bot_reply,
                result.label,
                result.score,
                result.subjectivity,
                result.confidence,
                source,
            ),
        )
        return int(cursor.lastrowid)


def update_feedback(
    message_id: int,
    feedback: str,
    db_path: Path | str = DEFAULT_DB_PATH,
) -> None:
    init_db(db_path)
    with get_connection(db_path) as connection:
        connection.execute(
            "UPDATE messages SET feedback = ? WHERE id = ?",
            (feedback, message_id),
        )


def fetch_messages(
    limit: Optional[int] = 200,
    source: Optional[str] = None,
    db_path: Path | str = DEFAULT_DB_PATH,
) -> list[dict[str, object]]:
    init_db(db_path)
    params: list[object] = []
    query = "SELECT * FROM messages"

    if source:
        query += " WHERE source = ?"
        params.append(source)

    query += " ORDER BY datetime(created_at) DESC, id DESC"

    if limit is not None:
        query += " LIMIT ?"
        params.append(limit)

    with get_connection(db_path) as connection:
        return [dict(row) for row in connection.execute(query, params).fetchall()]


def clear_messages(source: Optional[str] = None, db_path: Path | str = DEFAULT_DB_PATH) -> None:
    init_db(db_path)
    with get_connection(db_path) as connection:
        if source:
            connection.execute("DELETE FROM messages WHERE source = ?", (source,))
        else:
            connection.execute("DELETE FROM messages")


def save_batch(
    rows: Iterable[tuple[str, str, SentimentResult]],
    db_path: Path | str = DEFAULT_DB_PATH,
) -> int:
    count = 0
    for text, reply, result in rows:
        save_message(text, reply, result, source="batch", db_path=db_path)
        count += 1
    return count


def add_to_dataset(text: str, correct_label: str, dataset_path: str = "data/turkish_sentiment_dataset.csv") -> None:
    """Append a corrected sample to the CSV training dataset."""
    path = Path(dataset_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    write_header = not path.exists() or path.stat().st_size == 0
    cleaned_text = text.replace("\r", " ").replace("\n", " ").strip()
    
    with open(path, "a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["text", "label"])
        writer.writerow([cleaned_text, correct_label])

