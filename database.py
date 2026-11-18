"""
NewsLens - SQLite Database Manager
Author: B.Tech AI & Data Science Portfolio Project

This module provides persistent local storage for user news detection queries.
Features:
- Thread-safe SQLite operations for Streamlit integration
- Automated database and table initialization
- Safe parameterized queries to prevent SQL injection
- Functions to log predictions, retrieve chronological history, get aggregate statistics, and purge logs
"""

import os
import sqlite3
import datetime
from typing import List, Dict, Any, Optional

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
DB_PATH = os.path.join(DATA_DIR, "history.db")


def get_db_connection() -> sqlite3.Connection:
    """
    Creates and returns a thread-safe connection to the SQLite database.
    Ensures the parent data directory exists.
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row  # Return rows as dict-like objects
    return conn


def init_db() -> None:
    """
    Initializes the SQLite database schema if it doesn't already exist.
    Creates the 'prediction_history' table with index on timestamp.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS prediction_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            headline_text TEXT NOT NULL,
            predicted_label TEXT NOT NULL,
            confidence_score REAL NOT NULL,
            word_count INTEGER NOT NULL,
            top_indicators TEXT
        )
    """)
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_history_timestamp 
        ON prediction_history (timestamp DESC)
    """)
    conn.commit()
    conn.close()


def insert_prediction(
    headline_text: str,
    predicted_label: str,
    confidence_score: float,
    word_count: int,
    top_indicators: str = ""
) -> int:
    """
    Inserts a newly analyzed news query record into the database.

    Args:
        headline_text (str): The raw text submitted by the user.
        predicted_label (str): 'Real News' or 'Potentially Misleading / Fake'.
        confidence_score (float): Model confidence percentage (e.g., 94.5).
        word_count (int): Total words in submitted text.
        top_indicators (str): Comma-separated top predictive tokens.

    Returns:
        int: The primary key ID of the inserted record.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Sanitize and truncate excessively huge raw text to prevent disk exhaustion
    trimmed_text = headline_text.strip()
    if len(trimmed_text) > 10000:
        trimmed_text = trimmed_text[:10000] + "... [truncated]"

    cursor.execute(
        """
        INSERT INTO prediction_history 
        (timestamp, headline_text, predicted_label, confidence_score, word_count, top_indicators)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (now_str, trimmed_text, predicted_label, round(confidence_score, 2), word_count, top_indicators)
    )
    conn.commit()
    inserted_id = cursor.lastrowid
    conn.close()
    return inserted_id


def get_history(limit: int = 100) -> List[Dict[str, Any]]:
    """
    Fetches past prediction records ordered by latest timestamp first.

    Args:
        limit (int): Maximum number of recent records to return.

    Returns:
        List[Dict[str, Any]]: List of dictionary representations of records.
    """
    init_db()
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT id, timestamp, headline_text, predicted_label, confidence_score, word_count, top_indicators
        FROM prediction_history
        ORDER BY id DESC
        LIMIT ?
        """,
        (limit,)
    )
    rows = cursor.fetchall()
    conn.close()

    return [dict(row) for row in rows]


def delete_prediction(record_id: int) -> bool:
    """
    Deletes a single prediction record by ID.

    Args:
        record_id (int): ID of the record to delete.

    Returns:
        bool: True if a record was deleted, False otherwise.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM prediction_history WHERE id = ?", (record_id,))
    deleted = cursor.rowcount > 0
    conn.commit()
    conn.close()
    return deleted


def clear_history() -> bool:
    """
    Clears all records from the prediction_history table.

    Returns:
        bool: True if successful.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM prediction_history")
    conn.commit()
    conn.close()
    return True


def get_summary_stats() -> Dict[str, Any]:
    """
    Computes summary metrics across all user analyses logged in SQLite.

    Returns:
        Dict[str, Any]: Total predictions, count of real vs fake, and average confidence.
    """
    init_db()
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM prediction_history")
    total_count = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM prediction_history WHERE predicted_label LIKE '%Real%'")
    real_count = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM prediction_history WHERE predicted_label LIKE '%Fake%' OR predicted_label LIKE '%Misleading%'")
    fake_count = cursor.fetchone()[0]

    cursor.execute("SELECT AVG(confidence_score) FROM prediction_history")
    avg_conf_row = cursor.fetchone()[0]
    avg_conf = round(avg_conf_row, 2) if avg_conf_row is not None else 0.0

    conn.close()

    return {
        "total_analyzed": total_count,
        "real_count": real_count,
        "fake_count": fake_count,
        "avg_confidence": avg_conf
    }


if __name__ == "__main__":
    # Self-test unit verification
    print("[Database] Initializing test...")
    init_db()
    test_id = insert_prediction(
        headline_text="NASA confirms discovery of new exoplanet with water vapor.",
        predicted_label="Real News",
        confidence_score=96.4,
        word_count=10,
        top_indicators="nasa, exoplanet, discov"
    )
    print(f"[Database] Inserted sample test record ID: {test_id}")
    history = get_history(limit=5)
    print(f"[Database] Records retrieved: {len(history)}")
    stats = get_summary_stats()
    print(f"[Database] Summary stats: {stats}")
    print("[Database] Verification passed!")
