import asyncio
import sqlite3
from typing import Any, Dict, List

from config.settings import settings
from utils.logger import logger


DB_PATH = str(settings.DATABASE_PATH)


def _get_connection() -> sqlite3.Connection:
    """
    Create and configure a SQLite connection.

    WAL mode allows better read/write concurrency, while NORMAL
    synchronous mode provides a good balance between performance
    and durability.
    """

    conn = sqlite3.connect(
        DB_PATH,
        check_same_thread=False,
        timeout=10.0,
    )

    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")

    return conn


def init_db() -> None:
    """
    Initialize the EDITH database and required indexes.
    """

    try:
        settings.DATABASE_PATH.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with _get_connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_messages_session_id
                ON messages (session_id)
                """
            )

            conn.commit()

        logger.info(
            f"Database initialized at {DB_PATH}"
        )

    except Exception as e:
        logger.error(
            f"Failed to initialize SQLite database: {e}",
            exc_info=True,
        )
        raise


def save_message(
    session_id: str,
    role: str,
    content: str,
) -> bool:
    """
    Save a conversation message to SQLite.

    Returns:
        True if the message was saved successfully.
        False if the operation failed.
    """

    if not session_id or not session_id.strip():
        logger.warning(
            "Attempted to save message with empty session ID."
        )
        return False

    if not role or not role.strip():
        logger.warning(
            "Attempted to save message with empty role."
        )
        return False

    if content is None:
        logger.warning(
            "Attempted to save a message with None content."
        )
        return False

    try:
        with _get_connection() as conn:
            conn.execute(
                """
                INSERT INTO messages (
                    session_id,
                    role,
                    content
                )
                VALUES (?, ?, ?)
                """,
                (
                    session_id.strip(),
                    role.strip(),
                    str(content),
                ),
            )

            conn.commit()

        return True

    except Exception as e:
        logger.error(
            f"Error saving message to database: {e}",
            exc_info=True,
        )
        return False


async def save_message_async(
    session_id: str,
    role: str,
    content: str,
) -> bool:
    """
    Asynchronously save a message without blocking the event loop.
    """

    return await asyncio.to_thread(
        save_message,
        session_id,
        role,
        content,
    )


def get_chat_history(
    session_id: str,
    limit: int = 10,
) -> List[Dict[str, str]]:
    """
    Retrieve the most recent messages for a session.

    Messages are returned in chronological order.
    """

    if not session_id or not session_id.strip():
        return []

    if limit <= 0:
        return []

    try:
        with _get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT role, content
                FROM (
                    SELECT
                        id,
                        role,
                        content
                    FROM messages
                    WHERE session_id = ?
                    ORDER BY id DESC
                    LIMIT ?
                )
                ORDER BY id ASC
                """,
                (
                    session_id.strip(),
                    limit,
                ),
            )

            rows = cursor.fetchall()

        return [
            {
                "role": row[0],
                "content": row[1],
            }
            for row in rows
        ]

    except Exception as e:
        logger.error(
            f"Error fetching chat history for "
            f"session '{session_id}': {e}",
            exc_info=True,
        )
        return []


async def get_chat_history_async(
    session_id: str,
    limit: int = 10,
) -> List[Dict[str, str]]:
    """
    Asynchronously retrieve chat history without blocking
    the event loop.
    """

    return await asyncio.to_thread(
        get_chat_history,
        session_id,
        limit,
    )


def clear_session_history(
    session_id: str,
) -> bool:
    """
    Delete all stored messages for a specific session.

    Returns:
        True if the operation succeeds.
        False if it fails.
    """

    if not session_id or not session_id.strip():
        return False

    try:
        with _get_connection() as conn:
            conn.execute(
                """
                DELETE FROM messages
                WHERE session_id = ?
                """,
                (session_id.strip(),),
            )

            conn.commit()

        logger.info(
            f"Cleared history for session: {session_id}"
        )

        return True

    except Exception as e:
        logger.error(
            f"Error clearing session history: {e}",
            exc_info=True,
        )
        return False