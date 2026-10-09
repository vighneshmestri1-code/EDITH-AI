import sqlite3
from typing import List, Optional, Tuple

from config.settings import settings
from utils.logger import logger


class DatabaseManager:
    """
    SQLite database manager for EDITH's legacy/general
    chat logging and long-term key-value memory.

    Note:
        Conversation memory used by the current orchestrator
        is handled separately by database.memory.py.
    """

    def __init__(
        self,
        db_path: str | None = None,
    ):
        self.db_path = str(
            db_path or settings.DATABASE_PATH
        )

        database_path = settings.DATABASE_PATH

        if db_path:
            database_path = settings.DATABASE_PATH.__class__(
                db_path
            )

        database_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self._init_db()

    def get_connection(self) -> sqlite3.Connection:
        """Create a SQLite connection."""
        return sqlite3.connect(
            self.db_path,
            timeout=10.0,
        )

    def _init_db(self) -> None:
        """
        Initialize chat logging and long-term
        memory tables if they do not already exist.
        """

        try:
            with self.get_connection() as conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS chat_logs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        role TEXT NOT NULL,
                        content TEXT NOT NULL,
                        timestamp DATETIME
                            DEFAULT CURRENT_TIMESTAMP
                    )
                    """
                )

                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS long_term_memory (
                        key TEXT PRIMARY KEY,
                        value TEXT NOT NULL,
                        updated_at DATETIME
                            DEFAULT CURRENT_TIMESTAMP
                    )
                    """
                )

                conn.commit()

            logger.info(
                "DatabaseManager tables initialized successfully."
            )

        except Exception as e:
            logger.error(
                f"Failed to initialize DatabaseManager: {e}",
                exc_info=True,
            )
            raise

    def save_chat_message(
        self,
        role: str,
        content: str,
    ) -> bool:
        """Save a message to the general chat log."""

        if not role or not role.strip():
            logger.warning(
                "save_chat_message rejected empty role."
            )
            return False

        if content is None:
            logger.warning(
                "save_chat_message rejected empty content."
            )
            return False

        try:
            with self.get_connection() as conn:
                conn.execute(
                    """
                    INSERT INTO chat_logs
                        (role, content)
                    VALUES
                        (?, ?)
                    """,
                    (
                        role.strip(),
                        str(content),
                    ),
                )

                conn.commit()

            return True

        except Exception as e:
            logger.error(
                f"Failed to save chat message: {e}",
                exc_info=True,
            )
            return False

    def set_memory_fact(
        self,
        key: str,
        value: str,
    ) -> bool:
        """Create or update a long-term memory fact."""

        if not key or not key.strip():
            logger.warning(
                "set_memory_fact rejected empty key."
            )
            return False

        if value is None:
            logger.warning(
                "set_memory_fact rejected empty value."
            )
            return False

        try:
            with self.get_connection() as conn:
                conn.execute(
                    """
                    INSERT INTO long_term_memory
                        (key, value)
                    VALUES
                        (?, ?)
                    ON CONFLICT(key)
                    DO UPDATE SET
                        value = excluded.value,
                        updated_at = CURRENT_TIMESTAMP
                    """,
                    (
                        key.strip(),
                        str(value),
                    ),
                )

                conn.commit()

            logger.info(
                f"Long-term memory updated: {key.strip()}"
            )

            return True

        except Exception as e:
            logger.error(
                f"Failed to save memory fact "
                f"'{key}': {e}",
                exc_info=True,
            )
            return False

    def get_memory_fact(
        self,
        key: str,
    ) -> Optional[str]:
        """Retrieve a single long-term memory fact."""

        if not key or not key.strip():
            return None

        try:
            with self.get_connection() as conn:
                cursor = conn.execute(
                    """
                    SELECT value
                    FROM long_term_memory
                    WHERE key = ?
                    """,
                    (key.strip(),),
                )

                row = cursor.fetchone()

            return row[0] if row else None

        except Exception as e:
            logger.error(
                f"Failed to retrieve memory fact "
                f"'{key}': {e}",
                exc_info=True,
            )
            return None

    def get_all_memory_facts(
        self,
    ) -> List[Tuple[str, str]]:
        """Return all stored long-term memory facts."""

        try:
            with self.get_connection() as conn:
                cursor = conn.execute(
                    """
                    SELECT key, value
                    FROM long_term_memory
                    ORDER BY key ASC
                    """
                )

                return cursor.fetchall()

        except Exception as e:
            logger.error(
                f"Failed to retrieve all memory facts: {e}",
                exc_info=True,
            )
            return []