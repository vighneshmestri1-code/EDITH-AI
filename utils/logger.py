import logging
import sys
from logging.handlers import RotatingFileHandler

from config.settings import settings


LOG_FILE_NAME = "edith.log"
MAX_LOG_FILE_BYTES = 5 * 1024 * 1024
LOG_BACKUP_COUNT = 3


def _resolve_log_level(
    level: str | None,
) -> int:
    """
    Resolve a configured logging level safely.

    Invalid or unsupported values fall back to INFO.
    """

    requested_level = (
        level
        or settings.LOG_LEVEL
        or "INFO"
    )

    normalized_level = str(
        requested_level
    ).strip().upper()

    level_map = logging.getLevelNamesMapping()

    resolved_level = level_map.get(
        normalized_level
    )

    if isinstance(
        resolved_level,
        int,
    ):
        return resolved_level

    return logging.INFO


def _build_formatter() -> logging.Formatter:
    """Create the standard EDITH log formatter."""

    return logging.Formatter(
        "%(asctime)s | [%(levelname)s] | "
        "%(name)s : %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _close_handlers(
    logger: logging.Logger,
) -> None:
    """
    Close and remove handlers directly attached to
    the supplied logger.
    """

    for handler in logger.handlers[:]:

        try:
            handler.flush()

        except Exception:
            pass

        try:
            handler.close()

        except Exception:
            pass

        logger.removeHandler(
            handler
        )


def setup_logger(
    name: str = "EDITH",
    level: str | None = None,
) -> logging.Logger:
    """
    Configure and return the EDITH application logger.

    Logs are written to:
    - stdout
    - logs/edith.log

    The log file uses rotation to prevent unlimited growth.
    """

    logger = logging.getLogger(
        name
    )

    logger.setLevel(
        _resolve_log_level(
            level
        )
    )

    # Prevent EDITH messages from being handled again
    # by the root logger.
    logger.propagate = False

    # Reconfigure only handlers directly attached to EDITH.
    _close_handlers(
        logger
    )

    formatter = _build_formatter()

    # Console logging
    console_handler = logging.StreamHandler(
        sys.stdout
    )

    console_handler.setLevel(
        logging.DEBUG
    )

    console_handler.setFormatter(
        formatter
    )

    logger.addHandler(
        console_handler
    )

    # Rotating file logging
    try:
        settings.LOGS_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        log_file = (
            settings.LOGS_DIR
            / LOG_FILE_NAME
        )

        file_handler = RotatingFileHandler(
            filename=str(log_file),
            maxBytes=MAX_LOG_FILE_BYTES,
            backupCount=LOG_BACKUP_COUNT,
            encoding="utf-8",
            delay=True,
        )

        file_handler.setLevel(
            logging.DEBUG
        )

        file_handler.setFormatter(
            formatter
        )

        logger.addHandler(
            file_handler
        )

    except Exception as e:

        # Console logging remains available even when
        # the rotating file handler cannot be initialized.
        logger.warning(
            f"Failed to setup file logging: {e}"
        )

    return logger


logger = setup_logger()