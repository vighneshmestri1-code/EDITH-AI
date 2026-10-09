from pathlib import Path

from config.settings import settings
from tools.registry import register_tool
from utils.logger import logger


BASE_DIR = settings.BASE_DIR.resolve()

MAX_READ_SIZE_MB = 10
MAX_WRITE_SIZE_MB = 2

PROTECTED_FILENAMES = {
    ".env",
    "edith.db",
    "edith.db-wal",
    "edith.db-shm",
}

PROTECTED_DIRECTORIES = {
    ".git",
}


def _resolve_safe_path(path: str) -> Path:
    """
    Resolve a path inside EDITH's project directory.

    Relative paths are resolved from BASE_DIR.
    Absolute paths are only allowed if they are already
    inside BASE_DIR.
    """

    if not path or not str(path).strip():
        raise ValueError(
            "Path cannot be empty."
        )

    requested_path = Path(
        str(path).strip()
    )

    if requested_path.is_absolute():
        resolved_path = requested_path.resolve()
    else:
        resolved_path = (
            BASE_DIR / requested_path
        ).resolve()

    try:
        resolved_path.relative_to(
            BASE_DIR
        )

    except ValueError as e:
        raise PermissionError(
            "Access denied: path is outside "
            "EDITH's working directory."
        ) from e

    return resolved_path


def _is_protected_path(
    filepath: Path,
) -> bool:
    """
    Check whether a path belongs to a protected
    project file or directory.
    """

    try:
        relative_path = filepath.relative_to(
            BASE_DIR
        )

    except ValueError:
        return True

    parts = relative_path.parts

    if not parts:
        return False

    if parts[0] in PROTECTED_DIRECTORIES:
        return True

    filename = parts[-1]

    if (
        filename in PROTECTED_FILENAMES
        or filename.startswith(".env.")
    ):
        return True

    return False


def _validate_file_size(
    filepath: Path,
    max_size_mb: int = MAX_READ_SIZE_MB,
) -> None:
    """Prevent accidentally reading extremely large files."""

    if not filepath.is_file():
        return

    size_mb = filepath.stat().st_size / (
        1024 * 1024
    )

    if size_mb > max_size_mb:
        raise ValueError(
            f"File is too large to read. "
            f"Maximum allowed size is "
            f"{max_size_mb} MB."
        )


def _validate_write_size(
    content: str,
    max_size_mb: int = MAX_WRITE_SIZE_MB,
) -> None:
    """Prevent accidentally writing extremely large files."""

    content_size_mb = len(
        content.encode("utf-8")
    ) / (
        1024 * 1024
    )

    if content_size_mb > max_size_mb:
        raise ValueError(
            f"Content is too large to write. "
            f"Maximum allowed size is "
            f"{max_size_mb} MB."
        )


@register_tool(name="read_file")
def read_file(
    filepath: str,
) -> str:
    """
    Read and return the content of a text file.
    """

    try:
        target_path = _resolve_safe_path(
            filepath
        )

        if _is_protected_path(
            target_path
        ):
            return (
                "Error: Access to this file or "
                "directory is protected."
            )

        if not target_path.exists():
            return (
                f"Error: File '{filepath}' "
                "does not exist."
            )

        if not target_path.is_file():
            return (
                f"Error: '{filepath}' "
                "is not a file."
            )

        _validate_file_size(
            target_path
        )

        with target_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            content = file.read()

        logger.info(
            f"Successfully read file: "
            f"{target_path}"
        )

        return content

    except Exception as e:
        logger.error(
            f"Error reading file "
            f"{filepath}: {e}",
            exc_info=True,
        )

        return (
            f"Error reading file: {str(e)}"
        )


@register_tool(name="write_file")
def write_file(
    filepath: str,
    content: str,
    append: bool = False,
) -> str:
    """
    Write text content to a file.

    Overwrites by default.
    Set append=True to append content.
    """

    if content is None:
        return (
            "Error writing file: "
            "content cannot be empty."
        )

    content = str(content)

    try:
        target_path = _resolve_safe_path(
            filepath
        )

        if _is_protected_path(
            target_path
        ):
            return (
                "Error: Access to this file or "
                "directory is protected."
            )

        _validate_write_size(
            content
        )

        target_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        mode = (
            "a"
            if append
            else "w"
        )

        with target_path.open(
            mode,
            encoding="utf-8",
        ) as file:
            file.write(content)

        action = (
            "appended to"
            if append
            else "written to"
        )

        logger.info(
            f"Successfully {action} file: "
            f"{target_path}"
        )

        return (
            f"Content successfully {action} "
            f"'{filepath}'."
        )

    except Exception as e:
        logger.error(
            f"Error writing to file "
            f"{filepath}: {e}",
            exc_info=True,
        )

        return (
            f"Error writing file: {str(e)}"
        )


@register_tool(name="list_directory")
def list_directory(
    directory_path: str = ".",
) -> str:
    """
    List files and folders inside a directory.
    """

    try:
        target_path = _resolve_safe_path(
            directory_path
        )

        if _is_protected_path(
            target_path
        ):
            return (
                "Error: Access to this directory "
                "is protected."
            )

        if not target_path.exists():
            return (
                f"Error: Directory "
                f"'{directory_path}' "
                "does not exist."
            )

        if not target_path.is_dir():
            return (
                f"Error: '{directory_path}' "
                "is not a directory."
            )

        items = sorted(
            target_path.iterdir(),
            key=lambda item: (
                not item.is_dir(),
                item.name.lower(),
            ),
        )

        if not items:
            return "Directory is empty."

        formatted_list = []

        for item in items:
            item_type = (
                "DIR"
                if item.is_dir()
                else "FILE"
            )

            formatted_list.append(
                f"[{item_type}] "
                f"{item.name}"
            )

        logger.info(
            f"Listed directory: "
            f"{target_path}"
        )

        return "\n".join(
            formatted_list
        )

    except Exception as e:
        logger.error(
            f"Error listing directory "
            f"{directory_path}: {e}",
            exc_info=True,
        )

        return (
            f"Error listing directory: "
            f"{str(e)}"
        )


@register_tool(name="delete_file")
def delete_file(
    filepath: str,
) -> str:
    """
    Delete a file.

    Directory deletion is intentionally blocked
    to prevent accidental recursive deletion.
    """

    try:
        target_path = _resolve_safe_path(
            filepath
        )

        if _is_protected_path(
            target_path
        ):
            return (
                "Error: Access to this file or "
                "directory is protected."
            )

        if not target_path.exists():
            return (
                f"Error: File '{filepath}' "
                "does not exist."
            )

        if target_path.is_dir():
            return (
                f"Error: '{filepath}' is a "
                "directory. Directory deletion "
                "is not allowed."
            )

        target_path.unlink()

        logger.info(
            f"Deleted file: {target_path}"
        )

        return (
            f"File '{filepath}' deleted."
        )

    except Exception as e:
        logger.error(
            f"Error deleting path "
            f"{filepath}: {e}",
            exc_info=True,
        )

        return (
            f"Error deleting file: "
            f"{str(e)}"
        )