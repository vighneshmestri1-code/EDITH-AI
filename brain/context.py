from functools import lru_cache
from typing import Any, Dict, List

from database.memory import (
    get_chat_history,
    get_chat_history_async,
)
from tools.registry import get_tools_description


TOOL_SYSTEM_INSTRUCTION = """
You have access to internal tools.

When a tool is required, output ONLY this raw JSON:

{"tool": "<tool_name>", "args": {"argument": "value"}}

Do not use Markdown code fences.
Do not add explanatory text before or after a tool call.

When no tool is required, answer the user naturally.
""".strip()


def _clean_prompt_part(
    text: str,
) -> str:
    """Normalize a static prompt section."""

    if not text:
        return ""

    return str(text).strip()


@lru_cache(maxsize=8)
def _build_system_prompt_cached(
    system_prompt: str,
    tools_description: str,
) -> str:
    """
    Build and cache the static system prompt.

    The EDITH system prompt and registered tool descriptions
    normally remain unchanged during a process lifetime, so
    rebuilding them for every request is unnecessary.
    """

    sections = [
        _clean_prompt_part(system_prompt),
        TOOL_SYSTEM_INSTRUCTION,
        _clean_prompt_part(tools_description),
    ]

    return "\n\n".join(
        section
        for section in sections
        if section
    )


def _build_system_prompt(
    system_prompt: str,
) -> str:
    """
    Return the cached EDITH system/tool prompt.
    """

    clean_system_prompt = _clean_prompt_part(
        system_prompt
    )

    tools_description = _clean_prompt_part(
        get_tools_description()
    )

    return _build_system_prompt_cached(
        clean_system_prompt,
        tools_description,
    )


def _is_internal_tool_message(
    message: Dict[str, Any],
) -> bool:
    """
    Detect internal tool execution traces that should not
    be replayed to Ollama as normal conversation history.
    """

    role = str(
        message.get("role", "")
    ).strip().lower()

    content = str(
        message.get("content", "")
    ).strip()

    if (
        role == "assistant"
        and content.startswith("[Tool Call:")
    ):
        return True

    if (
        role == "system"
        and content.startswith("Tool Output:")
    ):
        return True

    return False


def _sanitize_history(
    history: List[Dict[str, Any]],
) -> List[Dict[str, str]]:
    """
    Keep only useful conversational history.

    Internal tool traces are removed because they are execution
    metadata, not part of the user's natural conversation.
    """

    sanitized: List[Dict[str, str]] = []

    for message in history:

        if not isinstance(
            message,
            dict,
        ):
            continue

        if _is_internal_tool_message(
            message
        ):
            continue

        role = str(
            message.get("role", "")
        ).strip().lower()

        if role not in {
            "user",
            "assistant",
        }:
            continue

        content = message.get(
            "content"
        )

        if content is None:
            continue

        content = str(
            content
        ).strip()

        if not content:
            continue

        sanitized.append(
            {
                "role": role,
                "content": content,
            }
        )

    return sanitized


def _append_history_and_input(
    messages: List[Dict[str, Any]],
    history: List[Dict[str, Any]],
    current_input: str,
) -> None:
    """
    Add sanitized history and the current user request.

    Prevent duplicate current-user messages.
    """

    clean_history = _sanitize_history(
        history
    )

    messages.extend(
        clean_history
    )

    clean_input = str(
        current_input
    ).strip()

    if not clean_input:
        return

    if (
        not clean_history
        or clean_history[-1].get("role")
        != "user"
        or clean_history[-1].get("content")
        != clean_input
    ):
        messages.append(
            {
                "role": "user",
                "content": clean_input,
            }
        )


def _normalize_limit(
    limit: int,
) -> int:
    """
    Normalize the requested conversation-history limit.
    """

    try:
        return max(
            1,
            int(limit),
        )

    except (TypeError, ValueError):
        return 10


def build_context(
    session_id: str,
    current_input: str,
    system_prompt: str = "",
    limit: int = 10,
) -> List[Dict[str, Any]]:
    """
    Build the context sent to Ollama.

    Includes:
    - EDITH system prompt
    - tool instructions
    - registered tool descriptions
    - useful conversational history
    - current user request
    """

    messages: List[
        Dict[str, Any]
    ] = []

    combined_system_prompt = (
        _build_system_prompt(
            system_prompt
        )
    )

    if combined_system_prompt:
        messages.append(
            {
                "role": "system",
                "content": combined_system_prompt,
            }
        )

    history = get_chat_history(
        session_id=session_id,
        limit=_normalize_limit(limit),
    )

    _append_history_and_input(
        messages,
        history,
        current_input,
    )

    return messages


async def async_build_context(
    session_id: str,
    current_input: str,
    system_prompt: str = "",
    limit: int = 10,
) -> List[Dict[str, Any]]:
    """
    Asynchronous context builder.

    Uses asynchronous database access so callers running inside
    asyncio do not need to perform SQLite work on the event loop.
    """

    messages: List[
        Dict[str, Any]
    ] = []

    combined_system_prompt = (
        _build_system_prompt(
            system_prompt
        )
    )

    if combined_system_prompt:
        messages.append(
            {
                "role": "system",
                "content": combined_system_prompt,
            }
        )

    history = await get_chat_history_async(
        session_id=session_id,
        limit=_normalize_limit(limit),
    )

    _append_history_and_input(
        messages,
        history,
        current_input,
    )

    return messages


def clear_prompt_cache() -> None:
    """
    Clear the cached system/tool prompt.

    Useful if tools are dynamically registered or the system
    prompt changes during runtime.
    """

    _build_system_prompt_cached.cache_clear()