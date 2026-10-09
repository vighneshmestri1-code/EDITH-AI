import ast
import json
import re
from typing import AsyncGenerator, Tuple

from brain.context import build_context
from brain.ollama_client import OllamaClient
from config.prompts import SYSTEM_PROMPT
from core.system_commands import execute_system_command
from database.memory import (
    clear_session_history,
    init_db,
    save_message,
)
from tools.registry import (
    TOOL_REGISTRY,
    execute_tool_async,
)
from utils.logger import logger


FINAL_RESPONSE_INSTRUCTION = """
The requested tool has already been executed.

Do NOT call another tool.
Do NOT output JSON.
Do NOT output a tool call.
Do NOT output Markdown code fences.

Use the tool result to answer the user's original request.

Return ONLY the final natural-language response that should be shown
to the user.

Keep the response concise and accurate.
"""


DIRECT_RESPONSE_TOOLS = {
    "calculate",
    "browser_search",
    "browser_open_url",
    "take_screenshot",
    "schedule_reminder",
    "open_application",
    "close_application",
}


def _remove_code_fences(text: str) -> str:
    """Remove Markdown code fences around model output."""

    clean_text = text.strip()

    if not clean_text.startswith("```"):
        return clean_text

    lines = clean_text.splitlines()

    if len(lines) >= 3 and lines[-1].strip() == "```":
        return "\n".join(lines[1:-1]).strip()

    return clean_text


def _extract_json_object(text: str) -> dict | None:
    """
    Extract a JSON object from model output.

    Supports clean JSON as well as JSON surrounded by
    accidental text.
    """

    if not text:
        return None

    clean_text = _remove_code_fences(text)

    try:
        parsed = json.loads(clean_text)

        if isinstance(parsed, dict):
            return parsed

    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()

    for index, character in enumerate(clean_text):

        if character != "{":
            continue

        try:
            parsed, _ = decoder.raw_decode(
                clean_text[index:]
            )

            if isinstance(parsed, dict):
                return parsed

        except json.JSONDecodeError:
            continue

    return None


def parse_tool_call(
    reply: str,
) -> Tuple[str | None, dict]:
    """
    Parse an EDITH tool call.

    Supported formats:

    {"tool": "web_search", "args": {"query": "Python"}}

    {"web_search": {"query": "Python"}}

    web_search(query="Python")
    """

    if not reply or not reply.strip():
        return None, {}

    clean_reply = reply.strip()

    tool_data = _extract_json_object(
        clean_reply
    )

    if isinstance(tool_data, dict):

        if "tool" in tool_data:

            tool_name = tool_data.get(
                "tool"
            )

            tool_args = tool_data.get(
                "args",
                {},
            )

            if (
                isinstance(tool_name, str)
                and tool_name in TOOL_REGISTRY
                and isinstance(tool_args, dict)
            ):
                return tool_name, tool_args

        for key, value in tool_data.items():

            if key in TOOL_REGISTRY:

                args = (
                    value
                    if isinstance(value, dict)
                    else {}
                )

                return key, args

    function_pattern = (
        r"^([a-zA-Z_][a-zA-Z0-9_]*)\((.*)\)$"
    )

    match = re.match(
        function_pattern,
        clean_reply,
        re.DOTALL,
    )

    if not match:
        return None, {}

    tool_name = match.group(1)

    if tool_name not in TOOL_REGISTRY:
        return None, {}

    args_string = match.group(2).strip()

    if not args_string:
        return tool_name, {}

    kwargs = {}

    try:

        fake_call = f"dict({args_string})"

        parsed = ast.parse(
            fake_call,
            mode="eval",
        )

        if not isinstance(
            parsed.body,
            ast.Call,
        ):
            return None, {}

        for keyword in parsed.body.keywords:

            if keyword.arg is None:
                continue

            kwargs[keyword.arg] = (
                ast.literal_eval(
                    keyword.value
                )
            )

        return tool_name, kwargs

    except Exception as e:

        logger.warning(
            "Could not parse function arguments "
            f"'{args_string}': {e}"
        )

        return None, {}


def _normalize_math_expression(
    user_input: str,
) -> str:
    """
    Normalize common human math notation into an
    expression understood by the safe calculator tool.
    """

    expression = user_input.strip()

    expression = re.sub(
        r"^(calculate|calc|solve|evaluate|work out)\s*:?\s*",
        "",
        expression,
        flags=re.IGNORECASE,
    )

    expression = re.sub(
        r"^(what is|what's)\s+",
        "",
        expression,
        flags=re.IGNORECASE,
    )

    expression = expression.replace("×", "*")
    expression = expression.replace("÷", "/")
    expression = expression.replace("−", "-")
    expression = expression.replace("–", "-")
    expression = expression.replace("—", "-")
    expression = expression.replace("^", "**")

    superscript_map = str.maketrans(
        "⁰¹²³⁴⁵⁶⁷⁸⁹",
        "0123456789",
    )

    def replace_superscript(
        match: re.Match,
    ) -> str:
        number = match.group(1)
        superscript = match.group(2)

        power = superscript.translate(
            superscript_map
        )

        return (
            f"{number}**{power}"
        )

    expression = re.sub(
        r"(\d+(?:\.\d+)?)\s*([⁰¹²³⁴⁵⁶⁷⁸⁹]+)",
        replace_superscript,
        expression,
    )

    def replace_percentage(
        match: re.Match,
    ) -> str:
        percentage = match.group(1)
        value = match.group(2)

        return (
            f"({percentage}/100)*{value}"
        )

    expression = re.sub(
        r"(\d+(?:\.\d+)?)\s*%\s*of\s*"
        r"(\d+(?:\.\d+)?)",
        replace_percentage,
        expression,
        flags=re.IGNORECASE,
    )

    expression = re.sub(
        r"\s+",
        " ",
        expression,
    ).strip()

    return expression


def _looks_like_math(
    user_input: str,
) -> bool:
    """
    Determine whether a user message is likely an
    arithmetic expression.

    This intentionally uses conservative rules so normal
    questions are not accidentally sent to the calculator.
    """

    text = user_input.strip().lower()

    if not text:
        return False

    explicit_math_prefix = re.match(
        r"^(calculate|calc|solve|evaluate|work out)\b",
        text,
    )

    if explicit_math_prefix:
        return True

    if "%" in text and re.search(
        r"\d",
        text,
    ):
        return True

    has_digit = bool(
        re.search(r"\d", text)
    )

    has_operator = bool(
        re.search(
            r"[+\-*/×÷^]",
            text,
        )
    )

    if not (
        has_digit
        and has_operator
    ):
        return False

    allowed_question_prefix = re.match(
        r"^(what is|what's|can you solve|solve)\b",
        text,
    )

    if allowed_question_prefix:
        return True

    compact = re.sub(
        r"\s+",
        "",
        text,
    )

    expression_like = re.fullmatch(
        r"[0-9().,+\-*/×÷%^]+",
        compact,
    )

    return expression_like is not None


def _looks_like_browser_search(
    user_input: str,
) -> bool:
    """Detect explicit browser-search commands."""

    text = user_input.strip().lower()

    return (
        text.startswith("search google for ")
        or text.startswith("search bing for ")
        or text.startswith("search youtube for ")
        or text.startswith("search duckduckgo for ")
    )


def _build_browser_search_call(
    user_input: str,
) -> Tuple[str | None, dict]:
    """Convert an explicit browser search into a tool call."""

    text = user_input.strip()

    patterns = [
        (
            "search google for ",
            "google",
        ),
        (
            "search bing for ",
            "bing",
        ),
        (
            "search youtube for ",
            "youtube",
        ),
        (
            "search duckduckgo for ",
            "duckduckgo",
        ),
    ]

    lowered = text.lower()

    for prefix, engine in patterns:

        if lowered.startswith(prefix):

            query = text[len(prefix):].strip()

            if not query:
                return None, {}

            return (
                "browser_search",
                {
                    "query": query,
                    "engine": engine,
                },
            )

    return None, {}


def _looks_like_known_url(
    user_input: str,
) -> bool:
    """Detect simple explicit 'open website' requests."""

    text = user_input.strip().lower()

    if not text.startswith("open "):
        return False

    target = text[5:].strip()

    if not target:
        return False

    if target.startswith(
        (
            "url ",
            "chrome",
            "google",
            "youtube",
            "calculator",
            "whatsapp",
        )
    ):
        return False

    return bool(
        re.fullmatch(
            r"(?:https?://)?"
            r"(?:www\.)?"
            r"[a-z0-9-]+\.[a-z]{2,}"
            r"(?:/[^\s]*)?",
            target,
        )
    )


def _build_url_tool_call(
    user_input: str,
) -> Tuple[str | None, dict]:
    """Convert 'open example.com' into browser_open_url."""

    if not _looks_like_known_url(user_input):
        return None, {}

    url = user_input.strip()[5:].strip()

    return (
        "browser_open_url",
        {
            "url": url,
        },
    )


def _build_youtube_play_command(
    user_input: str,
) -> str | None:
    """
    Detect requests to play something on YouTube and
    convert composite phrasing into the existing
    deterministic 'play <query>' system command.
    """

    text = re.sub(
        r"\s+",
        " ",
        user_input.strip(),
    )

    if not text:
        return None

    lowered = text.lower()

    patterns = [
        r"^play\s+(.+?)\s+on\s+youtube$",
        r"^play\s+(.+?)\s+on\s+youtube\s*$",
        r"^open\s+youtube\s+and\s+play\s+(.+)$",
        r"^open\s+youtube\s+then\s+play\s+(.+)$",
        r"^open\s+youtube\s+and\s+play\s+the\s+(.+)$",
    ]

    for pattern in patterns:

        match = re.match(
            pattern,
            lowered,
            re.IGNORECASE,
        )

        if match:

            query = match.group(1).strip()

            if query:
                return f"play {query}"

    if lowered.startswith("play "):

        query = text[5:].strip()

        if not query:
            return None

        return f"play {query}"

    return None


def _fast_conversation_response(
    user_input: str,
) -> str | None:
    """
    Handle tiny conversational interactions without an
    Ollama round-trip.
    """

    normalized = re.sub(
        r"[.!?,]+$",
        "",
        user_input.strip().lower(),
    )

    responses = {
        "hello": "Yes Boss How can I help you?",
        "hi": "Hello Boss. How can I help?",
        "hey": "Hey Boss. I'm listening.",
        "hey edith": "Yes Boss.",
        "hello edith": "Yes Boss. I'm here.",
        "thanks": "You're welcome.",
        "thank you": "You're welcome Boss.",
        "goodbye": "Take care Boss.",
        "bye": "Take care Boss.",
    }

    return responses.get(normalized)


def _clean_final_response(
    response: str,
) -> str:
    """
    Remove accidental formatting and reject accidental
    tool calls from a final response.
    """

    if not response:
        return ""

    clean_response = (
        _remove_code_fences(response)
        .strip()
    )

    tool_name, _ = parse_tool_call(
        clean_response
    )

    if tool_name:

        logger.warning(
            "Final response unexpectedly contained "
            f"tool call '{tool_name}'."
        )

        return ""

    return clean_response


def _normalize_confirmation(
    user_input: str,
) -> str | None:
    """Normalize a yes/no confirmation response."""

    text = re.sub(
        r"[.!?,]+$",
        "",
        user_input.strip().lower(),
    )

    yes_values = {
        "yes",
        "yeah",
        "yep",
        "yup",
        "sure",
        "sure thing",
        "send it",
        "go ahead",
        "do it",
        "confirm",
        "okay",
        "ok",
    }

    no_values = {
        "no",
        "nope",
        "cancel",
        "stop",
        "don't",
        "do not",
        "don't send",
        "do not send",
    }

    if text in yes_values:
        return "yes"

    if text in no_values:
        return "no"

    return None


def _extract_whatsapp_one_shot(
    user_input: str,
) -> Tuple[str | None, str | None]:
    """
    Extract a recipient and message from a one-shot WhatsApp request.

    Supported examples:

        send "Hello" to Rahul on WhatsApp
        message "Hello" to Rahul on WhatsApp
        send Hello to Rahul on WhatsApp
    """

    text = re.sub(
        r"\s+",
        " ",
        user_input.strip(),
    )

    pattern = re.compile(
        r"^(?:send|message)\s+"
        r"(?:[\"'](?P<quoted_message>.+?)[\"']|"
        r"(?P<plain_message>.+?))"
        r"\s+to\s+"
        r"(?P<recipient>.+?)"
        r"(?:\s+on\s+whatsapp)?$",
        re.IGNORECASE,
    )

    match = pattern.match(text)

    if not match:
        return None, None

    message = (
        match.group("quoted_message")
        or match.group("plain_message")
        or ""
    ).strip()

    recipient = (
        match.group("recipient")
        or ""
    ).strip()

    if not message or not recipient:
        return None, None

    return recipient, message





def _build_scheduler_reminder_call(
    user_input: str,
) -> Tuple[str | None, dict]:
    """Convert natural-language reminders into scheduler tool calls."""

    text = re.sub(
        r"\s+",
        " ",
        user_input.strip(),
    )

    if not text:
        return None, {}

    units = (
        r"seconds?|secs?|minutes?|mins?|"
        r"hours?|hrs?|days?"
    )

    patterns = [
        re.compile(
            r"^(?:remind me|set (?:a )?reminder)\s+"
            r"(?:in|after|for)\s+"
            r"(?P<amount>\d+(?:\.\d+)?)\s*"
            rf"(?P<unit>{units})\s+"
            r"(?:to|that)\s+"
            r"(?P<message>.+)$",
            re.IGNORECASE,
        ),
        re.compile(
            r"^(?:remind me|set (?:a )?reminder)\s+"
            r"(?:to|that)\s+"
            r"(?P<message>.+?)\s+"
            r"(?:in|after|for)\s+"
            r"(?P<amount>\d+(?:\.\d+)?)\s*"
            rf"(?P<unit>{units})"
            r"\s*[.!?,;:]*$",
            re.IGNORECASE,
        ),
    ]

    match = None

    for pattern in patterns:
        match = pattern.match(text)
        if match:
            break

    if not match:
        return None, {}

    try:
        amount = float(match.group("amount"))
    except (TypeError, ValueError):
        return None, {}

    unit = match.group("unit").lower()
    message = match.group("message").strip()

    if amount <= 0 or not message:
        return None, {}

    if unit.startswith(("second", "sec")):
        multiplier = 1
    elif unit.startswith(("minute", "min")):
        multiplier = 60
    elif unit.startswith(("hour", "hr")):
        multiplier = 3600
    else:
        multiplier = 86400

    delay_seconds = int(round(amount * multiplier))

    if delay_seconds <= 0:
        return None, {}

    return (
        "schedule_reminder",
        {
            "message": message,
            "delay_seconds": delay_seconds,
        },
    )


class EdithOrchestrator:

    def __init__(
        self,
        session_id: str = "default_session",
    ):
        self.brain = OllamaClient()

        self.session_id = (
            session_id.strip()
        )

        if not self.session_id:
            self.session_id = (
                "default_session"
            )

        self.whatsapp_state: str | None = None
        self.whatsapp_recipient: str | None = None
        self.whatsapp_message: str | None = None

        init_db()

        logger.info(
            "EdithOrchestrator initialized "
            "with session ID: "
            f"{self.session_id}"
        )

    def _reset_whatsapp_state(self) -> None:
        """Clear any pending WhatsApp conversation state."""

        self.whatsapp_state = None
        self.whatsapp_recipient = None
        self.whatsapp_message = None

    def _whatsapp_active(self) -> bool:
        """Return whether EDITH is currently handling a WhatsApp flow."""

        return self.whatsapp_state is not None

    async def _handle_whatsapp_turn(
        self,
        user_input: str,
    ) -> Tuple[bool, str]:
        """
        Handle the deterministic multi-turn WhatsApp workflow.

        States:
            recipient
            message
            confirmation
        """

        text = user_input.strip()
        lowered = text.lower()

        if not text:
            return False, ""

        if lowered in {
            "cancel",
            "cancel whatsapp",
            "stop",
            "never mind",
            "nevermind",
        } and self._whatsapp_active():

            self._reset_whatsapp_state()

            await self._save_message(
                "user",
                user_input,
            )

            response = (
                "Understood. I cancelled the "
                "WhatsApp message."
            )

            await self._save_message(
                "assistant",
                response,
            )

            return True, response

        if self.whatsapp_state == "recipient":

            recipient = text.strip()

            if not recipient:

                response = (
                    "Please tell me the contact "
                    "you want to message."
                )

                await self._save_message(
                    "user",
                    user_input,
                )

                await self._save_message(
                    "assistant",
                    response,
                )

                return True, response

            self.whatsapp_recipient = recipient
            self.whatsapp_state = "message"

            await self._save_message(
                "user",
                user_input,
            )

            response = (
                f"Got it. What message should I "
                f"send to {recipient}?"
            )

            await self._save_message(
                "assistant",
                response,
            )

            return True, response

        if self.whatsapp_state == "message":

            message = text

            if not message:
                response = (
                    "Please tell me what message "
                    "you'd like me to send."
                )

                await self._save_message(
                    "user",
                    user_input,
                )

                await self._save_message(
                    "assistant",
                    response,
                )

                return True, response

            self.whatsapp_message = message
            self.whatsapp_state = "confirmation"

            recipient = (
                self.whatsapp_recipient
                or "that contact"
            )

            await self._save_message(
                "user",
                user_input,
            )

            response = (
                f"I have the message ready for "
                f"{recipient}. Shall I send it?"
            )

            await self._save_message(
                "assistant",
                response,
            )

            return True, response

        if self.whatsapp_state == "confirmation":

            confirmation = _normalize_confirmation(
                text
            )

            if confirmation == "no":

                self._reset_whatsapp_state()

                await self._save_message(
                    "user",
                    user_input,
                )

                response = (
                    "Understood. I won't send it."
                )

                await self._save_message(
                    "assistant",
                    response,
                )

                return True, response

            if confirmation == "yes":

                recipient = (
                    self.whatsapp_recipient
                    or ""
                )

                message = (
                    self.whatsapp_message
                    or ""
                )

                logger.info(
                    "WhatsApp send confirmed. "
                    f"Recipient: '{recipient}'"
                )

                if not recipient or not message:

                    self._reset_whatsapp_state()

                    await self._save_message(
                        "user",
                        user_input,
                    )

                    response = (
                        "The WhatsApp message details "
                        "are incomplete."
                    )

                    await self._save_message(
                        "assistant",
                        response,
                    )

                    return True, response

                command = (
                    "whatsapp send "
                    f"to {recipient}: {message}"
                )

                try:

                    system_response = (
                        await execute_system_command(
                            command
                        )
                    )

                except Exception as e:

                    logger.error(
                        "WhatsApp send execution failed: "
                        f"{e}",
                        exc_info=True,
                    )

                    system_response = None

                self._reset_whatsapp_state()

                await self._save_message(
                    "user",
                    user_input,
                )

                if system_response:

                    response = system_response

                else:

                    response = (
                        "I couldn't send the WhatsApp "
                        "message yet. The WhatsApp sending "
                        "handler is not available."
                    )

                await self._save_message(
                    "assistant",
                    response,
                )

                return True, response

            await self._save_message(
                "user",
                user_input,
            )

            response = (
                "Please confirm with yes or no."
            )

            await self._save_message(
                "assistant",
                response,
            )

            return True, response

        return False, ""

    async def _start_whatsapp_flow(
        self,
        user_input: str,
    ) -> Tuple[bool, str]:
        """
        Detect and initialize WhatsApp messaging requests.

        This handles:

            open whatsapp
            launch whatsapp
        """

        text = user_input.strip().lower()

        is_open_request = text in {
            "open whatsapp",
            "launch whatsapp",
        }

        if is_open_request:

            try:

                system_response = (
                    await execute_system_command(
                        user_input
                    )
                )

            except Exception as e:

                logger.error(
                    "WhatsApp open command failed: "
                    f"{e}",
                    exc_info=True,
                )

                system_response = None

            if not system_response:

                return True, (
                    "I couldn't open WhatsApp."
                )

            if (
                "couldn't" in system_response.lower()
                or "could not" in system_response.lower()
                or "failed" in system_response.lower()
            ):

                await self._save_message(
                    "user",
                    user_input,
                )

                await self._save_message(
                    "assistant",
                    system_response,
                )

                return True, system_response

            self.whatsapp_state = "recipient"
            self.whatsapp_recipient = None
            self.whatsapp_message = None

            await self._save_message(
                "user",
                user_input,
            )

            response = (
                "WhatsApp is open. "
                "Who should I message?"
            )

            await self._save_message(
                "assistant",
                response,
            )

            return True, response

        recipient, message = (
            _extract_whatsapp_one_shot(
                user_input
            )
        )

        if recipient and message:

            self.whatsapp_recipient = recipient
            self.whatsapp_message = message
            self.whatsapp_state = "confirmation"

            await self._save_message(
                "user",
                user_input,
            )

            response = (
                f"I have the message ready for "
                f"{recipient}. Shall I send it?"
            )

            await self._save_message(
                "assistant",
                response,
            )

            return True, response

        return False, ""

    async def _execute_tool(
        self,
        tool_name: str,
        tool_args: dict,
    ) -> str:
        """Safely execute a registered tool asynchronously."""

        if tool_name not in TOOL_REGISTRY:

            logger.warning(
                f"Unknown tool requested: "
                f"{tool_name}"
            )

            return (
                f"Error: Unknown tool "
                f"'{tool_name}'."
            )

        if not isinstance(
            tool_args,
            dict,
        ):

            return (
                f"Error: Invalid arguments "
                f"for tool '{tool_name}'."
            )

        logger.info(
            f"Executing tool '{tool_name}' "
            f"with arguments: {tool_args}"
        )

        try:

            result = await execute_tool_async(
                tool_name,
                **tool_args,
            )

            return str(result)

        except Exception as e:

            logger.error(
                f"Tool '{tool_name}' execution failed: "
                f"{e}",
                exc_info=True,
            )

            return (
                f"Error executing tool "
                f"'{tool_name}': {str(e)}"
            )

    async def _save_message(
        self,
        role: str,
        content: str,
    ) -> None:
        """Persist a conversation message safely."""

        try:

            success = save_message(
                self.session_id,
                role,
                content,
            )

            if not success:

                logger.warning(
                    "Message could not be saved: "
                    f"role={role}"
                )

        except Exception as e:

            logger.error(
                f"Failed to save {role} message: "
                f"{e}",
                exc_info=True,
            )

    async def _direct_tool_response(
        self,
        user_input: str,
        tool_name: str,
        tool_args: dict,
    ) -> str:
        """
        Execute a deterministic tool and return its result
        without an unnecessary second Ollama generation.
        """

        tool_output = await self._execute_tool(
            tool_name,
            tool_args,
        )

        await self._save_message(
            "user",
            user_input,
        )

        await self._save_message(
            "assistant",
            f"[Tool Call: {tool_name}]",
        )

        await self._save_message(
            "system",
            f"Tool Output: {tool_output}",
        )

        return tool_output.strip()

    async def _build_final_response(
        self,
        messages: list,
        tool_output: str,
    ) -> str:
        """
        Generate a natural final response when a tool result
        actually requires LLM interpretation.
        """

        final_messages = list(messages)

        final_messages.append(
            {
                "role": "user",
                "content": (
                    "Tool Execution Result:\n"
                    f"{tool_output}\n\n"
                    f"{FINAL_RESPONSE_INSTRUCTION}"
                ),
            }
        )

        final_reply = (
            await self.brain.generate_response(
                final_messages
            )
        )

        final_reply = _clean_final_response(
            final_reply or ""
        )

        if final_reply:
            return final_reply

        return tool_output.strip()

    async def _run_fast_path(
        self,
        user_input: str,
    ) -> Tuple[bool, str]:
        """
        Resolve requests that should not require an Ollama
        reasoning pass.
        """

        whatsapp_handled, whatsapp_response = (
            await self._handle_whatsapp_turn(
                user_input
            )
        )

        if whatsapp_handled:
            return True, whatsapp_response

        whatsapp_started, whatsapp_response = (
            await self._start_whatsapp_flow(
                user_input
            )
        )

        if whatsapp_started:
            return True, whatsapp_response

        direct_response = (
            _fast_conversation_response(
                user_input
            )
        )

        if direct_response:

            await self._save_message(
                "user",
                user_input,
            )

            await self._save_message(
                "assistant",
                direct_response,
            )

            return True, direct_response

        reminder_tool_name, reminder_tool_args = (
            _build_scheduler_reminder_call(
                user_input
            )
        )

        if reminder_tool_name:

            logger.info(
                "Deterministic reminder command detected. "
                f"Arguments: {reminder_tool_args}"
            )

            try:

                result = (
                    await self._direct_tool_response(
                        user_input,
                        reminder_tool_name,
                        reminder_tool_args,
                    )
                )

                return True, result

            except Exception as e:

                logger.error(
                    "Direct reminder scheduler path failed: "
                    f"{e}",
                    exc_info=True,
                )

                return True, (
                    "I couldn't schedule the reminder."
                )

        youtube_play_command = (
            _build_youtube_play_command(
                user_input
            )
        )

        if youtube_play_command:

            logger.info(
                "YouTube play command detected. "
                f"Normalized command: "
                f"{youtube_play_command}"
            )

            try:

                system_response = (
                    await execute_system_command(
                        youtube_play_command
                    )
                )

                if system_response:

                    await self._save_message(
                        "user",
                        user_input,
                    )

                    await self._save_message(
                        "assistant",
                        system_response,
                    )

                    return True, system_response

            except Exception as e:

                logger.error(
                    "YouTube play command failed: "
                    f"{e}",
                    exc_info=True,
                )

            return True, (
                "I couldn't start that YouTube playback."
            )

        system_response = (
            await execute_system_command(
                user_input
            )
        )

        if system_response:

            await self._save_message(
                "user",
                user_input,
            )

            await self._save_message(
                "assistant",
                system_response,
            )

            return True, system_response

        if _looks_like_math(user_input):

            expression = (
                _normalize_math_expression(
                    user_input
                )
            )

            if expression:

                try:

                    result = (
                        await self._direct_tool_response(
                            user_input,
                            "calculate",
                            {
                                "expression": expression,
                            },
                        )
                    )

                    return True, result

                except Exception as e:

                    logger.error(
                        "Direct calculator path failed: "
                        f"{e}",
                        exc_info=True,
                    )

                    return True, (
                        "I couldn't calculate that expression."
                    )

        if _looks_like_browser_search(
            user_input
        ):

            tool_name, tool_args = (
                _build_browser_search_call(
                    user_input
                )
            )

            if tool_name:

                result = (
                    await self._direct_tool_response(
                        user_input,
                        tool_name,
                        tool_args,
                    )
                )

                return True, result

        if _looks_like_known_url(
            user_input
        ):

            tool_name, tool_args = (
                _build_url_tool_call(
                    user_input
                )
            )

            if tool_name:

                result = (
                    await self._direct_tool_response(
                        user_input,
                        tool_name,
                        tool_args,
                    )
                )

                return True, result

        return False, ""

    async def process_turn(
        self,
        user_input: str,
    ) -> str:
        """Process one complete EDITH conversation turn."""

        if (
            not user_input
            or not user_input.strip()
        ):
            return "I didn't receive any input."

        user_input = user_input.strip()

        handled, fast_response = (
            await self._run_fast_path(
                user_input
            )
        )

        if handled:
            return fast_response

        messages = build_context(
            session_id=self.session_id,
            current_input=user_input,
            system_prompt=SYSTEM_PROMPT,
            limit=10,
        )

        reply = (
            await self.brain.generate_response(
                messages
            )
        )

        if not reply:

            return (
                "I encountered an error "
                "processing your request."
            )

        tool_name, tool_args = (
            parse_tool_call(reply)
        )

        if tool_name:

            logger.info(
                "Tool invocation detected: "
                f"{tool_name} with args "
                f"{tool_args}"
            )

            if tool_name in DIRECT_RESPONSE_TOOLS:

                result = (
                    await self._direct_tool_response(
                        user_input,
                        tool_name,
                        tool_args,
                    )
                )

                await self._save_message(
                    "assistant",
                    result,
                )

                return result

            tool_output = (
                await self._execute_tool(
                    tool_name,
                    tool_args,
                )
            )

            await self._save_message(
                "user",
                user_input,
            )

            await self._save_message(
                "assistant",
                f"[Tool Call: {tool_name}]",
            )

            await self._save_message(
                "system",
                f"Tool Output: {tool_output}",
            )

            final_reply = (
                await self._build_final_response(
                    messages,
                    tool_output,
                )
            )

            await self._save_message(
                "assistant",
                final_reply,
            )

            return final_reply

        await self._save_message(
            "user",
            user_input,
        )

        await self._save_message(
            "assistant",
            reply.strip(),
        )

        return reply.strip()

    async def process_turn_stream(
        self,
        user_input: str,
    ) -> AsyncGenerator[str, None]:
        """
        Process a turn with streaming for normal conversation
        while keeping tool calls hidden.
        """

        if (
            not user_input
            or not user_input.strip()
        ):

            yield "I didn't receive any input."
            return

        user_input = user_input.strip()

        handled, fast_response = (
            await self._run_fast_path(
                user_input
            )
        )

        if handled:

            yield fast_response
            return

        messages = build_context(
            session_id=self.session_id,
            current_input=user_input,
            system_prompt=SYSTEM_PROMPT,
            limit=10,
        )

        accumulated_reply = ""

        stream_started = False
        possible_tool_call = False
        buffered_chunks: list[str] = []

        async for chunk in (
            self.brain.generate_stream(
                messages
            )
        ):

            if not chunk:
                continue

            accumulated_reply += chunk

            if stream_started:
                yield chunk
                continue

            buffered_chunks.append(chunk)

            probe = (
                "".join(buffered_chunks)
                .lstrip()
            )

            if not probe:
                continue

            first_char = probe[0]

            looks_json = (
                first_char == "{"
                or bool(
                    re.match(
                        r"^```(?:json)?\s*\{",
                        probe,
                        re.IGNORECASE | re.DOTALL,
                    )
                )
            )

            looks_function = bool(
                re.match(
                    r"^[a-zA-Z_][a-zA-Z0-9_]*\s*\(",
                    probe,
                )
            )

            if (
                looks_json
                or looks_function
            ):

                possible_tool_call = True
                continue

            if len(probe) >= 24:

                stream_started = True

                for buffered in buffered_chunks:
                    yield buffered

                buffered_chunks.clear()

        if not stream_started:

            if not possible_tool_call:

                for buffered in buffered_chunks:
                    yield buffered

        if not possible_tool_call:

            clean_reply = (
                _clean_final_response(
                    accumulated_reply
                )
            )

            if clean_reply != accumulated_reply.strip():

                accumulated_reply = clean_reply

            await self._save_message(
                "user",
                user_input,
            )

            await self._save_message(
                "assistant",
                accumulated_reply,
            )

            return

        tool_name, tool_args = (
            parse_tool_call(
                accumulated_reply
            )
        )

        if not tool_name:

            clean_reply = (
                _clean_final_response(
                    accumulated_reply
                )
            )

            if clean_reply:

                await self._save_message(
                    "user",
                    user_input,
                )

                await self._save_message(
                    "assistant",
                    clean_reply,
                )

                if not stream_started:
                    yield clean_reply

                return

            await self._save_message(
                "user",
                user_input,
            )

            await self._save_message(
                "assistant",
                accumulated_reply,
            )

            return

        logger.info(
            "Streaming tool invocation detected: "
            f"{tool_name} with args {tool_args}"
        )

        if tool_name in DIRECT_RESPONSE_TOOLS:

            result = (
                await self._direct_tool_response(
                    user_input,
                    tool_name,
                    tool_args,
                )
            )

            await self._save_message(
                "assistant",
                result,
            )

            yield result
            return

        tool_output = (
            await self._execute_tool(
                tool_name,
                tool_args,
            )
        )

        await self._save_message(
            "user",
            user_input,
        )

        await self._save_message(
            "assistant",
            f"[Tool Call: {tool_name}]",
        )

        await self._save_message(
            "system",
            f"Tool Output: {tool_output}",
        )

        final_messages = list(messages)

        final_messages.append(
            {
                "role": "user",
                "content": (
                    "Tool Execution Result:\n"
                    f"{tool_output}\n\n"
                    f"{FINAL_RESPONSE_INSTRUCTION}"
                ),
            }
        )

        final_accumulated = ""

        async for chunk in (
            self.brain.generate_stream(
                final_messages
            )
        ):

            final_accumulated += chunk

        final_reply = (
            _clean_final_response(
                final_accumulated
            )
        )

        if not final_reply:
            final_reply = tool_output.strip()

        await self._save_message(
            "assistant",
            final_reply,
        )

        yield final_reply

    def reset_conversation(
        self,
    ) -> bool:
        """Clear the current conversation history."""

        self._reset_whatsapp_state()

        try:

            success = (
                clear_session_history(
                    self.session_id
                )
            )

            if success:

                logger.info(
                    "Conversation history cleared "
                    "for session: "
                    f"{self.session_id}"
                )

                return True

            logger.warning(
                "Conversation history could not "
                "be cleared for session: "
                f"{self.session_id}"
            )

            return False

        except Exception as e:

            logger.error(
                "Failed to reset conversation "
                f"for session "
                f"{self.session_id}: {e}",
                exc_info=True,
            )

            return False