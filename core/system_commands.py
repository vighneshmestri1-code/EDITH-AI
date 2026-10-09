import asyncio
import ctypes
import os
import re
import subprocess
import time
import webbrowser
from urllib.parse import quote_plus

from utils.logger import logger


try:
    from AppOpener import open as launch_app, close as close_app

    HAS_APP_OPENER = True
except ImportError:
    HAS_APP_OPENER = False
    logger.warning(
        "AppOpener is not installed. "
        "Generic application launching/closing may be unavailable."
    )


try:
    import yt_dlp

    HAS_YT_DLP = True
except ImportError:
    HAS_YT_DLP = False
    logger.warning(
        "yt-dlp is not installed. "
        "YouTube playback will use search fallback."
    )


try:
    import pyautogui

    HAS_PYAUTOGUI = True
except ImportError:
    HAS_PYAUTOGUI = False
    logger.warning(
        "PyAutoGUI is not installed. "
        "WhatsApp message automation is unavailable."
    )


try:
    import pyperclip

    HAS_PYPERCLIP = True
except ImportError:
    HAS_PYPERCLIP = False
    logger.warning(
        "Pyperclip is not installed. "
        "WhatsApp text pasting is unavailable."
    )


def _find_chrome() -> str | None:
    """Find Google Chrome executable on Windows."""

    possible_paths = [
        os.path.join(
            os.environ.get("PROGRAMFILES", ""),
            "Google",
            "Chrome",
            "Application",
            "chrome.exe",
        ),
        os.path.join(
            os.environ.get("PROGRAMFILES(X86)", ""),
            "Google",
            "Chrome",
            "Application",
            "chrome.exe",
        ),
        os.path.join(
            os.environ.get("LOCALAPPDATA", ""),
            "Google",
            "Chrome",
            "Application",
            "chrome.exe",
        ),
    ]

    for path in possible_paths:
        if path and os.path.isfile(path):
            return path

    return None


def _find_whatsapp() -> str | None:
    """Find WhatsApp Desktop executable on Windows."""

    possible_paths = [
        os.path.join(
            os.environ.get("LOCALAPPDATA", ""),
            "WhatsApp",
            "WhatsApp.exe",
        ),
        os.path.join(
            os.environ.get("LOCALAPPDATA", ""),
            "Programs",
            "WhatsApp",
            "WhatsApp.exe",
        ),
        os.path.join(
            os.environ.get("PROGRAMFILES", ""),
            "WhatsApp",
            "WhatsApp.exe",
        ),
        os.path.join(
            os.environ.get("PROGRAMFILES(X86)", ""),
            "WhatsApp",
            "WhatsApp.exe",
        ),
    ]

    for path in possible_paths:
        if path and os.path.isfile(path):
            return path

    return None


def _launch_chrome(
    url: str | None = None,
) -> bool:
    """Launch Google Chrome, optionally opening a URL."""

    chrome_path = _find_chrome()

    if not chrome_path:
        return False

    try:
        command = [chrome_path]

        if url:
            command.append(url)

        subprocess.Popen(command)

        return True

    except Exception as e:
        logger.error(
            f"Failed to launch Google Chrome: {e}"
        )

        return False


def _open_default_browser(
    url: str,
) -> bool:
    """Open a URL using the user's default browser."""

    try:
        return bool(
            webbrowser.open(url)
        )

    except Exception as e:
        logger.error(
            f"Failed to open browser URL '{url}': {e}"
        )

        return False


def _launch_calculator() -> bool:
    """Launch Windows Calculator."""

    try:
        subprocess.Popen(["calc.exe"])

        return True

    except Exception as e:
        logger.error(
            f"Failed to open Calculator: {e}"
        )

        return False


def _close_application(
    target: str,
) -> bool:
    """
    Close a Windows application.

    Known applications use deterministic process names.
    Unknown applications fall back to AppOpener and then
    a direct executable-name attempt.
    """

    target = target.strip()

    if not target:
        return False

    normalized_target = re.sub(
        r"\s+",
        " ",
        target.lower(),
    ).strip()

    process_map = {
        "chrome": ["chrome.exe"],
        "google chrome": ["chrome.exe"],
        "whatsapp": ["WhatsApp.exe"],
        "calculator": [
            "CalculatorApp.exe",
            "ApplicationFrameHost.exe",
        ],
        "calc": [
            "CalculatorApp.exe",
            "ApplicationFrameHost.exe",
        ],
        "notepad": ["notepad.exe"],
        "visual studio code": ["Code.exe"],
        "vs code": ["Code.exe"],
        "code": ["Code.exe"],
        "microsoft edge": ["msedge.exe"],
        "edge": ["msedge.exe"],
        "firefox": ["firefox.exe"],
        "spotify": ["Spotify.exe"],
        "vlc": ["vlc.exe"],
        "discord": ["Discord.exe"],
        "telegram": ["Telegram.exe"],
        "steam": ["steam.exe"],
        "zoom": ["Zoom.exe"],
        "teams": [
            "ms-teams.exe",
            "Teams.exe",
        ],
    }

    processes = process_map.get(
        normalized_target
    )

    if processes:

        for process_name in processes:

            try:
                result = subprocess.run(
                    [
                        "taskkill",
                        "/F",
                        "/T",
                        "/IM",
                        process_name,
                    ],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )

                if result.returncode == 0:

                    logger.info(
                        f"Closed application '{target}' "
                        f"using process '{process_name}'."
                    )

                    return True

            except Exception as e:

                logger.error(
                    f"Failed to close process "
                    f"'{process_name}': {e}"
                )

    if HAS_APP_OPENER:

        try:

            close_app(
                target,
                match_closest=True,
            )

            time.sleep(0.5)

            logger.info(
                f"AppOpener attempted to close "
                f"'{target}'."
            )

            return True

        except Exception as e:

            logger.warning(
                f"AppOpener failed to close "
                f"'{target}': {e}"
            )

    executable_name = re.sub(
        r"[^a-zA-Z0-9_-]",
        "",
        normalized_target,
    )

    if executable_name:

        process_name = (
            f"{executable_name}.exe"
        )

        try:

            result = subprocess.run(
                [
                    "taskkill",
                    "/F",
                    "/T",
                    "/IM",
                    process_name,
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )

            if result.returncode == 0:

                logger.info(
                    f"Closed executable "
                    f"'{process_name}'."
                )

                return True

        except Exception as e:

            logger.error(
                f"Generic process close failed "
                f"for '{process_name}': {e}"
            )

    return False


def _extract_play_query(
    command_text: str,
) -> str | None:
    """
    Extract the song/search query from common play commands.
    """

    text = command_text.strip()

    patterns = [
        r"^play\s+(.+?)\s+on\s+youtube$",
        r"^play\s+(.+?)\s+song$",
        r"^play\s+(.+)$",
    ]

    for pattern in patterns:

        match = re.match(
            pattern,
            text,
            re.IGNORECASE,
        )

        if match:

            query = match.group(1).strip()

            if query:
                return query

    return None


def _resolve_youtube_video(
    query: str,
) -> str | None:
    """
    Resolve the first YouTube search result
    into a real video URL.
    """

    if not HAS_YT_DLP:
        return None

    try:

        search_query = (
            f"ytsearch1:{query}"
        )

        options = {
            "quiet": True,
            "no_warnings": True,
            "extract_flat": True,
            "skip_download": True,
            "noplaylist": True,
        }

        with yt_dlp.YoutubeDL(
            options
        ) as ydl:

            info = ydl.extract_info(
                search_query,
                download=False,
            )

        if not info:
            return None

        entries = info.get("entries")

        if not entries:
            return None

        first_entry = entries[0]

        if not first_entry:
            return None

        video_url = first_entry.get(
            "webpage_url"
        )

        if video_url:
            return video_url

        video_id = first_entry.get(
            "id"
        )

        if video_id:
            return (
                "https://www.youtube.com/watch?v="
                f"{video_id}"
            )

    except Exception as e:

        logger.error(
            f"YouTube video resolution failed "
            f"for '{query}': {e}",
            exc_info=True,
        )

    return None


def _play_youtube(
    query: str,
) -> str:
    """Open the requested song/video on YouTube."""

    query = query.strip()

    if not query:

        return (
            "Please tell me what you'd like me to play."
        )

    logger.info(
        f"Resolving YouTube playback for: "
        f"'{query}'"
    )

    video_url = _resolve_youtube_video(
        query
    )

    if video_url:

        success = _open_default_browser(
            video_url
        )

        if success:

            return (
                f"Playing {query} on YouTube."
            )

        return (
            f"I found {query} on YouTube, "
            "but couldn't open the video."
        )

    search_url = (
        "https://www.youtube.com/"
        "results?search_query="
        + quote_plus(query)
    )

    success = _open_default_browser(
        search_url
    )

    if success:

        if HAS_YT_DLP:

            return (
                f"I couldn't start {query} directly, "
                "so I opened its YouTube results."
            )

        return (
            f"I opened YouTube results for {query}. "
            "Install yt-dlp for direct video playback."
        )

    return (
        f"I couldn't open YouTube for {query}."
    )


def _open_whatsapp() -> bool:
    """
    Open WhatsApp Desktop when available.

    Falls back to WhatsApp Web.
    """

    whatsapp_path = _find_whatsapp()

    if whatsapp_path:

        try:

            subprocess.Popen(
                [whatsapp_path]
            )

            logger.info(
                "WhatsApp Desktop launched."
            )

            return True

        except Exception as e:

            logger.error(
                "Failed to launch WhatsApp Desktop: "
                f"{e}"
            )

    if HAS_APP_OPENER:

        try:

            launch_app(
                "WhatsApp",
                match_closest=True,
            )

            logger.info(
                "WhatsApp launched through AppOpener."
            )

            return True

        except Exception as e:

            logger.warning(
                "AppOpener could not launch WhatsApp: "
                f"{e}"
            )

    success = _open_default_browser(
        "https://web.whatsapp.com"
    )

    if success:

        logger.info(
            "WhatsApp Web opened."
        )

    return success


def _parse_whatsapp_send_command(
    command_text: str,
) -> tuple[str | None, str | None]:
    """
    Parse:

        whatsapp send to Rahul: Hello

    Returns:

        recipient, message
    """

    text = command_text.strip()

    match = re.match(
        r"^whatsapp\s+send\s+to\s+"
        r"(?P<recipient>[^:]+)"
        r":\s*(?P<message>.+)$",
        text,
        re.IGNORECASE | re.DOTALL,
    )

    if not match:
        return None, None

    recipient = (
        match.group("recipient")
        .strip()
    )

    message = (
        match.group("message")
        .strip()
    )

    if not recipient or not message:
        return None, None

    return recipient, message


def _copy_text_to_clipboard(
    text: str,
) -> bool:
    """Copy text to the Windows clipboard."""

    if not HAS_PYPERCLIP:
        return False

    try:

        pyperclip.copy(text)

        return True

    except Exception as e:

        logger.error(
            f"Clipboard operation failed: {e}"
        )

        return False


def _focus_whatsapp_message_box() -> bool:
    """Prepare keyboard focus for the active WhatsApp chat safely."""

    if not HAS_PYAUTOGUI:
        return False

    try:
        # After a contact is opened from New Chat, WhatsApp normally keeps
        # keyboard focus in the conversation. Avoid typing a probe character
        # or pressing Backspace because that can alter the chat contents.
        time.sleep(0.2)
        return True
    except Exception as e:
        logger.error(
            "Failed to prepare WhatsApp message composer: "
            f"{e}"
        )
        return False


def _focus_whatsapp_window() -> bool:
    """Bring a visible WhatsApp window to the foreground on Windows."""

    if os.name != "nt":
        return True

    try:
        user32 = ctypes.windll.user32
        windows: list[int] = []

        enum_proc = ctypes.WINFUNCTYPE(
            ctypes.c_bool,
            ctypes.c_void_p,
            ctypes.c_void_p,
        )

        @enum_proc
        def callback(hwnd, _lparam):
            if not user32.IsWindowVisible(hwnd):
                return True

            length = user32.GetWindowTextLengthW(hwnd)

            if length <= 0:
                return True

            buffer = ctypes.create_unicode_buffer(length + 1)
            user32.GetWindowTextW(
                hwnd,
                buffer,
                length + 1,
            )

            title = buffer.value.strip().lower()

            if "whatsapp" in title:
                windows.append(int(hwnd))
                return False

            return True

        user32.EnumWindows(
            callback,
            0,
        )

        if not windows:
            logger.warning(
                "No visible WhatsApp window was found to focus."
            )
            return False

        hwnd = windows[0]

        if user32.IsIconic(hwnd):
            user32.ShowWindow(hwnd, 9)

        user32.SetForegroundWindow(hwnd)

        time.sleep(0.5)

        foreground = int(
            user32.GetForegroundWindow()
        )

        if foreground == hwnd:
            logger.info(
                "WhatsApp window focused successfully."
            )
            return True

        logger.warning(
            "WhatsApp window was found but could not be focused."
        )
        return False

    except Exception as e:
        logger.error(
            "Failed to focus WhatsApp window: "
            f"{e}",
            exc_info=True,
        )
        return False


def _send_whatsapp_message(
    recipient: str,
    message: str,
) -> str:
    """
    Send a WhatsApp message through keyboard automation.

    Flow:

        Open WhatsApp
        New Chat
        Search contact
        Open contact
        Focus composer
        Paste message
        Press Enter
    """

    if not HAS_PYAUTOGUI:

        return (
            "WhatsApp sending requires PyAutoGUI. "
            "Install it with 'pip install pyautogui'."
        )

    if not HAS_PYPERCLIP:

        return (
            "WhatsApp sending requires Pyperclip. "
            "Install it with 'pip install pyperclip'."
        )

    recipient = recipient.strip()
    message = message.strip()

    if not recipient:

        return (
            "I need the WhatsApp contact name."
        )

    if not message:

        return (
            "I need the message to send."
        )

    logger.info(
        "Starting WhatsApp automation. "
        f"Recipient: '{recipient}'"
    )

    try:

        pyautogui.FAILSAFE = True
        pyautogui.PAUSE = 0.2

        if not _open_whatsapp():

            return (
                "I couldn't open WhatsApp."
            )

        logger.info(
            "Waiting for WhatsApp to become ready."
        )

        time.sleep(5)

        if not _focus_whatsapp_window():

            return (
                "I opened WhatsApp, but couldn't bring its window "
                "to the foreground for automation."
            )

        logger.info(
            "Opening WhatsApp New Chat."
        )

        # Ctrl+Alt+N is the documented New Chat shortcut on WhatsApp
        # Web and WhatsApp for Windows.
        pyautogui.hotkey(
            "ctrl",
            "alt",
            "n",
        )

        time.sleep(2)

        if not _copy_text_to_clipboard(
            recipient
        ):

            return (
                "I couldn't copy the WhatsApp "
                "contact name."
            )

        logger.info(
            f"Searching WhatsApp contact: "
            f"'{recipient}'"
        )

        pyautogui.hotkey(
            "ctrl",
            "v",
        )

        time.sleep(2)

        pyautogui.press(
            "enter"
        )

        logger.info(
            f"Opening chat for '{recipient}'."
        )

        time.sleep(3)

        if not _focus_whatsapp_window():

            return (
                f"I found {recipient}, but "
                "WhatsApp lost foreground focus."
            )

        if not _copy_text_to_clipboard(
            message
        ):

            return (
                "I found the contact, but couldn't "
                "prepare the message."
            )

        logger.info(
            "Pasting WhatsApp message."
        )

        pyautogui.hotkey(
            "ctrl",
            "v",
        )

        time.sleep(1)

        logger.info(
            f"Sending WhatsApp message to "
            f"'{recipient}'."
        )

        pyautogui.press(
            "enter"
        )

        time.sleep(1.5)

        logger.info(
            f"WhatsApp message send operation "
            f"completed for '{recipient}'."
        )

        return (
            f"Message sent to {recipient}."
        )

    except pyautogui.FailSafeException:

        logger.warning(
            "PyAutoGUI failsafe triggered during "
            "WhatsApp automation."
        )

        return (
            "WhatsApp automation was stopped "
            "for safety."
        )

    except Exception as e:

        logger.error(
            "WhatsApp automation failed: "
            f"{e}",
            exc_info=True,
        )

        return (
            "I couldn't complete the WhatsApp "
            "message."
        )


async def execute_system_command(
    command_text: str,
) -> str | None:
    """
    Handle deterministic Windows/system commands.

    Returns:
        A response string when the command is handled.
        None when the command should continue to the
        LLM/tool system.
    """

    if not command_text or not command_text.strip():
        return None

    cmd = command_text.strip().lower()

    if cmd in {
        "open google",
        "launch google",
    }:

        success = await asyncio.to_thread(
            _open_default_browser,
            "https://www.google.com",
        )

        if success:

            return (
                "Opening Google in your default browser."
            )

        return "I couldn't open Google."

    if cmd in {
        "open youtube",
        "launch youtube",
    }:

        success = await asyncio.to_thread(
            _open_default_browser,
            "https://www.youtube.com",
        )

        if success:

            return (
                "Opening YouTube in your default browser."
            )

        return "I couldn't open YouTube."

    if cmd in {
        "open chrome",
        "launch chrome",
        "open google chrome",
        "launch google chrome",
    }:

        success = await asyncio.to_thread(
            _launch_chrome
        )

        if success:
            return "Opening Google Chrome."

        if HAS_APP_OPENER:

            try:

                await asyncio.to_thread(
                    launch_app,
                    "Google Chrome",
                    match_closest=True,
                )

                return "Opening Google Chrome."

            except Exception as e:

                logger.error(
                    "AppOpener failed to launch Chrome: "
                    f"{e}"
                )

        return (
            "I couldn't find Google Chrome. "
            "Please make sure Chrome is installed."
        )

    if cmd in {
        "open calculator",
        "launch calculator",
        "open calc",
        "launch calc",
    }:

        success = await asyncio.to_thread(
            _launch_calculator
        )

        if success:
            return "Opening Calculator."

        return "I couldn't open Calculator."

    if cmd in {
        "open whatsapp",
        "launch whatsapp",
    }:

        success = await asyncio.to_thread(
            _open_whatsapp
        )

        if success:
            return "Opening WhatsApp."

        return "I couldn't open WhatsApp."

    if cmd.startswith(
        "whatsapp send to "
    ):

        recipient, message = (
            _parse_whatsapp_send_command(
                command_text
            )
        )

        if not recipient or not message:

            return (
                "I couldn't understand the "
                "WhatsApp recipient or message."
            )

        return await asyncio.to_thread(
            _send_whatsapp_message,
            recipient,
            message,
        )

    if cmd.startswith("play "):

        query = await asyncio.to_thread(
            _extract_play_query,
            command_text,
        )

        if query:

            return await asyncio.to_thread(
                _play_youtube,
                query,
            )

        return (
            "Please tell me what you'd like me to play."
        )

    if cmd.startswith("open url "):

        url = command_text.strip()[9:].strip()

        if not url:
            return None

        if not url.startswith(
            ("http://", "https://")
        ):

            url = f"https://{url}"

        success = await asyncio.to_thread(
            _open_default_browser,
            url,
        )

        if success:

            return (
                f"Opening {url}."
            )

        return (
            f"I couldn't open {url}."
        )

    if (
        cmd.startswith("open ")
        or cmd.startswith("launch ")
    ):

        if cmd.startswith("open "):

            target = command_text.strip()[5:].strip()

        else:

            target = command_text.strip()[7:].strip()

        if not target:
            return None

        if HAS_APP_OPENER:

            try:

                await asyncio.to_thread(
                    launch_app,
                    target,
                    match_closest=True,
                )

                return (
                    f"Launching {target.title()}."
                )

            except Exception as e:

                logger.error(
                    "AppOpener failed to launch "
                    f"'{target}': {e}"
                )

                return (
                    f"I couldn't launch "
                    f"{target.title()}."
                )

        return (
            f"I couldn't launch {target.title()} "
            "because AppOpener is not available."
        )

    if (
        cmd.startswith("close ")
        or cmd.startswith("terminate ")
    ):

        if cmd.startswith("close "):

            target = command_text.strip()[6:].strip()

        else:

            target = command_text.strip()[10:].strip()

        if not target:

            return (
                "Please tell me which application "
                "you want me to close."
            )

        success = await asyncio.to_thread(
            _close_application,
            target,
        )

        if success:

            return (
                f"Closing {target.title()}."
            )

        return (
            f"I couldn't close {target.title()}."
        )

    return None