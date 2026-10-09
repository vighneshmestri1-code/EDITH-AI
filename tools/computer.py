import os
from datetime import datetime

import pyautogui
from mss import MSS, tools

from config.settings import settings
from tools.registry import register_tool
from utils.logger import logger


pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.5


SCREENSHOT_DIR = os.path.join(
    str(settings.BASE_DIR),
    "screenshots",
)

os.makedirs(
    SCREENSHOT_DIR,
    exist_ok=True,
)

MAX_TYPE_TEXT_LENGTH = 10000


def _get_screen_dimensions() -> tuple[int, int]:
    """Return the primary display dimensions."""
    width, height = pyautogui.size()

    return int(width), int(height)


@register_tool(name="take_screenshot")
def take_screenshot() -> str:
    """
    Capture a screenshot of the primary display
    and save it locally.
    """

    try:
        timestamp = datetime.now().strftime(
            "%Y%m%d_%H%M%S_%f"
        )[:-3]

        filepath = os.path.join(
            SCREENSHOT_DIR,
            f"screenshot_{timestamp}.png",
        )

        with MSS() as screenshot:
            primary_monitor = (
                screenshot.primary_monitor
            )

            captured_image = screenshot.grab(
                primary_monitor
            )

            tools.to_png(
                captured_image.rgb,
                captured_image.size,
                output=filepath,
            )

        logger.info(
            f"Screenshot captured: {filepath}"
        )

        return (
            "Screenshot successfully saved to "
            f"{filepath}"
        )

    except Exception as e:
        logger.error(
            f"Failed to take screenshot: {e}",
            exc_info=True,
        )

        return (
            f"Error taking screenshot: {str(e)}"
        )


@register_tool(name="click_mouse")
def click_mouse(
    x: int,
    y: int,
    button: str = "left",
    clicks: int = 1,
) -> str:
    """
    Move the mouse to screen coordinates and click.

    Supported buttons:
        left
        middle
        right
    """

    try:
        x = int(x)
        y = int(y)
        clicks = int(clicks)

    except (TypeError, ValueError):
        return (
            "Error clicking mouse: "
            "x, y, and clicks must be valid numbers."
        )

    button = str(
        button
    ).strip().lower()

    valid_buttons = {
        "left",
        "middle",
        "right",
    }

    if button not in valid_buttons:
        return (
            "Error clicking mouse: "
            f"unsupported button '{button}'. "
            "Use left, middle, or right."
        )

    if clicks <= 0:
        return (
            "Error clicking mouse: "
            "clicks must be greater than zero."
        )

    if clicks > 10:
        return (
            "Error clicking mouse: "
            "clicks cannot be greater than 10."
        )

    width, height = _get_screen_dimensions()

    if not (0 <= x < width):
        return (
            f"Error clicking mouse: "
            f"x must be between 0 and {width - 1}."
        )

    if not (0 <= y < height):
        return (
            f"Error clicking mouse: "
            f"y must be between 0 and {height - 1}."
        )

    try:
        pyautogui.click(
            x=x,
            y=y,
            clicks=clicks,
            button=button,
        )

        logger.info(
            f"Mouse click: "
            f"button={button}, "
            f"x={x}, "
            f"y={y}, "
            f"clicks={clicks}"
        )

        return (
            f"Clicked {button} mouse button "
            f"at ({x}, {y}) "
            f"{clicks} time(s)."
        )

    except Exception as e:
        logger.error(
            f"Failed to click mouse: {e}",
            exc_info=True,
        )

        return (
            f"Error clicking mouse: {str(e)}"
        )


@register_tool(name="type_text")
def type_text(
    text: str,
    interval: float = 0.05,
) -> str:
    """
    Type text using the keyboard.
    """

    if text is None:
        return (
            "Error typing text: "
            "text cannot be empty."
        )

    text = str(text)

    if not text:
        return (
            "Error typing text: "
            "text cannot be empty."
        )

    if len(text) > MAX_TYPE_TEXT_LENGTH:
        return (
            "Error typing text: "
            f"text cannot exceed "
            f"{MAX_TYPE_TEXT_LENGTH} characters."
        )

    try:
        interval = float(interval)

    except (TypeError, ValueError):
        return (
            "Error typing text: "
            "interval must be a valid number."
        )

    if interval < 0:
        return (
            "Error typing text: "
            "interval cannot be negative."
        )

    if interval > 5:
        return (
            "Error typing text: "
            "interval cannot be greater than 5 seconds."
        )

    try:
        pyautogui.write(
            text,
            interval=interval,
        )

        logger.info(
            f"Typed {len(text)} characters."
        )

        return (
            f"Successfully typed: '{text}'"
        )

    except Exception as e:
        logger.error(
            f"Failed to type text: {e}",
            exc_info=True,
        )

        return (
            f"Error typing text: {str(e)}"
        )


@register_tool(name="press_key")
def press_key(
    key: str,
) -> str:
    """
    Press a supported keyboard key.
    """

    if not key or not str(key).strip():
        return (
            "Error pressing key: "
            "key cannot be empty."
        )

    key = str(
        key
    ).strip().lower()

    valid_keys = {
        "enter",
        "tab",
        "space",
        "backspace",
        "delete",
        "escape",
        "esc",
        "shift",
        "ctrl",
        "control",
        "alt",
        "win",
        "windows",
        "command",
        "cmd",
        "up",
        "down",
        "left",
        "right",
        "home",
        "end",
        "pageup",
        "pagedown",
        "insert",
        "capslock",
        "numlock",
        "scrolllock",
        "pause",
        "printscreen",
        "f1",
        "f2",
        "f3",
        "f4",
        "f5",
        "f6",
        "f7",
        "f8",
        "f9",
        "f10",
        "f11",
        "f12",
    }

    if (
        len(key) == 1
        and key.isalnum()
    ):
        is_valid = True
    else:
        is_valid = key in valid_keys

    if not is_valid:
        return (
            f"Error pressing key: "
            f"unsupported key '{key}'."
        )

    if key == "escape":
        key = "esc"

    if key == "control":
        key = "ctrl"

    if key in {
        "windows",
        "command",
        "cmd",
    }:
        key = "win"

    try:
        pyautogui.press(key)

        logger.info(
            f"Pressed keyboard key: {key}"
        )

        return (
            f"Pressed key: '{key}'"
        )

    except Exception as e:
        logger.error(
            f"Failed to press key '{key}': {e}",
            exc_info=True,
        )

        return (
            f"Error pressing key: {str(e)}"
        )


@register_tool(name="get_screen_size")
def get_screen_size() -> str:
    """
    Return the primary display resolution.
    """

    try:
        width, height = (
            _get_screen_dimensions()
        )

        return (
            f"Screen resolution: "
            f"{width}x{height} pixels."
        )

    except Exception as e:
        logger.error(
            f"Failed to get screen size: {e}",
            exc_info=True,
        )

        return (
            f"Error getting screen size: {str(e)}"
        )