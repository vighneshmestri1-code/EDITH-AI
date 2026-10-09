import asyncio
import uuid
from pathlib import Path

from automation.scheduler import TaskScheduler
from tools.registry import register_tool
from utils.logger import logger

try:
    import winsound
except ImportError:
    winsound = None


global_scheduler = TaskScheduler()


MAX_REMINDER_MESSAGE_LENGTH = 500

REMINDER_SFX_PATH = (
    Path(__file__).resolve().parent.parent
    / "audio"
    / "edith_reminder_sfx.wav"
)


def _play_reminder_sound() -> None:
    """Play EDITH's custom WAV alert on Windows, if available."""

    if winsound is None:
        logger.warning(
            "EDITH reminder sound is unavailable because winsound is not installed."
        )
        return

    if not REMINDER_SFX_PATH.is_file():
        logger.warning(
            "EDITH reminder sound file was not found at "
            f"'{REMINDER_SFX_PATH}'."
        )
        return

    winsound.PlaySound(
        str(REMINDER_SFX_PATH),
        winsound.SND_FILENAME,
    )


async def _reminder_callback(
    message: str,
) -> None:
    """Callback executed when a scheduled reminder fires."""

    logger.info(
        f"REMINDER TRIGGERED: {message}"
    )

    print(
        f"\n🔔 [EDITH AUTOMATION REMINDER]: "
        f"{message}\n",
        flush=True,
    )

    try:
        # Run blocking WAV playback off the asyncio event-loop thread.
        await asyncio.to_thread(
            _play_reminder_sound
        )
    except Exception as e:
        # Keep the original reminder behavior even if sound playback fails.
        logger.warning(
            f"Could not play EDITH reminder sound: {e}",
            exc_info=True,
        )


@register_tool(name="schedule_reminder")
async def schedule_reminder(
    message: str,
    delay_seconds: int,
) -> str:
    """
    Schedule a one-time background reminder.

    Args:
        message: Reminder message.
        delay_seconds: Number of seconds before the reminder fires.

    Returns:
        Scheduler result message.
    """

    if not message or not str(message).strip():
        return (
            "Reminder message cannot be empty."
        )

    message = str(
        message
    ).strip()

    if len(message) > MAX_REMINDER_MESSAGE_LENGTH:
        return (
            "Reminder message is too long. "
            f"Maximum length is "
            f"{MAX_REMINDER_MESSAGE_LENGTH} characters."
        )

    try:
        delay_seconds = int(
            delay_seconds
        )

    except (TypeError, ValueError):
        return (
            "Delay must be a valid number of seconds."
        )

    if delay_seconds <= 0:
        return (
            "Delay must be greater than zero seconds."
        )

    task_id = (
        f"reminder_"
        f"{uuid.uuid4().hex[:8]}"
    )

    try:
        result = (
            global_scheduler.add_one_time_task(
                _reminder_callback,
                delay_seconds=delay_seconds,
                task_id=task_id,
                message=message,
            )
        )

        logger.info(
            f"Reminder scheduled: "
            f"id={task_id}, "
            f"delay={delay_seconds}s, "
            f"message='{message}'"
        )

        return str(
            result
        )

    except Exception as e:
        logger.error(
            f"Failed to schedule reminder: {e}",
            exc_info=True,
        )

        return (
            "I couldn't schedule the reminder "
            "because of an error."
        )
