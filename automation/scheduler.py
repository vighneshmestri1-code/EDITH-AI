import asyncio
from datetime import datetime, timedelta
from typing import Any, Callable

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.date import DateTrigger
from apscheduler.triggers.interval import IntervalTrigger

from utils.logger import logger


class TaskScheduler:
    """
    Background task scheduler for EDITH.

    Uses APScheduler's AsyncIOScheduler so scheduled tasks
    can execute without blocking EDITH's main asyncio loop.

    Note:
        This scheduler uses the APScheduler 3.x API and is intended
        to run while EDITH's asyncio event loop is active.
    """

    def __init__(self):
        self.scheduler: AsyncIOScheduler | None = None

    def _ensure_started(self) -> None:
        """
        Start the scheduler when an active asyncio event loop
        is available.
        """

        if (
            self.scheduler is not None
            and self.scheduler.running
        ):
            return

        try:
            loop = asyncio.get_running_loop()

        except RuntimeError as e:
            raise RuntimeError(
                "TaskScheduler must be started "
                "inside a running asyncio event loop."
            ) from e

        try:
            self.scheduler = AsyncIOScheduler(
                event_loop=loop
            )

            self.scheduler.start()

            logger.info(
                "Automation TaskScheduler started."
            )

        except Exception as e:
            self.scheduler = None

            logger.error(
                f"Failed to start Automation TaskScheduler: {e}",
                exc_info=True,
            )

            raise

    @staticmethod
    def _normalize_task_id(
        task_id: str,
    ) -> str:
        """
        Validate and normalize a scheduler task ID.
        """

        if not task_id or not str(task_id).strip():
            raise ValueError(
                "Task ID is required."
            )

        normalized_id = str(
            task_id
        ).strip()

        if len(normalized_id) > 128:
            raise ValueError(
                "Task ID cannot exceed 128 characters."
            )

        return normalized_id

    def add_recurring_task(
        self,
        func: Callable,
        minutes: int,
        task_id: str,
        *args: Any,
        **kwargs: Any,
    ) -> str:
        """
        Schedule a task to execute repeatedly every N minutes.
        """

        if not callable(func):
            return (
                "Failed to schedule task: "
                "invalid callback."
            )

        try:
            normalized_id = (
                self._normalize_task_id(
                    task_id
                )
            )

        except ValueError as e:
            return (
                f"Failed to schedule task: {e}"
            )

        try:
            minutes = int(
                minutes
            )

        except (TypeError, ValueError):
            return (
                "Failed to schedule task: "
                "minutes must be a valid number."
            )

        if minutes <= 0:
            return (
                "Failed to schedule task: "
                "minutes must be greater than zero."
            )

        try:
            self._ensure_started()

            self.scheduler.add_job(
                func,
                trigger=IntervalTrigger(
                    minutes=minutes
                ),
                id=normalized_id,
                args=args,
                kwargs=kwargs,
                replace_existing=True,
                misfire_grace_time=30,
            )

            logger.info(
                f"Scheduled recurring task "
                f"'{normalized_id}' every "
                f"{minutes} minutes."
            )

            return (
                f"Task '{normalized_id}' scheduled "
                f"every {minutes} minutes."
            )

        except Exception as e:
            logger.error(
                f"Error scheduling recurring task "
                f"'{normalized_id}': {e}",
                exc_info=True,
            )

            return (
                f"Failed to schedule task: {str(e)}"
            )

    def add_one_time_task(
        self,
        func: Callable,
        delay_seconds: int,
        task_id: str,
        *args: Any,
        **kwargs: Any,
    ) -> str:
        """
        Schedule a task to execute once after a
        specified delay.
        """

        if not callable(func):
            return (
                "Failed to schedule task: "
                "invalid callback."
            )

        try:
            normalized_id = (
                self._normalize_task_id(
                    task_id
                )
            )

        except ValueError as e:
            return (
                f"Failed to schedule task: {e}"
            )

        try:
            delay_seconds = int(
                delay_seconds
            )

        except (TypeError, ValueError):
            return (
                "Failed to schedule task: "
                "delay_seconds must be a valid number."
            )

        if delay_seconds <= 0:
            return (
                "Failed to schedule task: "
                "delay_seconds must be greater than zero."
            )

        try:
            self._ensure_started()

            run_time = (
                datetime.now()
                + timedelta(
                    seconds=delay_seconds
                )
            )

            self.scheduler.add_job(
                func,
                trigger=DateTrigger(
                    run_date=run_time
                ),
                id=normalized_id,
                args=args,
                kwargs=kwargs,
                replace_existing=True,
                misfire_grace_time=30,
            )

            logger.info(
                f"Scheduled one-time task "
                f"'{normalized_id}' at "
                f"{run_time.strftime('%Y-%m-%d %H:%M:%S')}."
            )

            return (
                f"Task '{normalized_id}' scheduled "
                f"for execution in "
                f"{delay_seconds} seconds."
            )

        except Exception as e:
            logger.error(
                f"Error scheduling one-time task "
                f"'{normalized_id}': {e}",
                exc_info=True,
            )

            return (
                f"Failed to schedule task: {str(e)}"
            )

   
    def stop(self) -> None:
        """
        Stop the scheduler cleanly, even if its event loop
        has already closed.
        """

        scheduler = self.scheduler

        if scheduler is None:
            return

        try:
            event_loop = getattr(
                scheduler,
                "_eventloop",
                None,
            )

            # The asyncio loop may already be closed during
            # EDITH shutdown. In that case, shutdown cannot
            # safely be scheduled on that loop.
            if (
                event_loop is not None
                and event_loop.is_closed()
            ):
                logger.warning(
                    "Automation TaskScheduler event loop "
                    "is already closed; skipping shutdown."
                )
                return

            if scheduler.running:
                scheduler.shutdown(wait=False)

                logger.info(
                    "Automation TaskScheduler stopped."
                )

        except RuntimeError as e:
            # Handle a possible race where the event loop
            # closes after the check above.
            if "event loop is closed" in str(e).lower():
                logger.warning(
                    "Automation TaskScheduler event loop "
                    "closed during shutdown; skipping cleanup."
                )
            else:
                logger.error(
                    f"Error stopping Automation TaskScheduler: {e}",
                    exc_info=True,
                )

        except Exception as e:
            logger.error(
                f"Error stopping Automation TaskScheduler: {e}",
                exc_info=True,
            )

        finally:
            self.scheduler = None
