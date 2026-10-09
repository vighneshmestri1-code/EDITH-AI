import asyncio
import sys
import time

from brain.ollama_client import OllamaClient
from config.settings import settings
from core.orchestrator import EdithOrchestrator
from utils.logger import logger


async def async_input(
    prompt: str = "",
) -> str:
    """Read terminal input without blocking the event loop."""

    return await asyncio.to_thread(
        input,
        prompt,
    )


def stop_automation_scheduler() -> None:
    """
    Stop EDITH's global automation scheduler if it has
    been initialized.
    """

    try:
        from automation.triggers import global_scheduler

        global_scheduler.stop()

    except ImportError:
        # Automation tools may not have been imported yet.
        return

    except Exception as e:
        logger.warning(
            f"Failed to stop automation scheduler cleanly: {e}"
        )


async def close_orchestrator(
    orchestrator: EdithOrchestrator | None,
) -> None:
    """Close resources owned by an EDITH orchestrator."""

    if orchestrator is None:
        return

    try:
        await orchestrator.brain.close()

    except Exception as e:
        logger.warning(
            f"Failed to close EDITH Ollama resources: {e}"
        )


async def prepare_orchestrator(
    session_id: str,
) -> EdithOrchestrator:
    """
    Create one EDITH orchestrator and warm the local
    Ollama model before the first real request.
    """

    orchestrator = EdithOrchestrator(
        session_id=session_id
    )

    logger.info(
        "Preparing EDITH language model..."
    )

    start_time = time.perf_counter()

    warmed = await orchestrator.brain.warm_up()

    elapsed = (
        time.perf_counter()
        - start_time
    )

    if warmed:

        logger.info(
            f"Ollama model warm-up completed "
            f"in {elapsed:.2f}s."
        )

    else:

        logger.warning(
            f"Ollama model warm-up did not complete "
            f"successfully after {elapsed:.2f}s. "
            "EDITH will continue normally."
        )

    return orchestrator


async def run_cli_loop(
    orchestrator: EdithOrchestrator,
) -> None:
    """Run EDITH's interactive streaming CLI."""

    try:

        print(
            "\n=================================================="
        )

        print(
            f"        {settings.APP_NAME} "
            f"v{settings.VERSION} - CLI Mode"
        )

        print(
            "=================================================="
        )

        print(
            "Type 'exit', 'quit', or 'stop' "
            "to end the session.\n"
        )

        while True:

            try:

                user_input = (
                    await async_input(
                        "[You]: "
                    )
                ).strip()

                if not user_input:
                    continue

                if user_input.lower() in {
                    "exit",
                    "quit",
                    "stop",
                    "goodbye",
                }:

                    exit_msg = (
                        "Systems standing by. "
                        "Goodbye, Boss."
                    )

                    print(
                        f"\n[EDITH]: {exit_msg}"
                    )

                    break

                print(
                    "\n[EDITH]: ",
                    end="",
                    flush=True,
                )

                async for chunk in (
                    orchestrator.process_turn_stream(
                        user_input
                    )
                ):

                    if chunk:

                        print(
                            chunk,
                            end="",
                            flush=True,
                        )

                print(
                    "\n"
                )

            except (
                KeyboardInterrupt,
                EOFError,
            ):

                print(
                    "\n\n[EDITH]: "
                    "Session interrupted. "
                    "Goodbye, Boss."
                )

                break

            except Exception as e:

                logger.error(
                    f"Error in CLI loop: {e}",
                    exc_info=True,
                )

                print(
                    "\n❌ Error processing command. "
                    "Check the EDITH logs for details.\n"
                )

    finally:

        await close_orchestrator(
            orchestrator
        )


async def run_single_command(
    command: str,
) -> None:
    """Execute a single command supplied through CLI arguments."""

    orchestrator: EdithOrchestrator | None = None

    try:

        orchestrator = await prepare_orchestrator(
            session_id="cli_single_cmd"
        )

        print(
            f"\n[You]: {command}"
        )

        print(
            "\n[EDITH]: ",
            end="",
            flush=True,
        )

        async for chunk in (
            orchestrator.process_turn_stream(
                command
            )
        ):

            if chunk:

                print(
                    chunk,
                    end="",
                    flush=True,
                )

        print(
            "\n"
        )

    finally:

        await close_orchestrator(
            orchestrator
        )


async def verify_ollama() -> bool:
    """
    Verify Ollama connectivity and local model availability.

    Returns:
        True when the local Ollama server and configured
        model are available.
    """

    brain = OllamaClient()

    try:

        logger.info(
            f"Checking Ollama server at "
            f"{brain.host}..."
        )

        if not await brain.is_alive_async():

            logger.critical(
                f"Ollama server is unreachable at "
                f"{brain.host}."
            )

            print(
                "\n❌ Ollama server is offline."
            )

            print(
                "Please start Ollama and run EDITH again.\n"
            )

            return False

        logger.info(
            "Ollama server connection: ONLINE"
        )

        if not await brain.is_model_available_async():

            logger.critical(
                f"Required model '{brain.model_name}' "
                "is not available locally."
            )

            print(
                f"\n❌ Model '{brain.model_name}' "
                "was not found locally."
            )

            print(
                f"Run: ollama pull "
                f"{brain.model_name}\n"
            )

            return False

        logger.info(
            f"Local LLM model "
            f"'{brain.model_name}': READY"
        )

        return True

    finally:

        await brain.close()


async def main() -> None:
    """Initialize EDITH and launch the selected interface."""

    logger.info(
        f"Initializing {settings.APP_NAME} "
        f"v{settings.VERSION}..."
    )

    try:

        # 1. Verify local Ollama and the configured model.
        ollama_ready = await verify_ollama()

        if not ollama_ready:
            return

        # 2. Direct command-line execution.
        if len(sys.argv) > 1:

            command = " ".join(
                sys.argv[1:]
            ).strip()

            if command:

                await run_single_command(
                    command
                )

            return

        # 3. Mode selection.
        print(
            "\n=================================================="
        )

        print(
            f"        {settings.APP_NAME} "
            "SYSTEM CORE INITIALIZATION"
        )

        print(
            "=================================================="
        )

        print(
            "Select Operating Mode:"
        )

        print(
            "  [1] Voice Interface Mode "
            "(Microphone + Speaker)"
        )

        print(
            "  [2] Text CLI Mode "
            "(Terminal Interactive)"
        )

        print(
            "  [3] Exit System"
        )

        choice = (
            await async_input(
                "\nEnter choice (1-3) "
                "[Default: 2]: "
            )
        ).strip()

        if not choice:
            choice = "2"

        if choice == "1":

            try:

                from ui.voice_ui import main as run_voice_ui

                run_voice_ui()

            except ImportError as e:

                logger.error(
                    f"Failed to import Voice UI: {e}",
                    exc_info=True,
                )

                print(
                    "\n❌ Voice module unavailable."
                )

                print(
                    "Falling back to CLI Mode...\n"
                )

                orchestrator = (
                    await prepare_orchestrator(
                        session_id="cli_master_session"
                    )
                )

                await run_cli_loop(
                    orchestrator
                )

            return

        if choice == "3":

            print(
                "\n[EDITH]: Systems offline."
            )

            return

        if choice != "2":

            print(
                "\n❌ Invalid choice. "
                "Please restart EDITH and select 1, 2, or 3."
            )

            return

        # Default mode: Text CLI.
        orchestrator = (
            await prepare_orchestrator(
                session_id="cli_master_session"
            )
        )

        await run_cli_loop(
            orchestrator
        )

    finally:

        stop_automation_scheduler()


if __name__ == "__main__":

    try:

        asyncio.run(
            main()
        )

    except (
        KeyboardInterrupt,
        SystemExit,
    ):

        logger.info(
            "EDITH shut down gracefully."
        )