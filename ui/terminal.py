import asyncio

from config.settings import settings
from core.orchestrator import EdithOrchestrator
from utils.logger import logger


async def start_terminal_chat() -> None:
    """Start EDITH's interactive terminal session."""

    orchestrator = EdithOrchestrator(
        session_id="terminal_session"
    )

    try:
        # Verify Ollama is available before starting.
        if not await orchestrator.brain.is_alive_async():
            logger.critical(
                "Cannot start terminal chat: "
                "Ollama server is offline."
            )

            print(
                "\n❌ EDITH cannot start because "
                "the Ollama server is offline.\n"
            )

            return

        print("\n" + "=" * 50)
        print(
            f"  {settings.APP_NAME} "
            f"v{settings.VERSION} "
            "— Interactive Session"
        )
        print(
            "  Type 'exit', 'quit', or 'clear' "
            "to manage the chat."
        )
        print("=" * 50 + "\n")

        while True:
            try:
                user_input = (
                    await asyncio.to_thread(
                        input,
                        "User > ",
                    )
                ).strip()

                if not user_input:
                    continue

                command = user_input.lower()

                if command in {
                    "exit",
                    "quit",
                }:
                    print(
                        "\nShutting down session. "
                        "Goodbye!"
                    )
                    break

                if command == "clear":

                    cleared = await asyncio.to_thread(
                        orchestrator.reset_conversation
                    )

                    if cleared:
                        print(
                            "\n[Context cleared]\n"
                        )
                    else:
                        print(
                            "\n[Context could not be cleared. "
                            "Check the logs.]\n"
                        )

                    continue

                response = (
                    await orchestrator.process_turn(
                        user_input
                    )
                )

                print(
                    f"\nEDITH > {response}\n"
                )

            except (
                KeyboardInterrupt,
                EOFError,
            ):
                print(
                    "\nSession interrupted. "
                    "Exiting..."
                )
                break

            except Exception as e:
                logger.error(
                    f"Unexpected terminal error: {e}",
                    exc_info=True,
                )

                print(
                    "\n❌ EDITH encountered an "
                    "unexpected error. "
                    "Please check the logs.\n"
                )

    finally:
        try:
            await orchestrator.brain.close()

        except Exception as e:
            logger.warning(
                f"Failed to close EDITH resources cleanly: {e}"
            )