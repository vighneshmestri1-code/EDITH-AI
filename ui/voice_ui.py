from __future__ import annotations

import asyncio
import inspect
import os
import sys
import threading
import tkinter as tk
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from audio.stt import STTEngine
from audio.tts import TTSEngine
from core.orchestrator import EdithOrchestrator
from ui.gif_viewer import GIFViewer
from utils.logger import logger


STOP_KEYWORDS = {
    "exit",
    "quit",
    "stop",
    "goodbye",
    "go to sleep",
    "shut down",
}


class EDITHVoiceUI:
    """Main EDITH voice UI and animation window."""

    def __init__(self, root: tk.Tk, gif_path: Path) -> None:
        self.root = root
        self.gif_path = gif_path
        self.stop_event = threading.Event()
        self.worker: threading.Thread | None = None
        self.viewer: GIFViewer | None = None
        self._shutdown_started = False
        self._status_lock = threading.Lock()
        self._pending_status = "ONLINE"

        self.root.title("EDITH AI")
        self.root.configure(bg="black")
        self.root.overrideredirect(True)
        self.root.attributes("-topmost", True)
        self.root.protocol("WM_DELETE_WINDOW", self.request_stop)
        self.root.bind("<Escape>", self._escape_pressed)

        print(f"EDITH UI: Using GIF -> {self.gif_path}")

        self.viewer = GIFViewer(
            self.root,
            self.gif_path,
            title="EDITH AI",
        )
        self.viewer.pack()

        self._center_window()
        self.root.deiconify()
        self.root.lift()
        self.root.update_idletasks()
        self.viewer.start_animation()

        # Tkinter state updates always happen on the Tk main thread.
        self.root.after(50, self._process_status)

    def _center_window(self) -> None:
        self.root.update_idletasks()

        width = self.viewer.winfo_reqwidth() if self.viewer else 600
        height = self.viewer.winfo_reqheight() if self.viewer else 600

        if width <= 1 or height <= 1:
            width, height = 600, 600

        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()

        x = max(0, (screen_width - width) // 2)
        y = max(0, (screen_height - height) // 2)

        self.root.geometry(f"{width}x{height}+{x}+{y}")

    def _escape_pressed(self, _event: tk.Event) -> str:
        self.request_stop()
        return "break"

    def start(self) -> None:
        if self.worker is not None and self.worker.is_alive():
            return

        # Keep this thread non-daemon so Python cannot tear down its executors
        # while STT/TTS is still working. main() now joins it during shutdown.
        self.worker = threading.Thread(
            target=self._run_async_worker,
            name="edith-voice-worker",
            daemon=False,
        )
        self.worker.start()
        self.root.after(100, self._watch_worker)

    def _run_async_worker(self) -> None:
        try:
            asyncio.run(self.run_voice_loop())
        except Exception as exc:
            if self.stop_event.is_set():
                logger.debug(
                    f"Voice worker ended during shutdown: {exc}",
                    exc_info=True,
                )
            else:
                logger.error(
                    f"Voice worker terminated unexpectedly: {exc}",
                    exc_info=True,
                )

    def _watch_worker(self) -> None:
        if self._shutdown_started:
            return

        if self.worker is not None and self.worker.is_alive():
            try:
                self.root.after(100, self._watch_worker)
            except tk.TclError:
                pass
            return

        self._worker_finished()

    def _worker_finished(self) -> None:
        if self._shutdown_started:
            return

        self._shutdown_started = True

        if self.viewer is not None:
            try:
                self.viewer.stop_animation()
            except tk.TclError:
                pass

        try:
            self.root.quit()
        except tk.TclError:
            pass

    def request_stop(self) -> None:
        if self.stop_event.is_set():
            return

        self.stop_event.set()
        logger.info("EDITH voice interface stop requested.")

    def set_status(self, text: str) -> None:
        # This may be called by the async worker thread. Do not touch Tk here.
        with self._status_lock:
            self._pending_status = text

    def _process_status(self) -> None:
        if self._shutdown_started:
            return

        with self._status_lock:
            status = self._pending_status

        if self.viewer is not None:
            try:
                self.viewer.set_status(status)
            except tk.TclError:
                return

        try:
            self.root.after(50, self._process_status)
        except tk.TclError:
            pass

    async def run_voice_loop(self) -> None:
        orchestrator = EdithOrchestrator(session_id="voice_session")
        stt = STTEngine()
        tts = TTSEngine()

        print(
            "\n=========================================\n"
            "      EDITH AI - Voice Interface\n"
            "=========================================\n"
            "🎤 Voice mode initialized.\n"
            "🧠 Local Faster-Whisper STT enabled when available.\n"
            "🔊 Neural TTS enabled when available.\n"
            "=========================================\n"
        )

        try:
            self.set_status("ONLINE")
            await tts.speak_async("EDITH online and listening.")
            await asyncio.sleep(0.5)

            while not self.stop_event.is_set():
                self.set_status("LISTENING")

                user_speech = await stt.listen_async(
                    timeout=10,
                    phrase_time_limit=15,
                )

                # A stop request can arrive while STT is waiting for speech.
                # Do not process that result or start another listening cycle.
                if self.stop_event.is_set():
                    break

                if not user_speech:
                    continue

                cleaned_speech = user_speech.strip()
                if not cleaned_speech:
                    continue

                logger.info(f"User Speech Input: {cleaned_speech}")
                print(f"\n🎤 You: {cleaned_speech}")

                speech_lower = cleaned_speech.lower()

                if speech_lower in STOP_KEYWORDS:
                    shutdown_msg = "Standing by. Goodbye, Boss."
                    self.request_stop()
                    self.set_status("SPEAKING")
                    print(f"🔊 EDITH: {shutdown_msg}")
                    await tts.speak_async(shutdown_msg)
                    break

                self.set_status("THINKING")
                logger.info(
                    "Sending voice request to EDITH orchestrator..."
                )

                response = await orchestrator.process_turn(cleaned_speech)

                if self.stop_event.is_set():
                    break

                if not response:
                    logger.warning("EDITH returned an empty response.")
                    self.set_status("LISTENING")
                    continue

                response = response.strip()
                if not response:
                    self.set_status("LISTENING")
                    continue

                self.set_status("SPEAKING")
                print(f"🔊 EDITH: {response}")
                await tts.speak_async(response)

                if not self.stop_event.is_set():
                    await asyncio.sleep(0.35)

        except asyncio.CancelledError:
            logger.info("Voice loop task cancelled.")
            raise

        except KeyboardInterrupt:
            logger.info("Voice session interrupted.")
            if not self.stop_event.is_set():
                try:
                    self.set_status("SPEAKING")
                    await tts.speak_async("Voice session interrupted.")
                except Exception:
                    logger.debug(
                        "Could not speak interruption message.",
                        exc_info=True,
                    )

        except Exception as exc:
            logger.error(
                f"Error in voice loop: {exc}",
                exc_info=True,
            )

            # Do not start another TTS request after a shutdown has begun.
            if not self.stop_event.is_set():
                try:
                    self.set_status("SPEAKING")
                    await tts.speak_async(
                        "I encountered an error processing that request."
                    )
                except Exception:
                    logger.debug(
                        "Could not speak the voice-loop error message.",
                        exc_info=True,
                    )

        finally:
            # The event loop stays alive until all engine cleanup finishes.
            for engine in (stt, tts):
                close_method = getattr(engine, "close", None)
                if close_method is None:
                    continue

                try:
                    result = close_method()
                    if inspect.isawaitable(result):
                        await result
                except Exception:
                    logger.debug(
                        "Voice engine cleanup failed.",
                        exc_info=True,
                    )

            logger.info("Voice interface loop ended.")


def resolve_gif_path() -> Path:
    env_path = os.getenv("EDITH_GIF_PATH")
    if env_path:
        candidate = Path(env_path).expanduser().resolve()
        if candidate.is_file():
            return candidate

    candidates = [
        PROJECT_ROOT / "ui" / "edith_animation.gif",
        PROJECT_ROOT / "ui" / "edith_orb.gif",
        PROJECT_ROOT / "assets" / "edith_animation.gif",
        PROJECT_ROOT / "assets" / "edith_orb.gif",
        PROJECT_ROOT / "media" / "edith_animation.gif",
        PROJECT_ROOT / "media" / "edith_orb.gif",
    ]

    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()

    expected = PROJECT_ROOT / "ui" / "edith_animation.gif"
    raise FileNotFoundError(
        "EDITH animation GIF not found. Expected: "
        f"{expected}"
    )


def main() -> None:
    root: tk.Tk | None = None
    app: EDITHVoiceUI | None = None

    try:
        gif_path = resolve_gif_path()
        print(f"EDITH UI: Resolved GIF path -> {gif_path}")

        root = tk.Tk()
        app = EDITHVoiceUI(root, gif_path)
        app.start()
        root.mainloop()

    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received; requesting graceful shutdown.")
        if app is not None:
            app.request_stop()

    except FileNotFoundError as exc:
        print(f"EDITH UI error: {exc}")
        sys.exit(1)

    except Exception as exc:
        logger.error(
            f"Failed to start EDITH voice UI: {exc}",
            exc_info=True,
        )
        sys.exit(1)

    finally:
        # Never destroy the UI / finish the main thread while the voice worker
        # is still using STT, TTS, or its asyncio event loop.
        if app is not None:
            app.request_stop()
            if app.worker is not None and app.worker.is_alive():
                logger.info("Waiting for EDITH voice worker to finish...")
                try:
                    app.worker.join()
                except KeyboardInterrupt:
                    # A second Ctrl+C should not trigger interpreter shutdown
                    # while the non-daemon worker still owns live resources.
                    logger.warning(
                        "Shutdown is waiting for the active voice operation to finish."
                    )
                    app.worker.join()

            app._worker_finished()

        if root is not None:
            try:
                root.destroy()
            except tk.TclError:
                pass


if __name__ == "__main__":
    main()
