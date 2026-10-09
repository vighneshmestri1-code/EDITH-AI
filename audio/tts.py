import asyncio
import os
import re
import tempfile
import threading

from utils.logger import logger


try:
    import edge_tts
    import pygame

    EDGE_TTS_AVAILABLE = True

except ImportError:
    EDGE_TTS_AVAILABLE = False
    edge_tts = None
    pygame = None


try:
    import pyttsx3

except ImportError:
    pyttsx3 = None


class TTSEngine:
    """
    EDITH Text-To-Speech Engine.

    Primary:
        Microsoft Edge-TTS neural voice

    Fallback:
        pyttsx3 offline speech engine
    """

    def __init__(
        self,
        voice: str = "en-GB-RyanNeural",
        rate: str = "+10%",
    ):
        self.voice = voice
        self.rate = rate

        self.engine = None

        self.use_edge = EDGE_TTS_AVAILABLE

        self._playback_lock = threading.Lock()

        if self.use_edge:

            try:
                if not pygame.mixer.get_init():
                    pygame.mixer.init()

                logger.info(
                    "TTSEngine initialized using "
                    "Edge-TTS neural voice."
                )

            except Exception as e:

                logger.warning(
                    "Failed to initialize pygame mixer: "
                    f"{e}. Falling back to pyttsx3."
                )

                self.use_edge = False

        if not self.use_edge:
            self._init_pyttsx3()

    def _init_pyttsx3(
        self,
    ) -> None:
        """Initialize the offline pyttsx3 fallback."""

        if pyttsx3 is None:

            logger.error(
                "pyttsx3 is not installed. "
                "Voice synthesis disabled."
            )

            return

        try:

            self.engine = pyttsx3.init()

            self.engine.setProperty(
                "rate",
                185,
            )

            self.engine.setProperty(
                "volume",
                0.95,
            )

            voices = self.engine.getProperty(
                "voices"
            )

            selected_voice = None

            for voice in voices:

                voice_name = str(
                    getattr(
                        voice,
                        "name",
                        "",
                    )
                ).lower()

                voice_id = str(
                    getattr(
                        voice,
                        "id",
                        "",
                    )
                ).lower()

                if (
                    "english" in voice_name
                    or "english" in voice_id
                ):
                    selected_voice = voice
                    break

            if selected_voice is None and voices:
                selected_voice = voices[0]

            if selected_voice is not None:

                self.engine.setProperty(
                    "voice",
                    selected_voice.id,
                )

            logger.info(
                "TTSEngine initialized using "
                "pyttsx3 offline fallback."
            )

        except Exception as e:

            logger.error(
                f"Failed to initialize pyttsx3: {e}",
                exc_info=True,
            )

            self.engine = None

    def _clean_text_for_speech(
        self,
        text: str,
    ) -> str:
        """
        Remove Markdown and other content that should not
        be spoken aloud.
        """

        if not text:
            return ""

        text = str(
            text
        )

        text = re.sub(
            r"```[\s\S]*?```",
            "",
            text,
        )

        text = re.sub(
            r"`([^`]+)`",
            r"\1",
            text,
        )

        text = re.sub(
            r"https?://\S+|www\.\S+",
            "",
            text,
        )

        text = re.sub(
            r"^\s*#{1,6}\s*",
            "",
            text,
            flags=re.MULTILINE,
        )

        text = re.sub(
            r"^\s*[-*+]\s+",
            "",
            text,
            flags=re.MULTILINE,
        )

        text = re.sub(
            r"^\s*\d+\.\s+",
            "",
            text,
            flags=re.MULTILINE,
        )

        text = re.sub(
            r"[*_]{1,3}",
            "",
            text,
        )

        text = re.sub(
            r"\|",
            " ",
            text,
        )

        text = re.sub(
            r"\s+",
            " ",
            text,
        ).strip()

        return text

    async def speak_async(
        self,
        text: str,
    ) -> None:
        """
        Speak text asynchronously.

        Audio playback is serialized so multiple EDITH
        responses cannot interrupt each other.
        """

        clean_text = self._clean_text_for_speech(
            text
        )

        if not clean_text:
            return

        if self.use_edge:

            temp_file_path = None

            try:

                with tempfile.NamedTemporaryFile(
                    suffix=".mp3",
                    delete=False,
                ) as temp_file:

                    temp_file_path = (
                        temp_file.name
                    )

                logger.info(
                    "EDITH synthesizing neural voice..."
                )

                communicate = edge_tts.Communicate(
                    clean_text,
                    self.voice,
                    rate=self.rate,
                )

                await communicate.save(
                    temp_file_path
                )

                await asyncio.to_thread(
                    self._play_audio_file,
                    temp_file_path,
                )

                return

            except Exception as e:

                logger.error(
                    f"Edge-TTS failed: {e}. "
                    "Using pyttsx3 fallback.",
                    exc_info=True,
                )

                await self._speak_pyttsx3_async(
                    clean_text
                )

            finally:

                if (
                    temp_file_path
                    and os.path.exists(
                        temp_file_path
                    )
                ):

                    try:
                        os.remove(
                            temp_file_path
                        )

                    except OSError:
                        pass

        else:

            await self._speak_pyttsx3_async(
                clean_text
            )

    def _play_audio_file(
        self,
        file_path: str,
    ) -> None:
        """
        Play an Edge-TTS audio file.

        Playback is protected by a lock so only one
        audio response can use pygame at a time.
        """

        with self._playback_lock:

            try:

                pygame.mixer.music.load(
                    file_path
                )

                pygame.mixer.music.play()

                while pygame.mixer.music.get_busy():

                    pygame.time.Clock().tick(
                        10
                    )

                pygame.mixer.music.unload()

            except Exception as e:

                logger.error(
                    f"Error during audio playback: {e}",
                    exc_info=True,
                )

    async def _speak_pyttsx3_async(
        self,
        text: str,
    ) -> None:
        """Run pyttsx3 speech without blocking asyncio."""

        if not self.engine:

            logger.warning(
                "Voice synthesis unavailable: "
                f"{text}"
            )

            return

        await asyncio.to_thread(
            self._speak_pyttsx3_sync,
            text,
        )

    def _speak_pyttsx3_sync(
        self,
        text: str,
    ) -> None:
        """Speak using the synchronous pyttsx3 engine."""

        with self._playback_lock:

            try:

                logger.info(
                    "EDITH speaking using "
                    "offline pyttsx3..."
                )

                self.engine.say(
                    text
                )

                self.engine.runAndWait()

            except Exception as e:

                logger.error(
                    f"Error during pyttsx3 playback: {e}",
                    exc_info=True,
                )

    def speak(
        self,
        text: str,
    ) -> None:
        """
        Synchronous compatibility wrapper.

        When called inside an active asyncio event loop,
        speech is scheduled asynchronously.
        When called without an active loop, speech runs
        synchronously.
        """

        clean_text = self._clean_text_for_speech(
            text
        )

        if not clean_text:
            return

        try:
            loop = asyncio.get_running_loop()

        except RuntimeError:

            if self.use_edge:

                asyncio.run(
                    self.speak_async(
                        clean_text
                    )
                )

            else:

                self._speak_pyttsx3_sync(
                    clean_text
                )

            return

        loop.create_task(
            self.speak_async(
                clean_text
            )
        )