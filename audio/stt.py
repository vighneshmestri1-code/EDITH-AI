"""EDITH speech-to-text: local Faster-Whisper behind an always-on microphone stream.

This module keeps EDITH's STT boundary intentionally simple:

    microphone -> capture/VAD -> quality checks -> Faster-Whisper -> transcript

The transcript is never rewritten toward an expected command. Vocabulary bias
and alternate decoding are used only to help Whisper choose what was actually
spoken. Microphone arrays can be probed automatically so EDITH uses an active,
quieter input channel instead of blindly assuming channel 0.

Public API:
    STTEngine.listen(), listen_detailed(), listen_async(),
    listen_detailed_async(), mute(), unmute(), stop_listening(), close().

Run `python -m audio.stt --help` from the project root for diagnostics.
"""

from __future__ import annotations

import argparse
import asyncio
import atexit
import dataclasses
import gc
import importlib
import inspect
import math
import os
import queue
import re
import threading
import time
import wave
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from difflib import SequenceMatcher
from math import gcd
from numbers import Integral
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import sounddevice as sd

from utils.logger import logger


try:
    from faster_whisper import WhisperModel

    FASTER_WHISPER_AVAILABLE = True
except ImportError:
    WhisperModel = None
    FASTER_WHISPER_AVAILABLE = False

try:
    from faster_whisper.vad import VadOptions
except Exception:
    VadOptions = None

try:
    import ctranslate2
except ImportError:
    ctranslate2 = None

try:
    from scipy.signal import butter, resample_poly, sosfilt, sosfiltfilt

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


SAMPLE_RATE = 16000
_MISSING = object()


def _settings_sources() -> List[Any]:
    try:
        module = importlib.import_module("config.settings")
    except Exception:
        return []

    instance = getattr(module, "settings", None)
    return [instance, module] if instance is not None else [module]


def _lookup(sources: Sequence[Any], name: str) -> Any:
    for source in sources:
        if hasattr(source, name):
            return getattr(source, name)
    return _MISSING


def _coerce(value: Any, default: Any) -> Any:
    if isinstance(default, bool):
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        return bool(value)

    if isinstance(default, (int, float)):
        try:
            return type(default)(value)
        except (TypeError, ValueError):
            return default

    return value


def _default_input_index() -> Optional[int]:
    """Return the current PortAudio default input index safely.

    sounddevice versions expose `sd.default.device` as an input/output pair in
    some environments. Treating that object as a scalar can raise when EDITH
    tries `int(default_device)`, which previously caused fallback to device 0.
    """
    try:
        default = getattr(sd.default, "device", None)

        candidate = getattr(default, "input", None)
        if candidate is None and isinstance(default, (tuple, list)):
            candidate = default[0] if default else None
        if candidate is None:
            try:
                candidate = default[0]
            except (TypeError, IndexError, KeyError):
                candidate = default

        if candidate is None:
            return None

        candidate = int(candidate)
        return candidate if candidate >= 0 else None
    except Exception as exc:
        logger.debug(f"Could not read PortAudio default input device: {exc}")
        return None


def _input_device_candidates(devices: Sequence[Any], preferred: Optional[int]) -> List[int]:
    """Rank input devices so virtual capture endpoints are not chosen first."""
    virtual_terms = (
        "sound mapper",
        "primary sound capture",
        "stereo mix",
        "what u hear",
        "wave out mix",
        "loopback",
        "monitor",
    )
    physical_terms = (
        "microphone",
        "mic",
        "headset",
        "headphone mic",
        "usb",
        "array",
    )
    wasapi_hint = "windows wasapi"

    ranked: List[Tuple[int, int]] = []
    for index, device in enumerate(devices):
        try:
            if int(device.get("max_input_channels", 0)) <= 0:
                continue
            name = str(device.get("name", "")).lower()
            hostapi = str(device.get("hostapi", "")).lower()
            score = 0

            if preferred is not None and index == preferred:
                score += 10000
            if any(term in name for term in virtual_terms):
                score -= 250
            if any(term in name for term in physical_terms):
                score += 50
            if wasapi_hint in hostapi:
                score += 10

            ranked.append((score, index))
        except Exception:
            continue

    ranked.sort(key=lambda item: (-item[0], item[1]))
    return [index for _, index in ranked]


@dataclass
class STTConfig:
    """Every field maps to an optional STT_<FIELD> constant in settings."""

    model: str = "small.en"
    device: str = "auto"
    compute_type: str = "auto"
    cpu_threads: int = 0
    language: Optional[str] = "en"
    beam_size: Optional[int] = 5
    vad_filter: bool = True
    vad_threshold: float = 0.35
    hotwords: Optional[str] = (
        "hello, hi, yes, no, open, close, launch, start, stop, play, search, find, "
        "switch, type, call, send, take, screenshot, browser, weather, capital, "
        "Chrome, Google, YouTube, WhatsApp, Calculator, Python, Windows, EDITH"
    )
    precision_pass: bool = True
    precision_beam_size: int = 8
    precision_short_seconds: float = 1.8
    precision_min_confidence: float = 0.62
    precision_short_confidence: float = 0.78
    min_silence_duration: int = 500
    speech_pad_ms: int = 350
    no_speech_threshold: float = 0.75
    log_prob_threshold: float = -1.3
    compression_ratio_threshold: float = 2.4
    initial_prompt: Optional[str] = None
    end_silence: float = 0.70
    end_silence_short: float = 0.50
    continuation_window: float = 1.5
    speculative_decode: bool = False
    local_files_only: bool = False
    mic_index: Optional[int] = None
    mic_channel: Optional[int] = None
    save_debug_audio: bool = True
    console_output: bool = True

    def __post_init__(self) -> None:
        self.device = str(self.device or "auto").strip().lower()
        if self.device not in {"auto", "cuda", "cpu"}:
            logger.warning(f"Unknown STT_DEVICE {self.device!r}; using 'auto'.")
            self.device = "auto"

        self.compute_type = str(self.compute_type or "auto").strip().lower()

        if self.language is not None:
            self.language = str(self.language).strip().lower() or None

        if self.hotwords is not None:
            self.hotwords = str(self.hotwords).strip() or None

        if self.initial_prompt is not None:
            self.initial_prompt = str(self.initial_prompt).strip() or None

        self.precision_pass = bool(self.precision_pass)
        self.speculative_decode = bool(self.speculative_decode)

        try:
            self.cpu_threads = max(0, int(self.cpu_threads or 0))
        except (TypeError, ValueError):
            self.cpu_threads = 0

        try:
            self.precision_beam_size = max(1, int(self.precision_beam_size or 8))
        except (TypeError, ValueError):
            self.precision_beam_size = 8

        for name, lower, upper, default in (
            ("precision_short_seconds", 0.5, 10.0, 1.8),
            ("precision_min_confidence", 0.0, 0.99, 0.62),
            ("precision_short_confidence", 0.0, 0.99, 0.78),
        ):
            try:
                value = float(getattr(self, name))
                setattr(self, name, min(upper, max(lower, value)))
            except (TypeError, ValueError):
                setattr(self, name, default)

        for name in ("beam_size", "mic_index", "mic_channel"):
            value = getattr(self, name)
            if value is None:
                continue
            try:
                value = int(value)
            except (TypeError, ValueError):
                value = None
            setattr(self, name, value)

        if self.beam_size is not None:
            self.beam_size = max(1, self.beam_size)
        if self.mic_index is not None and self.mic_index < 0:
            self.mic_index = None
        if self.mic_channel is not None and self.mic_channel < 0:
            self.mic_channel = None

        try:
            self.vad_threshold = min(0.99, max(0.05, float(self.vad_threshold)))
        except (TypeError, ValueError):
            self.vad_threshold = 0.35

        try:
            self.min_silence_duration = min(
                3000, max(100, int(self.min_silence_duration))
            )
        except (TypeError, ValueError):
            self.min_silence_duration = 500

        try:
            self.speech_pad_ms = min(1000, max(0, int(self.speech_pad_ms)))
        except (TypeError, ValueError):
            self.speech_pad_ms = 350

        for name, lower, upper, default in (
            ("no_speech_threshold", 0.05, 0.99, 0.75),
            ("compression_ratio_threshold", 1.5, 5.0, 2.4),
        ):
            try:
                value = float(getattr(self, name))
                setattr(self, name, min(upper, max(lower, value)))
            except (TypeError, ValueError):
                setattr(self, name, default)

        try:
            self.log_prob_threshold = min(0.0, max(-5.0, float(self.log_prob_threshold)))
        except (TypeError, ValueError):
            self.log_prob_threshold = -1.3

        try:
            self.end_silence = min(max(float(self.end_silence), 0.30), 2.0)
        except (TypeError, ValueError):
            self.end_silence = 0.70

        try:
            self.end_silence_short = min(
                max(float(self.end_silence_short), 0.25), self.end_silence
            )
        except (TypeError, ValueError):
            self.end_silence_short = min(0.50, self.end_silence)

        try:
            self.continuation_window = min(5.0, max(0.0, float(self.continuation_window)))
        except (TypeError, ValueError):
            self.continuation_window = 1.5

    @classmethod
    def from_settings(cls) -> "STTConfig":
        sources = _settings_sources()
        base = cls()
        values: Dict[str, Any] = {}

        for field in dataclasses.fields(cls):
            raw = _lookup(sources, f"STT_{field.name.upper()}")
            if raw is _MISSING:
                continue

            default = getattr(base, field.name)
            values[field.name] = (
                raw if raw is None or default is None else _coerce(raw, default)
            )

        return cls(**values)


def normalize_text(text: str) -> str:
    """Comparison form only; returned transcripts are never normalized this way."""
    text = re.sub(r"[^\w\s']", " ", str(text).lower())
    return re.sub(r"\s+", " ", text).strip()


def rms(audio: np.ndarray) -> float:
    if audio is None or audio.size == 0:
        return 0.0
    array = np.asarray(audio, dtype=np.float32)
    return float(np.sqrt(np.mean(np.square(array))))


def resample_audio(audio: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    """Anti-aliased resample via scipy, with a numpy fallback."""
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)

    if audio.size == 0 or source_rate == target_rate:
        return audio
    if source_rate <= 0 or target_rate <= 0:
        raise ValueError("Invalid audio sample rate.")

    source_rate = int(source_rate)
    target_rate = int(target_rate)

    if SCIPY_AVAILABLE:
        divisor = gcd(source_rate, target_rate)
        out = resample_poly(audio, target_rate // divisor, source_rate // divisor)
        return np.asarray(out, dtype=np.float32)

    if target_rate < source_rate:
        cutoff = 0.40 * target_rate / source_rate
        taps = 161
        n = np.arange(taps) - (taps - 1) / 2
        kernel = 2 * cutoff * np.sinc(2 * cutoff * n) * np.hamming(taps)
        kernel /= kernel.sum()
        audio = np.convolve(audio, kernel, mode="same").astype(np.float32)

    target_length = int(round(audio.size * target_rate / source_rate))
    if target_length <= 1:
        return audio[:1].copy()

    positions = np.arange(target_length) * (source_rate / target_rate)
    return np.interp(positions, np.arange(audio.size), audio).astype(np.float32)


_HIGHPASS_SOS = (
    butter(2, 80.0, btype="highpass", fs=SAMPLE_RATE, output="sos")
    if SCIPY_AVAILABLE
    else None
)


def preprocess_audio(
    audio: np.ndarray,
    target_rms: float = 0.045,
    max_gain: float = 2.5,
    reference_rms: Optional[float] = None,
) -> np.ndarray:
    """Apply conservative DC removal, high-pass filtering and optional gain.

    `reference_rms` should be the detected speech RMS when available. Using
    the whole clip RMS tends to boost the silence around a command as much as
    the command itself, so capture-aware normalization is preferred.
    """
    audio = np.nan_to_num(
        np.asarray(audio, dtype=np.float32).reshape(-1),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    if audio.size == 0:
        return audio

    audio = audio - float(np.mean(audio))

    if _HIGHPASS_SOS is not None and audio.size >= 64:
        try:
            audio = sosfiltfilt(_HIGHPASS_SOS, audio).astype(np.float32)
        except ValueError:
            audio = sosfilt(_HIGHPASS_SOS, audio).astype(np.float32)

    reference = float(reference_rms or 0.0)
    level = reference if reference > 1e-4 else rms(audio)

    if 1e-4 < level < target_rms:
        gain = min(target_rms / level, max_gain)
        audio = audio * gain

    return np.clip(audio, -1.0, 1.0).astype(np.float32)


def write_wav(path: Path, audio: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    audio = np.nan_to_num(
        np.asarray(audio, dtype=np.float32).reshape(-1),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    audio = np.clip(audio, -1.0, 1.0)
    pcm = np.rint(audio * 32767.0).astype(np.int16)

    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(SAMPLE_RATE)
        handle.writeframes(pcm.tobytes())


def read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as handle:
        channels = handle.getnchannels()
        width = handle.getsampwidth()
        rate = handle.getframerate()
        frames = handle.readframes(handle.getnframes())

    if width != 2:
        raise ValueError(f"{path}: only 16-bit PCM WAV files are supported")

    audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)

    return resample_audio(audio, rate, SAMPLE_RATE)


class _BandFilter:
    """Stateful high-pass filter used only for speech-level detection."""

    def __init__(self, rate: int, cutoff: float):
        self._sos = None
        self._zi = None
        if SCIPY_AVAILABLE and rate > cutoff * 4:
            self._sos = butter(2, cutoff, btype="highpass", fs=rate, output="sos")

    def reset(self) -> None:
        self._zi = None

    def level(self, block: np.ndarray) -> float:
        if self._sos is None:
            return rms(block)

        if self._zi is None:
            self._zi = np.zeros((self._sos.shape[0], 2), dtype=np.float64)

        filtered, self._zi = sosfilt(self._sos, block, zi=self._zi)
        return rms(filtered)


@dataclass
class _Decode:
    text: str = ""
    avg_logprob: Optional[float] = None
    no_speech_prob: Optional[float] = None
    speech_seconds: float = 0.0
    decode_ms: float = 0.0
    had_speech: bool = True
    rejected: int = 0
    error: bool = False


@dataclass
class _Capture:
    audio: np.ndarray
    voiced_seconds: float
    speech_rms: float
    last_voice_at: float = 0.0
    pending: Optional[Future] = None


@dataclass
class _Processed:
    audio: np.ndarray
    reason: Optional[str] = None
    decode: Optional[_Decode] = None


@dataclass
class Transcription:
    """
    Structured result of listen_detailed().

    status values:
        ok | timeout | noise | unintelligible | no_mic | no_model |
        busy | cancelled | error

    `confidence` is a log-probability-derived quality proxy, not a calibrated
    probability of correctness.
    """

    text: str = ""
    status: str = "timeout"
    avg_logprob: Optional[float] = None
    no_speech_prob: Optional[float] = None
    confidence: Optional[float] = None
    speech_seconds: float = 0.0
    audio_seconds: float = 0.0
    decode_ms: float = 0.0
    response_ms: float = 0.0
    attempts: int = 0
    continued: bool = False

    def __bool__(self) -> bool:
        return bool(self.text)


class WhisperTranscriber:
    """Load Faster-Whisper once and keep decoding decisions conservative."""

    GPU_ERROR_HINTS = ("cuda", "cublas", "cudnn", "out of memory", "driver", "nvrtc")
    CUDA_COMPUTE_PREFERENCE = ("int8_float16", "float16", "int8")
    CPU_COMPUTE_PREFERENCE = ("int8", "int8_float32", "float32")

    WEAK_NO_SPEECH = 0.30
    WEAK_AVG_LOGPROB = -0.70

    HALLUCINATION_PHRASES = frozenset(
        {
            "thank you",
            "thank you very much",
            "thank you so much",
            "thanks",
            "thanks for watching",
            "thank you for watching",
            "please subscribe",
            "subscribe",
            "bye",
            "bye bye",
            "you",
            "the end",
            "uh",
            "um",
            "hmm",
        }
    )

    def __init__(self, config: STTConfig, prompt: Optional[str] = None):
        self.config = config
        self.model = None
        self.device = "cpu"
        self.compute_type = ""
        self.beam_size = 1
        self.prompt: Optional[str] = (prompt or "").strip() or None
        self._prompt_norm = normalize_text(self.prompt or "")
        self._lock = threading.Lock()
        self._accepted_keys: Optional[frozenset] = None
        self._warned: set = set()

    @staticmethod
    def cuda_available() -> bool:
        if ctranslate2 is None:
            return False
        try:
            return ctranslate2.get_cuda_device_count() > 0
        except Exception:
            return False

    @staticmethod
    def _supported_types(device: str) -> Optional[set]:
        if ctranslate2 is None:
            return None
        try:
            return set(ctranslate2.get_supported_compute_types(device))
        except Exception:
            return None

    def _compute_candidates(self, device: str) -> List[str]:
        preference = (
            self.CUDA_COMPUTE_PREFERENCE
            if device == "cuda"
            else self.CPU_COMPUTE_PREFERENCE
        )
        supported = self._supported_types(device)
        requested = self.config.compute_type

        order: List[str] = []
        if requested != "auto":
            if supported is None or requested in supported:
                order.append(requested)
            else:
                logger.warning(
                    f"STT compute type {requested!r} is not supported on {device}; "
                    "choosing automatically."
                )
        order.extend(preference)

        result: List[str] = []
        for compute_type in order:
            if compute_type in result:
                continue
            if supported is None or compute_type in supported:
                result.append(compute_type)

        return result or [preference[0]]

    def _plan(self) -> List[Tuple[str, str]]:
        plan: List[Tuple[str, str]] = []
        wanted = self.config.device

        if wanted in ("auto", "cuda"):
            if self.cuda_available():
                plan += [("cuda", t) for t in self._compute_candidates("cuda")]
            elif wanted == "cuda":
                logger.warning(
                    "STT_DEVICE='cuda' requested but no CUDA device is available; "
                    "using CPU."
                )

        plan += [("cpu", t) for t in self._compute_candidates("cpu")]
        return plan

    def _is_gpu_failure(self, exc: Exception) -> bool:
        message = str(exc).lower()
        return any(hint in message for hint in self.GPU_ERROR_HINTS)

    def load(self) -> bool:
        if not FASTER_WHISPER_AVAILABLE:
            logger.error(
                "Faster-Whisper is not installed. Install with: "
                "python -m pip install faster-whisper"
            )
            return False

        skip_cuda = False
        for device, compute_type in self._plan():
            if device == "cuda" and skip_cuda:
                continue

            try:
                self._load_on(device, compute_type)
                return True
            except Exception as exc:
                logger.warning(
                    f"STT model load failed on {device}/{compute_type}: {exc}"
                )

                if device == "cuda" and self._is_gpu_failure(exc):
                    skip_cuda = True
                    logger.warning(
                        "CUDA is unusable (missing cuBLAS/cuDNN, driver, or out of memory). "
                        "Falling back to CPU."
                    )

        logger.error("STT model could not be loaded on any device.")
        return False

    def _load_on(self, device: str, compute_type: str) -> None:
        logger.info(
            f"Loading Whisper '{self.config.model}' on {device} ({compute_type})..."
        )

        kwargs: Dict[str, Any] = {
            "device": device,
            "compute_type": compute_type,
        }
        if self.config.local_files_only:
            kwargs["local_files_only"] = True

        threads = 0
        if device == "cpu" and self._model_accepts("cpu_threads"):
            threads = self._cpu_threads()
            kwargs["cpu_threads"] = threads

        self.model = WhisperModel(self.config.model, **kwargs)
        self.device = device
        self.compute_type = compute_type
        self.beam_size = self.config.beam_size or (5 if device == "cuda" else 3)
        self._accepted_keys = None

        try:
            self._warm_up()
        except Exception as exc:
            if self._is_gpu_failure(exc):
                self.model = None
                raise
            logger.warning(f"STT warm-up skipped: {exc}")

        logger.info(f"STT Device: {device.upper()}")
        logger.info(f"STT Compute Type: {compute_type}")
        logger.info(f"STT Model: {self.config.model}")
        if threads:
            logger.info(f"STT CPU threads: {threads}")

    @staticmethod
    def _model_accepts(name: str) -> bool:
        try:
            params = inspect.signature(WhisperModel).parameters
        except (TypeError, ValueError):
            return False
        return name in params or any(
            p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()
        )

    def _cpu_threads(self) -> int:
        if self.config.cpu_threads > 0:
            return self.config.cpu_threads
        return max(4, min(8, (os.cpu_count() or 8) // 2))

    def _warm_up(self) -> None:
        noise = (
            np.random.default_rng(0)
            .standard_normal(SAMPLE_RATE)
            .astype(np.float32)
            * 0.003
        )

        options = self._options()
        options["beam_size"] = 1
        list(self.model.transcribe(noise, **options)[0])

        bare = dict(options, vad_filter=False)
        bare.pop("vad_parameters", None)
        list(self.model.transcribe(noise, **bare)[0])

    def _vad_parameters(self) -> Dict[str, Any]:
        cfg = self.config
        params = {
            "threshold": cfg.vad_threshold,
            "min_speech_duration_ms": 100,
            "min_silence_duration_ms": cfg.min_silence_duration,
            "speech_pad_ms": cfg.speech_pad_ms,
        }

        if VadOptions is not None and dataclasses.is_dataclass(VadOptions):
            known = {field.name for field in dataclasses.fields(VadOptions)}
            params = {key: value for key, value in params.items() if key in known}

        return params

    def _options(self, precision: bool = False) -> Dict[str, Any]:
        cfg = self.config
        beam_size = (
            max(self.beam_size, cfg.precision_beam_size)
            if precision
            else self.beam_size
        )

        options: Dict[str, Any] = {
            "language": cfg.language,
            "task": "transcribe",
            "beam_size": beam_size,
            "temperature": 0.0,
            "condition_on_previous_text": False,
            "initial_prompt": self.prompt,
            "vad_filter": cfg.vad_filter,
            "without_timestamps": True,
            "suppress_blank": True,
            "compression_ratio_threshold": cfg.compression_ratio_threshold,
            "log_prob_threshold": cfg.log_prob_threshold,
            "no_speech_threshold": cfg.no_speech_threshold,
        }

        if cfg.hotwords:
            options["hotwords"] = cfg.hotwords

        if precision:
            options["vad_filter"] = False

        if cfg.vad_filter and not precision:
            options["vad_parameters"] = self._vad_parameters()

        return self._supported(options)

    def _supported(self, options: Dict[str, Any]) -> Dict[str, Any]:
        """Only pass kwargs understood by the installed Faster-Whisper build."""
        if self._accepted_keys is None:
            try:
                params = inspect.signature(self.model.transcribe).parameters
                if any(
                    parameter.kind is inspect.Parameter.VAR_KEYWORD
                    for parameter in params.values()
                ):
                    self._accepted_keys = frozenset()
                else:
                    self._accepted_keys = frozenset(params)
            except (TypeError, ValueError):
                self._accepted_keys = frozenset()

        if not self._accepted_keys:
            return options

        dropped = sorted(key for key in options if key not in self._accepted_keys)
        if dropped and tuple(dropped) not in self._warned:
            self._warned.add(tuple(dropped))
            logger.warning(
                f"Installed faster-whisper ignores unsupported options: {dropped}"
            )

        return {key: value for key, value in options.items() if key in self._accepted_keys}

    def _reject_reason(
        self,
        segment: Any,
        text: str,
        no_speech_threshold: Optional[float] = None,
        log_prob_threshold: Optional[float] = None,
        compression_ratio_threshold: Optional[float] = None,
    ) -> Optional[str]:
        cfg = self.config
        no_speech_limit = (
            cfg.no_speech_threshold
            if no_speech_threshold is None
            else float(no_speech_threshold)
        )
        logprob_limit = (
            cfg.log_prob_threshold
            if log_prob_threshold is None
            else float(log_prob_threshold)
        )
        compression_limit = (
            cfg.compression_ratio_threshold
            if compression_ratio_threshold is None
            else float(compression_ratio_threshold)
        )

        no_speech = float(getattr(segment, "no_speech_prob", 0.0) or 0.0)
        avg_logprob = float(getattr(segment, "avg_logprob", 0.0) or 0.0)
        compression = float(getattr(segment, "compression_ratio", 0.0) or 0.0)

        # Faster-Whisper treats a segment as silent when BOTH no-speech probability
        # is high and average log probability is low. Do not independently reject
        # a real spoken segment only because no_speech_prob is high.
        if no_speech > no_speech_limit and avg_logprob < logprob_limit:
            return (
                f"no-speech/low-confidence combination "
                f"(no-speech {no_speech:.2f}, logprob {avg_logprob:.2f})"
            )
        if compression > compression_limit:
            return f"repetitive output (compression {compression:.2f})"

        norm = normalize_text(text)
        if not norm:
            return "no words"

        weak = no_speech > self.WEAK_NO_SPEECH or avg_logprob < self.WEAK_AVG_LOGPROB

        if weak and norm in self.HALLUCINATION_PHRASES:
            return "known filler hallucination"

        if self._prompt_norm and norm in self._prompt_norm:
            if weak or len(norm.split()) >= 6:
                return "echo of initial prompt"

        return None

    def _decode(self, audio: np.ndarray, precision: bool = False) -> _Decode:
        decode = _Decode()
        started = time.perf_counter()

        options = self._options(precision=precision)
        segments, info = self.model.transcribe(audio, **options)

        kept: List[str] = []
        logprobs: List[float] = []
        no_speech: List[float] = []

        rejected = 0
        for segment in segments:
            text = str(getattr(segment, "text", "") or "").strip()
            if not text:
                continue

            reason = self._reject_reason(
                segment,
                text,
                no_speech_threshold=options.get(
                    "no_speech_threshold", self.config.no_speech_threshold
                ),
                log_prob_threshold=options.get(
                    "log_prob_threshold", self.config.log_prob_threshold
                ),
                compression_ratio_threshold=options.get(
                    "compression_ratio_threshold",
                    self.config.compression_ratio_threshold,
                ),
            )
            if reason:
                rejected += 1
                logger.info(f"Rejected segment {text!r}: {reason}")
                continue

            kept.append(text)
            logprobs.append(float(getattr(segment, "avg_logprob", 0.0) or 0.0))
            no_speech.append(float(getattr(segment, "no_speech_prob", 0.0) or 0.0))

        decode.decode_ms = (time.perf_counter() - started) * 1000.0
        decode.rejected = rejected

        after_vad = getattr(info, "duration_after_vad", None)
        total = getattr(info, "duration", None)
        decode.had_speech = after_vad is None or float(after_vad) > 0.0
        decode.speech_seconds = float(
            after_vad if after_vad is not None else (total or 0.0)
        )
        decode.text = " ".join(kept).strip()

        if logprobs:
            decode.avg_logprob = float(np.mean(logprobs))
            decode.no_speech_prob = float(np.max(no_speech))

        return decode

    @staticmethod
    def _decode_confidence(decode: _Decode) -> float:
        if decode.error or not decode.text or decode.avg_logprob is None:
            return 0.0
        return float(min(1.0, max(0.0, math.exp(decode.avg_logprob))))

    @staticmethod
    def _word_count(text: str) -> int:
        return len(normalize_text(text).split())

    @staticmethod
    def _text_similarity(first: str, second: str) -> float:
        a = normalize_text(first)
        b = normalize_text(second)
        if not a or not b:
            return 0.0
        return float(SequenceMatcher(None, a, b, autojunk=False).ratio())

    @classmethod
    def _needs_precision_pass(
        cls,
        decode: _Decode,
        audio_seconds: float,
        config: STTConfig,
    ) -> bool:
        if decode.error or not decode.text:
            return True

        confidence = cls._decode_confidence(decode)
        if confidence < config.precision_min_confidence:
            return True

        if audio_seconds <= config.precision_short_seconds:
            return confidence < config.precision_short_confidence

        return False

    @classmethod
    def _choose_decode(cls, first: _Decode, second: _Decode) -> _Decode:
        """Choose conservatively; an alternate decoder must earn the switch."""
        if second.error or not second.text:
            return first
        if first.error or not first.text:
            return second

        first_conf = cls._decode_confidence(first)
        second_conf = cls._decode_confidence(second)
        similarity = cls._text_similarity(first.text, second.text)
        first_words = cls._word_count(first.text)
        second_words = cls._word_count(second.text)

        logger.info(
            "STT decode comparison: "
            f"first_conf={first_conf:.2f}, second_conf={second_conf:.2f}, "
            f"similarity={similarity:.2f}, first_words={first_words}, "
            f"second_words={second_words}"
        )

        # A strong first decode is the safest answer. The precision pass is not
        # allowed to replace it merely because the second log probability is higher.
        if first_conf >= 0.78 and second_conf < first_conf + 0.12:
            return first

        # When both decoders substantially agree, confidence is a reasonable
        # tie-breaker. This is the common case where the second pass confirms it.
        if similarity >= 0.68:
            return second if second_conf > first_conf + 0.03 else first

        # For divergent candidates, require a meaningful confidence improvement.
        # This avoids replacing "open chrome" with a different plausible phrase
        # merely because beam search assigned it a slightly better score.
        if second_conf >= first_conf + 0.10:
            return second

        # A materially longer candidate is helpful only when the first is weak.
        if first_conf < 0.55 and second_words > first_words and second_conf >= first_conf + 0.04:
            return second

        return first

    def _transcribe_once(self, audio: np.ndarray, precision: bool = False) -> _Decode:
        with self._lock:
            if self.model is None:
                return _Decode(error=True)
            return self._decode(audio, precision=precision)

    def _transcribe_pair(self, audio: np.ndarray) -> _Decode:
        first = self._transcribe_once(audio, precision=False)

        if not self.config.precision_pass:
            return first

        audio_seconds = audio.size / SAMPLE_RATE
        if not self._needs_precision_pass(first, audio_seconds, self.config):
            return first

        logger.info(
            "Running STT precision pass "
            f"(audio={audio_seconds:.2f}s, first={first.text!r}, "
            f"confidence={self._decode_confidence(first):.2f})."
        )

        second = self._transcribe_once(audio, precision=True)
        chosen = self._choose_decode(first, second)

        if chosen.text != first.text and not chosen.error:
            logger.info(
                f"Precision pass selected alternate transcript: {first.text!r} -> {chosen.text!r}"
            )

        return chosen

    def transcribe(self, audio: np.ndarray) -> _Decode:
        if self.model is None:
            return _Decode(error=True)

        audio = np.clip(
            np.nan_to_num(
                np.asarray(audio, dtype=np.float32).reshape(-1),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            ),
            -1.0,
            1.0,
        )

        try:
            return self._transcribe_pair(audio)
        except Exception as exc:
            if (
                self.device == "cuda"
                and self._is_gpu_failure(exc)
                and self._fall_back_to_cpu(exc)
            ):
                try:
                    return self._transcribe_pair(audio)
                except Exception as fallback_exc:
                    logger.error(
                        f"Whisper CPU fallback transcription failed: {fallback_exc}",
                        exc_info=True,
                    )
                    return _Decode(error=True)

            logger.error(f"Whisper transcription failed: {exc}", exc_info=True)
            return _Decode(error=True)

    def _fall_back_to_cpu(self, exc: Exception) -> bool:
        logger.warning(
            f"GPU failure during transcription ({exc}); reloading Whisper on CPU."
        )

        with self._lock:
            if self.device != "cuda":
                return self.model is not None

            self.model = None
            gc.collect()

            for compute_type in self._compute_candidates("cpu"):
                try:
                    self._load_on("cpu", compute_type)
                    return True
                except Exception as err:
                    logger.warning(
                        f"CPU fallback load failed ({compute_type}): {err}"
                    )

        return False

    def close(self) -> None:
        acquired = self._lock.acquire(timeout=5)
        try:
            self.model = None
        finally:
            if acquired:
                self._lock.release()
        gc.collect()


class STTEngine:
    TARGET_SAMPLE_RATE = SAMPLE_RATE
    CHANNELS = 1
    BLOCK_DURATION = 0.032

    DEFAULT_TIMEOUT = 10.0
    DEFAULT_PHRASE_TIME_LIMIT = 15.0

    QUEUE_SECONDS = 30.0
    STREAM_SETTLE_SECONDS = 0.15
    POST_MUTE_COOLDOWN = 0.30
    BLOCK_READ_TIMEOUT = 2.0

    PRE_ROLL_DURATION = 0.80
    TRAILING_PAD = 0.40
    SPECULATE_AFTER = 0.35
    SHORT_UTTERANCE_SECONDS = 1.4
    START_CONFIRMATION_BLOCKS = 2
    START_FOLLOW_RATIO = 0.72

    CALIBRATION_DURATION = 1.2
    CALIBRATION_SKIP = 0.20
    DETECTION_HIGHPASS_HZ = 120.0

    DEAD_AIR_RMS = 5e-5

    MIN_SPEECH_RMS = 0.0030
    MIN_END_RMS = 0.0018
    SPEECH_MULTIPLIER = 1.8
    END_SPEECH_MULTIPLIER = 1.20
    MAX_SPEECH_THRESHOLD = 0.020
    NOISE_RISE_RATE = 0.005
    NOISE_FALL_RATE = 0.05
    MAX_NOISE_RMS = 0.05

    MAX_ATTEMPTS = 3
    MIN_RETRY_WINDOW = 1.5

    INCOMPLETE_TAILS = frozenset(
        {
            "open",
            "launch",
            "start",
            "run",
            "close",
            "search",
            "find",
            "switch",
            "type",
            "call",
            "send",
            "the",
            "a",
            "an",
            "to",
            "for",
            "and",
            "my",
            "of",
            "is",
            "are",
        }
    )

    MIN_AUDIO_DURATION = 0.25
    MIN_VOICED_DURATION = 0.20
    MIN_PEAK = 0.01
    MAX_CLIPPING_RATIO = 0.02

    LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
    DEBUG_AUDIO_FILE = LOG_DIR / "stt_last_capture.wav"
    DEBUG_DIR = LOG_DIR / "stt_debug"
    DEBUG_KEEP = 12

    def __init__(
        self,
        device_index: Optional[int] = None,
        model_size: Optional[str] = None,
        initial_prompt: Optional[str] = None,
        apps: Optional[Dict[str, List[str]]] = None,
        config: Optional[STTConfig] = None,
    ):
        # apps remains accepted for compatibility; STT does not rewrite text.
        self.config = config or STTConfig.from_settings()
        if model_size:
            self.config = dataclasses.replace(self.config, model=model_size)

        self.requested_device_index = (
            device_index if device_index is not None else self.config.mic_index
        )
        if self.requested_device_index is not None:
            try:
                self.requested_device_index = int(self.requested_device_index)
                if self.requested_device_index < 0:
                    self.requested_device_index = None
            except (TypeError, ValueError):
                self.requested_device_index = None

        self.device_index: Optional[int] = None
        self.device_name = ""
        self.device_hostapi = ""
        self.device_sample_rate = float(SAMPLE_RATE)
        self.capture_rate = SAMPLE_RATE
        self.capture_channel = 0
        self.capture_channels = 1
        self.microphone_available = False

        self.noise_rms = 0.0
        self.speech_threshold = self.MIN_SPEECH_RMS
        self.end_speech_threshold = self.MIN_END_RMS
        self._detector = _BandFilter(self.capture_rate, self.DETECTION_HIGHPASS_HZ)

        prompt = initial_prompt if initial_prompt is not None else self.config.initial_prompt
        self.transcriber = WhisperTranscriber(self.config, prompt)

        self._microphone_error_reported = False
        self._model_retry_done = False
        self._closed = False
        self._listen_lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="edith-stt")

        self._stream = None
        self._blocks: "queue.Queue[np.ndarray]" = queue.Queue()
        self._muted = False
        self._unmuted = threading.Event()
        self._unmuted.set()
        self._discard_until = 0.0
        self._status_events = 0
        self._last_capture_failed = False
        self._cancel = threading.Event()
        self.last_result: Optional[Transcription] = None

        logger.info("Initializing EDITH STT engine...")

        if not SCIPY_AVAILABLE:
            logger.warning(
                "scipy not installed: numpy fallback resampler, no high-pass. "
                "Install with: python -m pip install scipy"
            )

        model_ready = self._executor.submit(self.transcriber.load)

        self._setup_microphone()
        if self.microphone_available:
            self._calibrate_microphone()
        else:
            logger.warning("STT initialized without a usable microphone.")

        if model_ready.result():
            if (
                self.transcriber.device == "cpu"
                and self.config.device != "cpu"
                and self.transcriber.cuda_available()
            ):
                logger.warning(
                    "A CUDA device is present but STT is running on CPU. "
                    "Check the earlier model-load messages for the reason."
                )

            self._say(f"STT Device: {self.transcriber.device.upper()}")
            self._say(f"STT Compute Type: {self.transcriber.compute_type}")
            self._say(f"STT Model: {self.config.model}")
            logger.info(
                f"STT initialized (model={self.config.model}, "
                f"device={self.transcriber.device}, "
                f"compute={self.transcriber.compute_type}, "
                f"beam={self.transcriber.beam_size}, "
                f"prompt={'on' if self.transcriber.prompt else 'off'}, "
                f"hotwords={'on' if self.config.hotwords else 'off'}, "
                f"precision_pass={self.config.precision_pass}, "
                f"mic={self.device_index}, "
                f"channel={self.capture_channel})"
            )
        else:
            self._say("Whisper model could not be loaded.")

        atexit.register(self.close)

    @property
    def whisper_model(self):
        return self.transcriber.model

    @property
    def whisper_device(self) -> str:
        return self.transcriber.device

    @property
    def model_size(self) -> str:
        return self.config.model

    def _say(self, message: str) -> None:
        if not self.config.console_output:
            return
        try:
            print(message, flush=True)
        except UnicodeEncodeError:
            print(message.encode("ascii", "replace").decode("ascii"), flush=True)

    def register_apps(self, apps: Dict[str, List[str]]) -> None:
        logger.debug("register_apps() ignored: STT does not rewrite transcripts.")

    def _setup_microphone(self) -> bool:
        try:
            devices = sd.query_devices()
            if not devices:
                self._report_microphone_error("The system returned no audio devices.")
                return False

            default_input = _default_input_index()

            if self.requested_device_index is not None:
                if self._is_valid_input_device(self.requested_device_index):
                    if self._select_microphone(self.requested_device_index):
                        return True
                logger.warning(
                    f"Requested microphone {self.requested_device_index} is unavailable."
                )

            if default_input is not None and self._is_valid_input_device(default_input):
                if self._select_microphone(default_input):
                    return True

            for index in _input_device_candidates(devices, default_input):
                if index == self.requested_device_index or index == default_input:
                    continue
                if self._select_microphone(index):
                    logger.info(f"Selected fallback input device [{index}].")
                    return True

            self._report_microphone_error("No usable input microphone was found.")
            return False

        except Exception as exc:
            logger.error(f"Microphone discovery failed: {exc}", exc_info=True)
            self._report_microphone_error("Unable to access the system audio devices.")
            return False

    def _get_default_input_device(self) -> Optional[int]:
        return _default_input_index()

    def _is_valid_input_device(self, device_index: int) -> bool:
        try:
            device = sd.query_devices(device_index)
            return int(device.get("max_input_channels", 0)) > 0
        except Exception:
            return False

    def _pick_capture_rate(
        self,
        device_index: int,
        native_rate: float,
        channels_required: int = 1,
    ) -> Optional[int]:
        """Choose a rate accepted by the device for the requested channel count."""
        channels_required = max(1, int(channels_required))

        for candidate in (SAMPLE_RATE, int(round(native_rate))):
            if candidate <= 0:
                continue
            try:
                sd.check_input_settings(
                    device=device_index,
                    channels=channels_required,
                    dtype="float32",
                    samplerate=candidate,
                )
                return candidate
            except Exception:
                continue

        return None

    def _choose_capture_channel(
        self,
        device_index: int,
        capture_rate: int,
        max_input_channels: int,
    ) -> Tuple[int, int]:
        """Choose an input channel and how many input channels the stream must open."""
        requested = self.config.mic_channel

        if requested is not None:
            requested = int(requested)
            if requested >= max_input_channels:
                raise ValueError(
                    f"STT_MIC_CHANNEL={requested} is unavailable; "
                    f"device exposes {max_input_channels} input channels."
                )

            required = requested + 1
            if self._pick_capture_rate(device_index, capture_rate, required) is None:
                raise ValueError(
                    f"Microphone channel {requested} cannot be opened at a supported rate."
                )
            return requested, required

        # Mono devices have only channel 0. For microphone arrays, probe a small
        # multi-channel stream during the quiet-room initialization path. Dead
        # channels are excluded; among active channels, the quietest idle channel
        # is preferred because it gives the detector the largest useful margin.
        if max_input_channels < 2:
            return 0, 1

        try:
            probe_channels = min(max_input_channels, 4)
            probe_rate = self._pick_capture_rate(
                device_index,
                capture_rate,
                probe_channels,
            )
            if probe_rate is None:
                for count in (2, 1):
                    if count > max_input_channels:
                        continue
                    probe_rate = self._pick_capture_rate(
                        device_index,
                        capture_rate,
                        count,
                    )
                    if probe_rate is not None:
                        probe_channels = count
                        break

            if probe_rate is None or probe_channels < 2:
                return 0, 1

            duration = 0.35
            samples = max(1, int(round(duration * probe_rate)))
            data = sd.rec(
                samples,
                samplerate=probe_rate,
                channels=probe_channels,
                dtype="float32",
                device=device_index,
            )
            sd.wait()
            data = np.nan_to_num(
                np.asarray(data, dtype=np.float32),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
            if data.ndim != 2 or data.shape[1] < 2:
                return 0, 1

            levels = np.sqrt(np.mean(np.square(data), axis=0))
            levels = np.asarray(levels, dtype=np.float64)
            active = np.flatnonzero(levels >= self.DEAD_AIR_RMS * 2.0)

            if active.size == 0:
                logger.warning(
                    f"Automatic microphone-channel probe found no clearly active channels "
                    f"on device [{device_index}]; using channel 0."
                )
                return 0, 1

            best = int(active[np.argmin(levels[active])])
            logger.info(
                f"Microphone channel probe [{device_index}]: "
                + ", ".join(
                    f"ch{idx}={levels[idx]:.6f}" for idx in range(len(levels))
                )
                + f" -> channel {best}"
            )
            return best, best + 1
        except Exception as exc:
            logger.warning(
                f"Automatic microphone-channel probe failed on device [{device_index}]: "
                f"{exc}. Using channel 0."
            )
            return 0, 1

    def _select_microphone(self, device_index: int) -> bool:
        try:
            device = sd.query_devices(device_index)
            max_input_channels = int(device.get("max_input_channels", 0))
            if max_input_channels <= 0:
                return False

            native_rate = float(device.get("default_samplerate", SAMPLE_RATE))
            if native_rate <= 0:
                native_rate = float(SAMPLE_RATE)

            requested_channel = self.config.mic_channel
            if requested_channel is not None:
                requested_channel = int(requested_channel)
                if requested_channel >= max_input_channels:
                    logger.warning(
                        f"Microphone [{device_index}] has only {max_input_channels} input "
                        f"channels; requested channel {requested_channel} is unavailable."
                    )
                    return False
                channel_counts = [requested_channel + 1]
            else:
                preferred = min(max_input_channels, 4)
                channel_counts = [preferred, 2, 1]

            unique_counts: List[int] = []
            for count in channel_counts:
                count = max(1, min(int(count), max_input_channels))
                if count not in unique_counts:
                    unique_counts.append(count)

            capture_rate: Optional[int] = None
            supported_channels = 1
            for count in unique_counts:
                capture_rate = self._pick_capture_rate(
                    device_index,
                    native_rate,
                    count,
                )
                if capture_rate is not None:
                    supported_channels = count
                    break

            if capture_rate is None:
                logger.warning(
                    f"Microphone [{device_index}] does not accept float32 input "
                    "at 16 kHz or its native rate."
                )
                return False

            # When an explicit channel was requested, validate that exact channel.
            if requested_channel is not None:
                channel = requested_channel
                stream_channels = channel + 1
            elif supported_channels >= 2:
                # Re-probe only the channel count that was actually confirmed above.
                # _choose_capture_channel() will return an active channel within it.
                channel, stream_channels = self._choose_capture_channel(
                    device_index,
                    capture_rate,
                    supported_channels,
                )
            else:
                channel, stream_channels = 0, 1

            if stream_channels > supported_channels:
                logger.warning(
                    f"Selected microphone channel {channel} needs {stream_channels} input "
                    f"channels, but only {supported_channels} could be opened."
                )
                return False

            self.device_index = int(device_index)
            self.device_name = str(device.get("name", "Unknown microphone"))
            hostapi_index = device.get("hostapi")
            try:
                hostapi_index = int(hostapi_index)
                hostapis = sd.query_hostapis()
                self.device_hostapi = str(
                    hostapis[hostapi_index].get("name", hostapi_index)
                )
            except Exception:
                self.device_hostapi = str(hostapi_index or "Unknown")

            self.device_sample_rate = native_rate
            self.capture_rate = capture_rate
            self.capture_channel = channel
            self.capture_channels = stream_channels
            self._detector = _BandFilter(
                self.capture_rate,
                self.DETECTION_HIGHPASS_HZ,
            )

            self.microphone_available = True
            self._microphone_error_reported = False

            logger.info(
                f"Microphone initialized: [{self.device_index}] {self.device_name} "
                f"(hostapi {self.device_hostapi}, native {native_rate:.0f} Hz, "
                f"capture {self.capture_rate} Hz, channel {self.capture_channel}, "
                f"stream channels {self.capture_channels})"
            )
            self._say(
                f"Microphone: {self.device_name} "
                f"(channel {self.capture_channel})"
            )
            return True

        except Exception as exc:
            logger.warning(f"Unable to select microphone {device_index}: {exc}")
            return False

    @staticmethod
    def list_microphones() -> None:
        try:
            devices = sd.query_devices()
            default_input = _default_input_index()
            default_output = None

            try:
                default = getattr(sd.default, "device", None)
                candidate = getattr(default, "output", None)
                if candidate is None and isinstance(default, (tuple, list)):
                    candidate = default[1] if len(default) > 1 else None
                if candidate is not None:
                    candidate = int(candidate)
                    default_output = candidate if candidate >= 0 else None
            except Exception:
                default_output = None

            hostapis = sd.query_hostapis()

            print("\n=== EDITH MICROPHONE DEVICES ===")
            found = False

            for index, device in enumerate(devices):
                try:
                    channels = int(device.get("max_input_channels", 0))
                    if channels <= 0:
                        continue

                    found = True
                    marker = "  <- DEFAULT INPUT" if default_input == index else ""
                    hostapi_index = int(device.get("hostapi", -1))
                    hostapi_name = (
                        str(hostapis[hostapi_index].get("name", hostapi_index))
                        if 0 <= hostapi_index < len(hostapis)
                        else str(hostapi_index)
                    )
                    print(f"[{index}] {device.get('name', 'Unknown')}{marker}")
                    print(
                        f"     channels: {channels}, "
                        f"native rate: {device.get('default_samplerate', 'unknown')}, "
                        f"hostapi: {hostapi_name}"
                    )
                except Exception:
                    continue

            if default_output is not None:
                print(f"Default output device index: {default_output}")
            if not found:
                print("No input microphones detected.")
            print()

        except Exception as exc:
            print(f"Unable to list microphones: {exc}")

    def _report_microphone_error(self, message: str) -> None:
        if self._microphone_error_reported:
            return

        self._microphone_error_reported = True
        logger.error(message)
        self._say(f"Microphone error: {message}")
        self._say("Check microphone permissions and connected input devices.")

    def _refresh_microphone(self) -> bool:
        logger.info("Refreshing microphone...")
        self._stop_stream()
        self.microphone_available = False
        self.device_index = None
        self.device_name = ""
        self.device_hostapi = ""
        self.capture_channel = 0
        self.capture_channels = 1
        return self._setup_microphone()

    def _block_size(self) -> int:
        return max(128, int(self.capture_rate * self.BLOCK_DURATION))

    def _audio_callback(self, indata, frames, time_info, status) -> None:
        """PortAudio callback: copy the selected microphone channel and return immediately."""
        if status:
            self._status_events += 1

        if self._muted or time.monotonic() < self._discard_until:
            return

        try:
            channel = self.capture_channel
            if indata.ndim != 2 or channel >= indata.shape[1]:
                channel = 0
            block = np.array(indata[:, channel], dtype=np.float32, copy=True)
        except Exception:
            return

        q = self._blocks
        try:
            q.put_nowait(block)
        except queue.Full:
            try:
                q.get_nowait()
            except queue.Empty:
                pass
            try:
                q.put_nowait(block)
            except queue.Full:
                pass

    def _ensure_stream(self) -> bool:
        stream = self._stream

        if stream is not None:
            try:
                if stream.active:
                    return True
            except Exception:
                pass
            self._stop_stream()

        if not self.microphone_available:
            return False

        try:
            maxsize = max(8, int(self.QUEUE_SECONDS / self.BLOCK_DURATION))
            self._blocks = queue.Queue(maxsize=maxsize)
            self._discard_until = time.monotonic() + self.STREAM_SETTLE_SECONDS
            self._detector.reset()

            self._stream = sd.InputStream(
                samplerate=int(self.capture_rate),
                blocksize=self._block_size(),
                channels=self.capture_channels,
                dtype="float32",
                device=self.device_index,
                callback=self._audio_callback,
            )
            self._stream.start()

            logger.info("Microphone stream started (always-on).")
            return True

        except Exception as exc:
            logger.error(f"Could not start microphone stream: {exc}", exc_info=True)
            self._stream = None
            if self._is_audio_device_error(exc):
                self.microphone_available = False
            return False

    def _stop_stream(self) -> None:
        stream, self._stream = self._stream, None
        if stream is None:
            return

        for action in (stream.stop, stream.close):
            try:
                action()
            except Exception:
                pass

    def close(self) -> None:
        """Release the microphone and model. Safe to call repeatedly."""
        if self._closed:
            return

        self._closed = True
        self._cancel.set()
        self._stop_stream()
        self._executor.shutdown(wait=False, cancel_futures=True)
        self.transcriber.close()

    def __enter__(self) -> "STTEngine":
        return self

    def __exit__(self, *exc_info) -> bool:
        self.close()
        return False

    def _drain_queue(self) -> None:
        q = self._blocks
        try:
            while True:
                q.get_nowait()
        except queue.Empty:
            pass

    def _next_block(self) -> Optional[np.ndarray]:
        deadline = time.monotonic() + self.BLOCK_READ_TIMEOUT
        while True:
            if self._cancel.is_set():
                return None

            try:
                return self._blocks.get(timeout=0.1)
            except queue.Empty:
                if self._muted:
                    deadline = time.monotonic() + self.BLOCK_READ_TIMEOUT
                elif time.monotonic() >= deadline:
                    raise RuntimeError("PortAudio input stopped delivering audio")

    def mute(self) -> None:
        """Mute capture before EDITH speaks and discard already-buffered audio."""
        self._muted = True
        self._unmuted.clear()
        self._drain_queue()

    def unmute(self, cooldown: Optional[float] = None) -> None:
        """Resume capture after EDITH finishes speaking."""
        wait = self.POST_MUTE_COOLDOWN if cooldown is None else max(0.0, cooldown)
        self._discard_until = time.monotonic() + wait
        self._drain_queue()
        self._muted = False
        self._unmuted.set()

    def stop_listening(self) -> None:
        """Abort the current capture; the result status becomes `cancelled`."""
        self._cancel.set()

    def _update_thresholds(self, noise: float) -> float:
        noise = float(min(max(noise, 0.0), self.MAX_NOISE_RMS))
        self.noise_rms = noise

        dynamic = noise * self.SPEECH_MULTIPLIER
        self.speech_threshold = min(
            self.MAX_SPEECH_THRESHOLD,
            max(self.MIN_SPEECH_RMS, dynamic),
        )

        # End-of-phrase detection must remain comfortably below the start
        # threshold so quiet syllables and word tails do not disappear.
        end = max(self.MIN_END_RMS, noise * self.END_SPEECH_MULTIPLIER)
        self.end_speech_threshold = min(
            end,
            self.speech_threshold * 0.65,
        )
        return noise

    def _calibrate_microphone(self) -> None:
        """Measure the quiet-room baseline once; capture thresholds then adapt slowly."""
        if not self.microphone_available:
            return

        self._say("\nCalibrating microphone...")
        self._say(f"Please remain silent for {self.CALIBRATION_DURATION:.1f} seconds.")

        try:
            if not self._ensure_stream():
                raise RuntimeError("Input stream unavailable.")

            self._drain_queue()
            block_seconds = self._block_size() / self.capture_rate
            total = int(
                math.ceil(
                    (self.CALIBRATION_DURATION + self.CALIBRATION_SKIP) / block_seconds
                )
            )
            skip = int(self.CALIBRATION_SKIP / block_seconds)

            levels: List[float] = []
            for index in range(total):
                block = self._next_block()
                if block is None:
                    raise RuntimeError("Calibration cancelled.")
                level = self._detector.level(block)
                if index >= skip:
                    levels.append(level)

            if not levels:
                raise RuntimeError("Calibration captured no audio.")

            noise = float(np.percentile(levels, 45))

            if noise > self.MAX_NOISE_RMS:
                self._say("Very high background noise (or speech during calibration).")

            if noise < self.DEAD_AIR_RMS:
                logger.warning(
                    f"Microphone is digitally silent between words (noise {noise:.6f}). "
                    "An OS audio gate/noise suppression may be muting the signal."
                )
                self._say("Warning: mic looks digitally gated. Run: python -m audio.stt --mic-check")

            noise = self._update_thresholds(noise)
            logger.info(
                f"Calibration: noise {noise:.6f}, speech threshold "
                f"{self.speech_threshold:.6f}, end threshold {self.end_speech_threshold:.6f}"
            )
            self._say("Microphone ready.")

        except Exception as exc:
            logger.warning(f"Microphone calibration failed: {exc}")
            self._update_thresholds(0.0)

            if self._is_audio_device_error(exc):
                self.microphone_available = False
                self._stop_stream()

            self._say("Calibration failed. Using safe speech thresholds.")

    def listen(
        self,
        timeout: float = DEFAULT_TIMEOUT,
        phrase_time_limit: float = DEFAULT_PHRASE_TIME_LIMIT,
        discard_buffered: bool = True,
    ) -> str:
        return self.listen_detailed(timeout, phrase_time_limit, discard_buffered).text

    def listen_detailed(
        self,
        timeout: float = DEFAULT_TIMEOUT,
        phrase_time_limit: float = DEFAULT_PHRASE_TIME_LIMIT,
        discard_buffered: bool = True,
        cancel: Optional[threading.Event] = None,
    ) -> Transcription:
        if not self._listen_lock.acquire(blocking=False):
            logger.warning("listen() called while another capture is running.")
            return Transcription(status="busy")

        try:
            self._cancel = cancel or threading.Event()
            result = self._listen_once(timeout, phrase_time_limit, discard_buffered)
        finally:
            self._listen_lock.release()

        self.last_result = result
        return result

    async def listen_async(
        self,
        timeout: float = DEFAULT_TIMEOUT,
        phrase_time_limit: float = DEFAULT_PHRASE_TIME_LIMIT,
        discard_buffered: bool = True,
    ) -> str:
        return (
            await self.listen_detailed_async(timeout, phrase_time_limit, discard_buffered)
        ).text

    async def listen_detailed_async(
        self,
        timeout: float = DEFAULT_TIMEOUT,
        phrase_time_limit: float = DEFAULT_PHRASE_TIME_LIMIT,
        discard_buffered: bool = True,
    ) -> Transcription:
        cancel = threading.Event()
        try:
            return await asyncio.to_thread(
                self.listen_detailed,
                timeout,
                phrase_time_limit,
                discard_buffered,
                cancel,
            )
        except asyncio.CancelledError:
            cancel.set()
            raise

    def _ensure_ready(self, result: Transcription) -> bool:
        if self._closed:
            result.status = "error"
            return False

        if not self.microphone_available:
            if not self._refresh_microphone():
                result.status = "no_mic"
                return False
            self._calibrate_microphone()

        if self.transcriber.model is None and not self._model_retry_done:
            self._model_retry_done = True
            self._executor.submit(self.transcriber.load).result()

        if self.transcriber.model is None:
            self._say("Speech recognition engine is unavailable.")
            result.status = "no_model"
            return False

        if not self._ensure_stream():
            result.status = "no_mic"
            return False

        return True

    def _failure_status(self) -> str:
        if self._cancel.is_set():
            return "cancelled"
        if self._last_capture_failed:
            return "error" if self.microphone_available else "no_mic"
        return "timeout"

    def _listen_once(
        self,
        timeout: float,
        phrase_time_limit: float,
        discard_buffered: bool,
    ) -> Transcription:
        started = time.monotonic()
        result = Transcription()
        last_voice_at = 0.0

        try:
            if self._closed:
                result.status = "error"
                return result

            timeout = max(0.1, float(timeout))
            phrase_time_limit = max(0.5, float(phrase_time_limit))

            if not self._unmuted.wait(timeout):
                result.status = "timeout"
                return result

            if not self._ensure_ready(result):
                return result

            if discard_buffered:
                self._drain_queue()

            logger.info("Listening...")
            announce = True

            for attempt in range(1, self.MAX_ATTEMPTS + 1):
                remaining = timeout - (time.monotonic() - started)
                if attempt > 1 and remaining < self.MIN_RETRY_WINDOW:
                    break

                result.attempts = attempt

                capture = self._capture_speech(
                    timeout=timeout if attempt == 1 else max(remaining, self.MIN_RETRY_WINDOW),
                    phrase_time_limit=phrase_time_limit,
                    verbose=announce,
                    speculate=self.config.speculative_decode,
                )
                announce = False

                if capture is None:
                    result.status = self._failure_status()
                    break

                last_voice_at = capture.last_voice_at
                logger.info("Transcribing...")
                self._say("Transcribing...")

                future = capture.pending or self._executor.submit(self._process, capture)
                processed = future.result()

                audio = processed.audio
                result.audio_seconds = audio.size / SAMPLE_RATE

                if processed.reason:
                    result.status = "noise"
                    logger.info(f"Audio refused: {processed.reason}")
                    self._archive(audio, "noise")
                    self._say(f"{processed.reason.capitalize()}.")
                    continue

                decode = processed.decode
                if decode is None or decode.error:
                    result.status = "error"
                    self._archive(audio, "error")
                    break

                if not decode.text:
                    result.status = "unintelligible" if decode.had_speech else "noise"
                    logger.info("No usable speech detected.")
                    self._archive(audio, result.status)
                    continue

                if (
                    self.config.continuation_window > 0
                    and self._looks_incomplete(decode.text)
                ):
                    merged = self._try_continuation(audio, decode, phrase_time_limit)
                    if merged is not None:
                        audio, decode, last_voice_at = merged
                        result.continued = True
                        result.audio_seconds = audio.size / SAMPLE_RATE

                text = self._clean_text(decode.text)
                if not text:
                    result.status = "unintelligible"
                    continue

                result.status = "ok"
                result.text = text
                result.avg_logprob = decode.avg_logprob
                result.no_speech_prob = decode.no_speech_prob
                result.confidence = (
                    math.exp(decode.avg_logprob)
                    if decode.avg_logprob is not None
                    else None
                )
                result.speech_seconds = decode.speech_seconds
                result.decode_ms = decode.decode_ms
                result.response_ms = (
                    max(0.0, (time.monotonic() - last_voice_at) * 1000.0)
                    if last_voice_at
                    else 0.0
                )

                self._archive(audio, f"ok_{text}")
                self._say(f"You said: {text}")
                break

            if result.status in ("noise", "unintelligible"):
                self._say("I couldn't understand that.")

            return result

        except Exception as exc:
            logger.error(f"STT error: {exc}", exc_info=True)
            if self._is_audio_device_error(exc):
                self.microphone_available = False
                self._stop_stream()
            self._say("Speech input could not be processed.")
            result.status = "error"
            return result

        finally:
            confidence = (
                "n/a" if result.confidence is None else f"{result.confidence:.2f}"
            )
            logger.info(
                f"STT result: status={result.status} text={result.text!r} "
                f"confidence={confidence} audio={result.audio_seconds:.2f}s "
                f"speech={result.speech_seconds:.2f}s decode={result.decode_ms:.0f}ms "
                f"response={result.response_ms:.0f}ms attempts={result.attempts} "
                f"continued={result.continued} total={time.monotonic() - started:.2f}s"
            )

    def _looks_incomplete(self, text: str) -> bool:
        tokens = normalize_text(text).split()
        return bool(tokens) and tokens[-1] in self.INCOMPLETE_TAILS

    def _try_continuation(
        self,
        audio: np.ndarray,
        first: _Decode,
        phrase_time_limit: float,
    ) -> Optional[Tuple[np.ndarray, _Decode, float]]:
        used = audio.size / SAMPLE_RATE
        extra = self._capture_speech(
            timeout=self.config.continuation_window,
            phrase_time_limit=max(2.0, phrase_time_limit - used),
            verbose=False,
            speculate=False,
        )

        if extra is None or extra.voiced_seconds < self.MIN_VOICED_DURATION:
            return None

        extra_audio = resample_audio(extra.audio, self.capture_rate, SAMPLE_RATE)
        merged = np.concatenate([audio, extra_audio])
        decode = self._decode_audio(merged, speech_rms=extra.speech_rms)

        if decode.error or not decode.text:
            return None

        first_words = self.transcriber._word_count(first.text)
        merged_words = self.transcriber._word_count(decode.text)
        first_conf = self.transcriber._decode_confidence(first)
        merged_conf = self.transcriber._decode_confidence(decode)

        if merged_words < first_words and merged_conf < first_conf + 0.04:
            return None

        logger.info(f"Continuation merged: {first.text!r} -> {decode.text!r}")
        return merged, decode, extra.last_voice_at

    def _capture_speech(
        self,
        timeout: float,
        phrase_time_limit: float,
        verbose: bool = True,
        speculate: bool = False,
    ) -> Optional[_Capture]:
        """Capture one phrase using the live stream and adaptive level detector."""
        self._last_capture_failed = False
        if not self._ensure_stream():
            self._last_capture_failed = True
            return None

        block_s = self._block_size() / self.capture_rate

        def blocks(seconds: float) -> int:
            return max(1, int(round(seconds / block_s)))

        pre_roll_blocks = blocks(self.PRE_ROLL_DURATION)
        tail_blocks = blocks(self.TRAILING_PAD)
        end_long = blocks(self.config.end_silence)
        end_short = blocks(self.config.end_silence_short)
        short_voiced = blocks(self.SHORT_UTTERANCE_SECONDS)
        speculate_at = (
            min(blocks(self.SPECULATE_AFTER), max(1, end_short - 2))
            if speculate
            else 0
        )
        max_phrase_blocks = max(1, int(phrase_time_limit / block_s))
        timeout_blocks = max(1, int(timeout / block_s))

        pre_roll: deque = deque(maxlen=pre_roll_blocks)
        recent: deque = deque(maxlen=3)
        recorded: List[np.ndarray] = []
        levels: List[float] = []
        observed_peak = 0.0

        noise = self.noise_rms
        started = False
        loud = 0
        silent_run = 0
        recording_blocks = 0
        waited = 0
        voiced_blocks = 0
        last_voice_at = 0.0
        pending: Optional[Future] = None

        try:
            if verbose:
                self._say("\nEDITH is listening...")

            while True:
                block = self._next_block()
                if block is None:
                    if pending is not None:
                        pending.cancel()
                    return None
                if block.size == 0:
                    continue

                level = self._detector.level(block)
                if level > observed_peak:
                    observed_peak = level

                if not started:
                    pre_roll.append((block, level))
                    waited += 1

                    threshold = self.speech_threshold
                    if level >= threshold:
                        loud = 1
                    elif loud == 1 and level >= threshold * self.START_FOLLOW_RATIO:
                        loud = self.START_CONFIRMATION_BLOCKS
                    else:
                        loud = 0

                        # Only quiet-enough blocks are allowed to influence the
                        # room baseline. This prevents soft speech from becoming
                        # the new noise floor and chasing the speaker upward.
                        if level < threshold * 0.60:
                            rate = (
                                self.NOISE_RISE_RATE
                                if level > noise
                                else self.NOISE_FALL_RATE
                            )
                            noise += rate * (level - noise)
                            self._update_thresholds(noise)

                    if loud >= self.START_CONFIRMATION_BLOCKS:
                        started = True
                        for pre_block, pre_level in pre_roll:
                            recorded.append(pre_block)
                            levels.append(pre_level)

                        recent.extend(lv for _, lv in list(pre_roll)[-3:])
                        voiced_blocks = 1
                        last_voice_at = time.monotonic()

                        logger.info(
                            f"Speech detected (level={level:.6f}, "
                            f"threshold={threshold:.6f})."
                        )
                        if verbose:
                            self._say("Speech detected...")
                        continue

                    if waited >= timeout_blocks:
                        logger.info(
                            f"No speech start detected: max level={observed_peak:.6f}, "
                            f"threshold={self.speech_threshold:.6f}, "
                            f"noise={self.noise_rms:.6f}."
                        )
                        if verbose:
                            self._say("Listening timeout.")
                        return None
                    continue

                recorded.append(block)
                levels.append(level)
                recent.append(level)
                recording_blocks += 1

                smoothed = float(np.median(recent)) if recent else level
                active = (
                    level >= self.end_speech_threshold
                    or smoothed >= self.end_speech_threshold
                )
                if active:
                    if pending is not None:
                        pending.cancel()
                        pending = None
                    silent_run = 0
                    voiced_blocks += 1
                    last_voice_at = time.monotonic()
                else:
                    silent_run += 1

                    if (
                        speculate_at
                        and pending is None
                        and silent_run == speculate_at
                        and voiced_blocks * block_s >= self.MIN_VOICED_DURATION
                    ):
                        snapshot = self._finalize_capture(
                            recorded,
                            levels,
                            block_s,
                            pre_roll_blocks,
                            tail_blocks,
                        )
                        if snapshot is not None:
                            snapshot.last_voice_at = last_voice_at
                            pending = self._executor.submit(self._process, snapshot)

                needed = end_short if voiced_blocks < short_voiced else end_long
                if silent_run >= needed:
                    break
                if recording_blocks >= max_phrase_blocks:
                    logger.info("Phrase time limit reached.")
                    break

            if self._status_events:
                logger.warning(
                    f"Audio stream reported {self._status_events} overflow/underflow event(s)."
                )
                self._status_events = 0

            capture = (
                self._finalize_capture(
                    recorded,
                    levels,
                    block_s,
                    pre_roll_blocks,
                    tail_blocks,
                )
                if recorded
                else None
            )

            if capture is None:
                if pending is not None:
                    pending.cancel()
                return None

            capture.last_voice_at = last_voice_at
            capture.pending = pending
            return capture

        except Exception as exc:
            logger.error(f"Audio capture failed: {exc}", exc_info=True)
            if pending is not None:
                pending.cancel()

            self._last_capture_failed = True
            self._stop_stream()
            if self._is_audio_device_error(exc):
                self.microphone_available = False
            self._say("Microphone capture failed.")
            return None

    def _finalize_capture(
        self,
        recorded: List[np.ndarray],
        levels: List[float],
        block_seconds: float,
        lead_blocks: int,
        tail_blocks: int,
    ) -> Optional[_Capture]:
        """Finalize the phrase while preserving the entire pre-roll and onset."""
        if not recorded or not levels:
            return None

        level_arr = np.asarray(levels, dtype=np.float32)
        smoothed = self._median3(level_arr)

        # Use a deliberately soft mask only for locating the spoken region and
        # estimating speech energy. The actual audio start is never trimmed: the
        # full pre-roll is retained so a quiet consonant cannot be cut away.
        voice_floor = max(
            self.MIN_END_RMS,
            self.end_speech_threshold * 0.75,
        )
        voice_mask = smoothed >= voice_floor
        if not voice_mask.any():
            voice_mask = smoothed >= self.MIN_END_RMS

        if not voice_mask.any():
            # Capture started from a confirmed speech event; preserve the clip
            # rather than throwing away a quiet but real utterance.
            voice_mask = np.ones_like(smoothed, dtype=bool)

        last = len(voice_mask) - 1 - int(np.argmax(voice_mask[::-1]))

        # Keep all recorded samples from the beginning of the pre-roll. Only the
        # trailing silence is trimmed. This is intentional: extra leading silence
        # is cheap for Whisper, while a lost first syllable is expensive.
        start = 0
        end = min(len(recorded), last + 1 + tail_blocks)

        kept = recorded[start:end]
        kept_levels = smoothed[start:end]
        kept_mask = voice_mask[start:end]
        if not kept:
            return None

        audio = np.concatenate(kept).astype(np.float32)

        # Estimate speech RMS from the strongest portion of the captured signal
        # rather than letting the quiet pre-roll define the gain.
        active_levels = kept_levels[kept_mask]
        if active_levels.size:
            cutoff = float(np.percentile(active_levels, 70))
            speech_selector = kept_levels >= cutoff
            speech_blocks = [
                block
                for block, selected in zip(kept, speech_selector)
                if selected
            ]
        else:
            speech_blocks = []

        speech_rms = rms(np.concatenate(speech_blocks)) if speech_blocks else rms(audio)

        return _Capture(
            audio=audio,
            voiced_seconds=float(kept_mask.sum()) * block_seconds,
            speech_rms=speech_rms,
        )

    @staticmethod
    def _median3(values: np.ndarray) -> np.ndarray:
        if values.size < 3:
            return values

        padded = np.pad(values, 1, mode="edge")
        return np.median(
            np.stack([padded[:-2], padded[1:-1], padded[2:]]),
            axis=0,
        )

    def _validate_audio(self, audio: np.ndarray, capture: _Capture) -> Optional[str]:
        if audio is None or audio.size == 0:
            return "no audio was captured"

        duration = audio.size / SAMPLE_RATE
        peak = float(np.max(np.abs(audio)))
        clipping_ratio = float(np.mean(np.abs(audio) >= 0.98))

        logger.info(
            f"Audio: {duration:.2f}s total, {capture.voiced_seconds:.2f}s voiced, "
            f"speech RMS {capture.speech_rms:.6f}, peak {peak:.4f}, "
            f"clipping {clipping_ratio:.4f}, noise floor {self.noise_rms:.6f}"
        )

        if duration < self.MIN_AUDIO_DURATION:
            return "speech was too short"
        if capture.voiced_seconds < self.MIN_VOICED_DURATION:
            return "that sounded like noise, not speech"
        if capture.speech_rms < self.MIN_SPEECH_RMS:
            return "the microphone captured almost no voice"
        if peak < self.MIN_PEAK:
            return "the captured audio level is extremely low"

        if clipping_ratio > self.MAX_CLIPPING_RATIO:
            logger.warning("Audio is heavily clipped; lower the microphone input gain.")

        return None

    def _process(self, capture: _Capture) -> _Processed:
        audio = resample_audio(capture.audio, self.capture_rate, SAMPLE_RATE)
        reason = self._validate_audio(audio, capture)
        if reason:
            return _Processed(audio, reason=reason)

        return _Processed(
            audio,
            decode=self._decode_audio(audio, speech_rms=capture.speech_rms),
        )

    def _decode_audio(
        self,
        audio: np.ndarray,
        speech_rms: Optional[float] = None,
    ) -> _Decode:
        prepared = preprocess_audio(
            audio,
            reference_rms=speech_rms,
        )
        return self.transcriber.transcribe(prepared)

    def _archive(self, audio: np.ndarray, tag: str) -> None:
        if not self.config.save_debug_audio or audio is None or audio.size == 0:
            return

        try:
            write_wav(self.DEBUG_AUDIO_FILE, audio)

            now = time.time()
            stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(now))
            stamp += f"{int((now % 1) * 1000):03d}"
            slug = re.sub(r"[^a-z0-9]+", "-", tag.lower()).strip("-")[:40] or "x"

            write_wav(self.DEBUG_DIR / f"{stamp}_{slug}.wav", audio)

            old_files = sorted(self.DEBUG_DIR.glob("*.wav"))
            for old in old_files[:-self.DEBUG_KEEP]:
                try:
                    old.unlink()
                except OSError:
                    pass

        except Exception as exc:
            logger.warning(f"Could not save debug audio: {exc}")

    @staticmethod
    def _clean_text(text: str) -> str:
        if not text:
            return ""

        text = re.sub(r"\s+", " ", text.strip())
        return text if re.search(r"\w", text) else ""

    @staticmethod
    def _is_audio_device_error(error: Exception) -> bool:
        message = str(error).lower()
        keywords = (
            "portaudio",
            "invalid device",
            "input device",
            "no default input",
            "device unavailable",
            "device not found",
            "invalid number of channels",
            "invalid sample rate",
            "paerror",
            "unanticipated host error",
            "host api",
        )
        return any(keyword in message for keyword in keywords)


@dataclass
class ClipReport:
    """Signal-level report produced by the raw microphone diagnostic."""

    floor_db: float
    peak_db: float
    speech_db: float
    snr_db: float
    onset_s: Optional[float]
    onset_step_db: float
    gated: bool
    notes: List[str]
    profile: str


def analyse_clip(audio: np.ndarray, rate: int = SAMPLE_RATE) -> ClipReport:
    """Analyse a raw silence/voice/silence clip without invoking Whisper."""
    audio = np.nan_to_num(np.asarray(audio, dtype=np.float32).reshape(-1))
    frame = max(1, int(rate * 0.02))
    count = audio.size // frame

    if count < 10:
        raise ValueError("clip is too short to analyse")

    frames = audio[: count * frame].reshape(count, frame)
    db = 20.0 * np.log10(np.sqrt(np.mean(np.square(frames), axis=1)) + 1e-9)

    quiet = np.sort(db)[: max(3, count * 3 // 10)]
    floor_db = float(np.median(quiet))
    peak_db = float(np.max(db))

    speech_cut = max(
        floor_db + 6.0,
        float(np.percentile(db, 85)),
    )
    loud = db >= speech_cut
    onset = int(np.argmax(loud)) if loud.any() else None
    speech_db = float(np.median(db[loud])) if loud.any() else peak_db

    step = 0.0
    if onset is not None:
        window = np.diff(db)[max(0, onset - 4) : onset + 4]
        step = float(window.max()) if window.size else 0.0

    gated = floor_db < -85.0 or (floor_db < -75.0 and step >= 40.0)
    snr = speech_db - floor_db

    notes: List[str] = []
    if peak_db < -50.0:
        notes.append("No speech captured: the clip is almost silent. Wrong input device or muted mic?")
    if floor_db < -85.0:
        notes.append(
            f"Digital silence between words ({floor_db:.0f} dBFS). A noise gate or "
            "audio enhancement may be muting the microphone and cutting word onsets."
        )
    elif gated:
        notes.append(
            f"Hard gate-like onset: {step:.0f} dB jump in 20 ms out of a very quiet floor."
        )

    if float(np.mean(np.abs(audio) >= 0.98)) > 0.001:
        notes.append("Clipping: lower the microphone input level.")
    elif -50.0 <= peak_db < -35.0:
        notes.append("Very quiet voice: raise the microphone input level.")

    if not gated and 0 < snr < 15.0:
        notes.append(
            f"Low signal-to-noise ratio ({snr:.0f} dB): background noise is close to your voice."
        )

    if not notes:
        notes.append("The input chain looks healthy.")

    chars = " .:-=+*#%@"
    group = 3
    profile = "".join(
        chars[
            int(
                np.clip(
                    (float(db[i : i + group].max()) + 100.0) / 90.0 * 9.0,
                    0,
                    9,
                )
            )
        ]
        for i in range(0, count, group)
    )

    return ClipReport(
        floor_db=floor_db,
        peak_db=peak_db,
        speech_db=speech_db,
        snr_db=snr,
        onset_s=None if onset is None else onset * frame / rate,
        onset_step_db=step,
        gated=gated,
        notes=notes,
        profile=profile,
    )


def _record_clip(
    seconds: float,
    requested: Optional[int],
    requested_channel: Optional[int],
) -> Tuple[np.ndarray, str, int]:
    """Record directly from the selected input device, without Whisper."""
    devices = sd.query_devices()
    default_input = _default_input_index()

    if isinstance(requested, Integral):
        index = int(requested)
        if index < 0:
            index = None
    else:
        index = None

    if index is None:
        index = default_input

    if index is None or index < 0:
        candidates = _input_device_candidates(devices, default_input)
        if not candidates:
            raise RuntimeError("No input microphone is available.")
        index = candidates[0]

    index = int(index)
    info = sd.query_devices(index)
    max_input_channels = int(info.get("max_input_channels", 0))
    if max_input_channels <= 0:
        raise RuntimeError(f"Device [{index}] has no input channels.")

    native_rate = int(round(float(info.get("default_samplerate", SAMPLE_RATE))))
    if requested_channel is not None:
        requested_channel = int(requested_channel)
        if requested_channel < 0 or requested_channel >= max_input_channels:
            raise RuntimeError(
                f"Requested microphone channel {requested_channel} is unavailable; "
                f"device exposes {max_input_channels} input channels."
            )
        channel_counts = [requested_channel + 1]
    else:
        channel_counts = [min(max_input_channels, 4), 2, 1]

    unique_counts = []
    for count in channel_counts:
        count = max(1, min(int(count), max_input_channels))
        if count not in unique_counts:
            unique_counts.append(count)

    rate = None
    needed_channels = 1
    for count in unique_counts:
        for candidate in (SAMPLE_RATE, native_rate):
            try:
                sd.check_input_settings(
                    device=index,
                    channels=count,
                    dtype="float32",
                    samplerate=candidate,
                )
                rate = candidate
                needed_channels = count
                break
            except Exception:
                continue
        if rate is not None:
            break

    if rate is None:
        raise RuntimeError("No supported input channel/rate combination was found.")

    if requested_channel is None and needed_channels >= 2:
        probe_channels = needed_channels
        try:
            probe_data = sd.rec(
                max(1, int(round(0.30 * rate))),
                samplerate=rate,
                channels=probe_channels,
                dtype="float32",
                device=index,
            )
            sd.wait()
            probe_data = np.nan_to_num(
                np.asarray(probe_data, dtype=np.float32),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
            levels = np.sqrt(np.mean(np.square(probe_data), axis=0))
            active = np.flatnonzero(levels >= STTEngine.DEAD_AIR_RMS * 2.0)
            if active.size:
                requested_channel = int(active[np.argmin(levels[active])])
            else:
                requested_channel = 0
        except Exception:
            requested_channel = 0

    if requested_channel is None:
        requested_channel = 0

    requested_channel = max(0, min(int(requested_channel), needed_channels - 1))
    stream_channels = requested_channel + 1

    # Re-check the exact stream shape we are about to open.
    sd.check_input_settings(
        device=index,
        channels=stream_channels,
        dtype="float32",
        samplerate=rate,
    )

    data = sd.rec(
        int(seconds * rate),
        samplerate=rate,
        channels=stream_channels,
        dtype="float32",
        device=index,
    )
    sd.wait()
    data = np.nan_to_num(
        np.asarray(data, dtype=np.float32),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    return (
        resample_audio(data[:, requested_channel], rate, SAMPLE_RATE),
        str(info.get("name", index)),
        requested_channel,
    )


def _run_mic_check(config: STTConfig, seconds: float = 4.0) -> int:
    print(
        "\nMic check: stay quiet for about 1 second, say 'hello' clearly, "
        "then stay quiet."
    )
    try:
        input("Press Enter to start... ")
    except EOFError:
        pass

    print("Recording...")
    try:
        audio, name, channel = _record_clip(
            seconds,
            config.mic_index,
            config.mic_channel,
        )
    except Exception as exc:
        print(f"Could not record: {exc}")
        return 1

    path = STTEngine.LOG_DIR / "stt_mic_check.wav"
    write_wav(path, audio)

    try:
        report = analyse_clip(audio)
    except ValueError as exc:
        print(f"Could not analyse the clip: {exc}")
        return 1

    onset = "n/a" if report.onset_s is None else f"{report.onset_s:.2f}s"
    print(f"\nMicrophone : {name}")
    print(f"Channel    : {channel}")
    print(
        f"Floor      : {report.floor_db:.0f} dBFS   Speech: {report.speech_db:.0f} dBFS   "
        f"Peak: {report.peak_db:.0f} dBFS   SNR: {report.snr_db:.0f} dB"
    )
    print(
        f"Onset      : {onset}  "
        f"(largest 20 ms step {report.onset_step_db:.0f} dB)"
    )
    print(f"Profile    : |{report.profile}|  (60 ms per character)")
    print(f"Verdict    : {'LIKELY GATED' if report.gated else 'ok'}")
    for note in report.notes:
        print(f"  - {note}")
    print(f"\nSaved: {path}")
    print("Playback tip: if 'hello' sounds like 'ello', the capture chain is cutting the onset.")
    return 0


SUITE: Tuple[Tuple[str, Optional[str]], ...] = (
    ("Say: hello", "hello"),
    ("Say: open chrome", "open chrome"),
    ("Say: open calculator", "open calculator"),
    ("Say: open youtube", "open youtube"),
    ("Say: open whatsapp", "open whatsapp"),
    ("Say: take a screenshot", "take a screenshot"),
    ("Say: search google for Python tutorials", "search google for python tutorials"),
    ("Say: search youtube for music", "search youtube for music"),
    ("Say: what is the capital of India", "what is the capital of india"),
    ("Say: stop", "stop"),
    ("Say ONE short word: yes", "yes"),
    ("Say FAST: open calculator", "open calculator"),
    ("Say with a pause: open ... chrome", "open chrome"),
    ("Say a normal sentence: what is the weather like today", "what is the weather like today"),
    ("Say: hello, then stay silent for a few seconds", "hello"),
    ("Stay completely silent", None),
    ("Background noise only (typing / music / TV) - do not speak", None),
)


def _print_result(result: Transcription) -> None:
    print(f"\n  status   : {result.status}")
    print(f"  text     : {result.text or '[empty]'}")
    if result.confidence is not None:
        print(
            f"  quality  : confidence {result.confidence:.2f} "
            f"(logprob {result.avg_logprob:.2f}, no-speech {result.no_speech_prob:.2f})"
        )
    print(
        f"  timing   : {result.audio_seconds:.2f}s audio | "
        f"decode {result.decode_ms:.0f} ms | "
        f"response {result.response_ms:.0f} ms after you stopped"
    )


def _run_live(engine: STTEngine, loop: bool) -> int:
    print("\nSpeak a short command (Ctrl+C to stop).")
    try:
        while True:
            _print_result(
                engine.listen_detailed(timeout=10, phrase_time_limit=15)
            )
            if not loop:
                break
    except KeyboardInterrupt:
        print("\nStopped.")
    return 0


def _run_suite(engine: STTEngine) -> int:
    rows = []

    for instruction, expected in SUITE:
        try:
            input(f"\n{instruction}\n  Press Enter, then do it... ")
        except EOFError:
            break

        result = engine.listen_detailed(
            timeout=6 if expected is None else 8,
            phrase_time_limit=15,
        )
        heard = normalize_text(result.text)
        passed = (heard == normalize_text(expected)) if expected else not heard

        rows.append((instruction, expected, result, passed))
        print(
            f"  -> {'PASS' if passed else 'FAIL'}  heard {result.text or '[nothing]'!r}"
            f"  ({result.response_ms:.0f} ms)"
        )

    if not rows:
        return 1

    print(f"\n{sum(1 for *_, passed in rows if passed)}/{len(rows)} passed")

    spoken = [
        result.response_ms
        for _, expected, result, passed in rows
        if expected and passed and result.response_ms
    ]
    if spoken:
        print(
            f"Response after you stop talking: median {float(np.median(spoken)):.0f} ms, "
            f"worst {max(spoken):.0f} ms"
        )
    return 0


def _run_bench(paths: List[str], config: STTConfig, runs: int) -> int:
    clips: List[Path] = []
    for item in paths:
        path = Path(item)
        clips += sorted(path.glob("*.wav")) if path.is_dir() else [path]

    if not clips:
        print("No WAV clips found.")
        return 1

    transcriber = WhisperTranscriber(config, config.initial_prompt)
    if not transcriber.load():
        print("Whisper model could not be loaded.")
        return 1

    print(
        f"Model {config.model} on {transcriber.device.upper()} "
        f"({transcriber.compute_type}), beam {transcriber.beam_size}, "
        f"prompt {'on' if transcriber.prompt else 'off'}\n"
    )

    timings: List[float] = []
    for clip in clips:
        try:
            audio = preprocess_audio(read_wav(clip))
        except Exception as exc:
            print(f"{clip.name}: unreadable ({exc})")
            continue

        runs_ms: List[float] = []
        decode = _Decode()
        for _ in range(max(1, runs)):
            decode = transcriber.transcribe(audio)
            runs_ms.append(decode.decode_ms)

        ms = float(np.median(runs_ms))
        timings.append(ms)
        confidence = (
            "n/a"
            if decode.avg_logprob is None
            else f"{math.exp(decode.avg_logprob):.2f}"
        )
        print(
            f"{clip.name[:42]:42} {audio.size / SAMPLE_RATE:5.1f}s "
            f"{ms:6.0f} ms  conf {confidence:>5}  "
            f"{decode.text or '[empty]'}"
        )

    transcriber.close()

    if timings:
        print(
            f"\n{len(timings)} clips | decode median {float(np.median(timings)):.0f} ms, "
            f"worst {max(timings):.0f} ms"
        )
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m audio.stt",
        description="EDITH STT diagnostics (run from the project root).",
    )
    parser.add_argument("--mics", action="store_true", help="list input devices and exit")
    parser.add_argument(
        "--mic-check",
        action="store_true",
        help="record one word and report raw microphone quality",
    )
    parser.add_argument("--loop", action="store_true", help="keep listening until Ctrl+C")
    parser.add_argument("--suite", action="store_true", help="guided accuracy / latency test")
    parser.add_argument(
        "--bench",
        nargs="*",
        metavar="WAV_OR_DIR",
        help="time and re-transcribe saved clips (default: logs/stt_debug)",
    )
    parser.add_argument("--runs", type=int, default=3, help="decodes per clip for --bench")
    parser.add_argument("--model", help="override STT_MODEL")
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        help="override STT_DEVICE",
    )
    parser.add_argument("--compute-type", help="override STT_COMPUTE_TYPE")
    parser.add_argument("--cpu-threads", type=int, help="override STT_CPU_THREADS")
    parser.add_argument("--beam", type=int, help="override STT_BEAM_SIZE")
    parser.add_argument("--mic", type=int, help="override STT_MIC_INDEX")
    parser.add_argument("--mic-channel", type=int, help="override STT_MIC_CHANNEL (0-based)")
    parser.add_argument("--prompt", help="override STT_INITIAL_PROMPT")
    args = parser.parse_args(argv)

    if args.mics:
        STTEngine.list_microphones()
        return 0

    overrides = {
        key: value
        for key, value in {
            "model": args.model,
            "device": args.device,
            "compute_type": args.compute_type,
            "cpu_threads": args.cpu_threads,
            "beam_size": args.beam,
            "mic_index": args.mic,
            "mic_channel": args.mic_channel,
            "initial_prompt": args.prompt,
        }.items()
        if value is not None
    }
    config = dataclasses.replace(STTConfig.from_settings(), **overrides)

    if args.mic_check:
        return _run_mic_check(config)

    if args.bench is not None:
        return _run_bench(
            args.bench or [str(STTEngine.DEBUG_DIR)],
            config,
            args.runs,
        )

    engine = STTEngine(config=config)

    try:
        if not engine.microphone_available:
            print("No usable microphone is available.")
            return 1
        if engine.whisper_model is None:
            print("Whisper could not be initialized.")
            return 1

        return _run_suite(engine) if args.suite else _run_live(engine, args.loop)
    finally:
        engine.close()
        print(f"\nLast capture : {STTEngine.DEBUG_AUDIO_FILE}")
        print(f"Recent clips : {STTEngine.DEBUG_DIR}")


if __name__ == "__main__":
    raise SystemExit(main())
