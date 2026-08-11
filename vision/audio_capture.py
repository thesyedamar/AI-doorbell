"""
vision/audio_capture.py — Microphone Recording
===============================================
Records audio from the default microphone and saves as WAV.
Uses numpy for normalization (replaces deprecated audioop).

Settings from config:
    sample_rate:    16000 Hz  (Whisper expects this)
    channels:       1         (mono)
    record_seconds: 6         (per dialogue turn)
"""

import wave
import time
import logging
import numpy as np
from pathlib import Path
from datetime import datetime

import sounddevice as sd

from config import cfg

logger = logging.getLogger(__name__)


def record_audio(filename_prefix: str = "visitor") -> str:
    """
    Record audio from the microphone and save as a WAV file.

    Args:
        filename_prefix: Prefix for the output filename.
                         Final name: {prefix}_{timestamp}.wav

    Returns:
        Full path to the saved WAV file as a string.
        Returns "" if recording failed.
    """
    sample_rate = cfg.audio.sample_rate
    channels    = cfg.audio.channels
    duration    = cfg.audio.record_seconds

    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename    = f"{filename_prefix}_{timestamp}.wav"
    save_path   = cfg.paths.audio_dir / filename

    logger.info(f"Recording {duration}s of audio → {filename}")

    try:
        # Record as float32
        recording = sd.rec(
            frames=int(duration * sample_rate),
            samplerate=sample_rate,
            channels=channels,
            dtype="float32",
        )
        sd.wait()  # Block until recording is complete

        # Convert float32 → int16 with peak normalization
        audio_int16 = _normalize_to_int16(recording)

        # Save as WAV
        _save_wav(save_path, audio_int16, sample_rate, channels)

        logger.info(f"Audio saved: {save_path}")
        return str(save_path)

    except Exception as e:
        logger.error(f"Audio recording failed: {e}")
        return ""


def _normalize_to_int16(audio_float32: np.ndarray) -> np.ndarray:
    """
    Convert float32 audio [-1.0, 1.0] to int16 [-32768, 32767].
    Applies peak normalization to maximize volume without clipping.

    Replaces audioop (removed in Python 3.13) with numpy.
    """
    # Flatten to 1D if stereo accidentally captured
    audio = audio_float32.flatten()

    # Peak normalization — scale so loudest sample hits 90% of int16 max
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak * 0.9

    # Convert to int16
    audio_int16 = (audio * 32767).astype(np.int16)
    return audio_int16


def _save_wav(
    path: Path,
    audio_int16: np.ndarray,
    sample_rate: int,
    channels: int,
) -> None:
    """Write int16 numpy array to a WAV file."""
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)          # 2 bytes = int16
        wf.setframerate(sample_rate)
        wf.writeframes(audio_int16.tobytes())


def list_audio_devices() -> None:
    """
    Print all available audio input devices.
    Useful for debugging microphone issues.
    Run: python -c "from vision.audio_capture import list_audio_devices; list_audio_devices()"
    """
    print("\nAvailable audio input devices:")
    print("-" * 40)
    devices = sd.query_devices()
    for i, device in enumerate(devices):
        if device["max_input_channels"] > 0:
            print(f"  [{i}] {device['name']}")
    print(f"\nDefault input: {sd.query_devices(kind='input')['name']}")
