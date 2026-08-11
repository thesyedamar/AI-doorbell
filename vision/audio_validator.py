"""
vision/audio_validator.py — Audio Quality Validator
====================================================
Checks recorded audio before sending to Whisper.
Rejects silent recordings, too-short clips, and
files where the visitor didn't actually speak.

Uses numpy for RMS calculation — no audioop needed
(audioop was removed in Python 3.13).

Checks:
    1. File exists and is readable
    2. Duration >= 1.0 seconds
    3. RMS volume >= 200 (not silent)
"""

import wave
import logging
import numpy as np
from pathlib import Path

from config import cfg

logger = logging.getLogger(__name__)


def validate_audio(audio_path: str) -> bool:
    """
    Validate a recorded WAV file before transcription.

    Args:
        audio_path: Full path to the WAV file.

    Returns:
        True  — audio is valid, safe to transcribe.
        False — audio is silent, too short, or unreadable.
    """
    if not audio_path:
        logger.warning("validate_audio: empty path provided.")
        return False

    path = Path(audio_path)
    if not path.exists():
        logger.warning(f"validate_audio: file not found: {path}")
        return False

    try:
        with wave.open(str(path), "rb") as wf:
            channels    = wf.getnchannels()
            sample_width= wf.getsampwidth()
            frame_rate  = wf.getframerate()
            n_frames    = wf.getnframes()
            raw_bytes   = wf.readframes(n_frames)

        # Check 1 — Duration
        duration = n_frames / frame_rate
        if duration < cfg.audio.min_audio_duration:
            logger.warning(
                f"Audio too short: {duration:.2f}s "
                f"(minimum {cfg.audio.min_audio_duration}s)"
            )
            return False

        # Check 2 — RMS volume using numpy (no audioop)
        audio_array = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32)
        if len(audio_array) == 0:
            logger.warning("Audio file is empty.")
            return False

        rms = float(np.sqrt(np.mean(audio_array ** 2)))

        if rms < cfg.audio.min_rms_volume:
            logger.warning(
                f"Audio too quiet: RMS={rms:.1f} "
                f"(minimum {cfg.audio.min_rms_volume})"
            )
            return False

        logger.info(
            f"Audio valid | duration={duration:.2f}s "
            f"rms={rms:.1f} path={path.name}"
        )
        return True

    except Exception as e:
        logger.error(f"Audio validation error: {e}")
        return False


def get_audio_stats(audio_path: str) -> dict:
    """
    Return detailed stats about a WAV file.
    Useful for debugging microphone issues.

    Returns dict with: duration, rms, peak, sample_rate, channels
    """
    path = Path(audio_path)
    if not path.exists():
        return {}

    try:
        with wave.open(str(path), "rb") as wf:
            frame_rate = wf.getframerate()
            n_frames   = wf.getnframes()
            channels   = wf.getnchannels()
            raw_bytes  = wf.readframes(n_frames)

        audio = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32)
        duration = n_frames / frame_rate

        return {
            "duration":    round(duration, 2),
            "rms":         round(float(np.sqrt(np.mean(audio ** 2))), 1),
            "peak":        round(float(np.max(np.abs(audio))), 1),
            "sample_rate": frame_rate,
            "channels":    channels,
            "valid":       validate_audio(audio_path),
        }
    except Exception as e:
        logger.error(f"get_audio_stats error: {e}")
        return {}
