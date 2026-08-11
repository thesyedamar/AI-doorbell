"""
vision/speaker.py — Audio Playback
====================================
Plays pre-recorded Urdu WAV response files.
Uses pygame.mixer — reliable on Windows and Linux/RPi.
Replaces the abandoned playsound library.

Pygame mixer is initialized once and reused for all playback.
"""

import time
import logging
from pathlib import Path
from typing import Optional

from config import cfg

logger = logging.getLogger(__name__)

# Track init state — only initialize pygame mixer once
_mixer_ready = False


def _init_mixer() -> bool:
    """
    Initialize pygame mixer on first use.
    Returns True if ready, False if initialization failed.
    """
    global _mixer_ready

    if _mixer_ready:
        return True

    try:
        import pygame
        pygame.mixer.init(
            frequency=cfg.audio.sample_rate,  # 16000 Hz
            size=-16,                          # signed 16-bit
            channels=1,                        # mono
            buffer=512,
        )
        _mixer_ready = True
        logger.info("pygame mixer initialized.")
        return True

    except Exception as e:
        logger.error(f"Failed to initialize pygame mixer: {e}")
        return False


def play_audio(audio_path: str, block: bool = True) -> bool:
    """
    Play a WAV file through the speaker.

    Args:
        audio_path: Full path to the WAV file.
        block:      If True, wait for playback to finish before returning.
                    If False, return immediately (non-blocking).

    Returns:
        True if playback started successfully, False otherwise.
    """
    if not audio_path:
        logger.warning("play_audio: empty path provided.")
        return False

    # Dry run mode — skip actual playback during testing
    if cfg.features.dry_run:
        logger.info(f"[DRY RUN] Would play: {audio_path}")
        return True

    path = Path(audio_path)
    if not path.exists():
        logger.warning(f"play_audio: file not found: {path}")
        return False

    if not _init_mixer():
        return False

    try:
        import pygame

        pygame.mixer.music.load(str(path))
        pygame.mixer.music.play()
        logger.info(f"Playing: {path.name}")

        if block:
            # Wait for playback to finish
            while pygame.mixer.music.get_busy():
                time.sleep(0.05)

        return True

    except Exception as e:
        logger.error(f"Playback failed for {path.name}: {e}")
        return False


def play_response(response_key: str, block: bool = True) -> bool:
    """
    Play a response by key name instead of full path.
    Looks up the file in data/audio/responses/.

    Args:
        response_key: e.g. "ask_name_ur" or "greeting_known_ur"
                      (.wav extension added automatically)

    Returns:
        True if played successfully, False otherwise.
    """
    filename = response_key if response_key.endswith(".wav") else f"{response_key}.wav"
    full_path = cfg.paths.responses_dir / filename
    return play_audio(str(full_path), block=block)


def stop_audio() -> None:
    """Stop any currently playing audio immediately."""
    global _mixer_ready
    if _mixer_ready:
        try:
            import pygame
            pygame.mixer.music.stop()
            logger.debug("Audio stopped.")
        except Exception as e:
            logger.warning(f"Could not stop audio: {e}")


def is_playing() -> bool:
    """Return True if audio is currently playing."""
    if not _mixer_ready:
        return False
    try:
        import pygame
        return pygame.mixer.music.get_busy()
    except Exception:
        return False
