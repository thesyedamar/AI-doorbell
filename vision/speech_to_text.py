"""
vision/speech_to_text.py — Whisper Speech-to-Text
==================================================
Transcribes recorded audio to Urdu text using OpenAI Whisper.
Runs fully on CPU — no GPU required.

Model is loaded once and reused across all calls (lazy loading).
First call will download the model weights (~460MB for 'small').
Subsequent calls use the cached model instantly.
"""

import os
import logging
from typing import Optional

from config import cfg

logger = logging.getLogger(__name__)

# Force CPU mode — prevents Whisper from looking for CUDA
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Module-level model cache — loaded once, reused forever
_whisper_model = None


def _get_model():
    """
    Lazy-load Whisper model on first use.
    Subsequent calls return the cached model instantly.
    """
    global _whisper_model

    if _whisper_model is not None:
        return _whisper_model

    model_name = cfg.audio.whisper_model
    logger.info(
        f"Loading Whisper model: '{model_name}' "
        f"(first run downloads ~460MB — please wait...)"
    )

    try:
        import whisper
        _whisper_model = whisper.load_model(model_name, device="cpu")
        logger.info(f"Whisper '{model_name}' loaded successfully.")
        return _whisper_model

    except Exception as e:
        logger.error(f"Failed to load Whisper model: {e}")
        return None


def transcribe_audio(audio_path: str) -> str:
    """
    Transcribe a WAV file to Urdu text.

    Args:
        audio_path: Full path to the WAV file to transcribe.

    Returns:
        Transcribed text string.
        Returns "" if transcription fails or file is missing.
    """
    if not audio_path:
        logger.warning("transcribe_audio called with empty path.")
        return ""

    from pathlib import Path
    if not Path(audio_path).exists():
        logger.warning(f"Audio file not found: {audio_path}")
        return ""

    model = _get_model()
    if model is None:
        logger.error("Whisper model unavailable — cannot transcribe.")
        return ""

    try:
        logger.info(f"Transcribing: {audio_path}")

        result = model.transcribe(
            audio_path,
            language=cfg.audio.whisper_language,   # "ur" = Urdu
            fp16=cfg.audio.whisper_fp16,            # False on CPU
            verbose=False,
        )

        text = result.get("text", "").strip()
        logger.info(f"Transcript: '{text}'")
        return text

    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        return ""


def transcribe_with_fallback(audio_path: str) -> str:
    """
    Transcribe with automatic language fallback.
    Tries Urdu first. If result is empty, tries without
    language constraint (auto-detect).

    Useful when visitor accidentally speaks in a mix of languages.
    """
    text = transcribe_audio(audio_path)

    if not text:
        logger.info("Urdu transcription empty — trying auto-detect.")
        model = _get_model()
        if model:
            try:
                result = model.transcribe(
                    audio_path,
                    fp16=cfg.audio.whisper_fp16,
                    verbose=False,
                )
                text = result.get("text", "").strip()
                logger.info(f"Auto-detect transcript: '{text}'")
            except Exception as e:
                logger.error(f"Auto-detect transcription failed: {e}")

    return text
