"""
nlp/response_mapper.py — Response Audio Mapper
===============================================
Maps dialogue response keys to their audio file paths.
Supports multiple languages via config.

Adding a new response:
    1. Add key to RESPONSE_MAP below
    2. Place the WAV file in data/audio/responses/
    Done — no other file needs to change.

Adding a new language (e.g. Pashto):
    1. Set FEATURE_PASHTO=1 in .env
    2. Add _ps.wav files to data/audio/responses/
    Done — mapper auto-selects based on language.
"""

import logging
from pathlib import Path
from typing import Optional

from config import cfg, Language

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Response registry
# ---------------------------------------------------------------------------
# Key → base filename (without language suffix and .wav extension)
# Language suffix is appended automatically: _ur.wav, _ps.wav, _en.wav
#
# To add a new response:
#   "MY_NEW_KEY": "my_new_file"
#   → looks for data/audio/responses/my_new_file_ur.wav
#
RESPONSE_MAP: dict = {
    # Name collection
    "ASK_NAME":               "ask_name",
    "ASK_NAME_RETRY":         "ask_name_retry",

    # Intent collection
    "ASK_INTENT":             "ask_intent",
    "ASK_INTENT_RETRY":       "ask_intent_retry",

    # Greetings
    "GREETING_KNOWN":         "greeting_known",
    "GREETING_UNKNOWN":       "greeting_unknown",

    # Intent responses
    "DELIVERY_INSTRUCTION":   "parcel_instruction",
    "EMERGENCY_ALERT":        "emergency",

    # Fallbacks
    "FALLBACK_LOW_CONFIDENCE":"fallback_low_confidence",
    "FALLBACK_NO_NAME":       "fallback_no_name",
}


def get_response(
    response_key: str,
    language: Optional[Language] = None,
) -> Optional[dict]:
    """
    Get the audio file path for a response key.

    Args:
        response_key: One of the keys in RESPONSE_MAP (e.g. "ASK_NAME")
        language:     Language enum. Defaults to config default language.

    Returns:
        dict with keys: key, path, exists
        Returns None if key is not in RESPONSE_MAP.

    Example:
        result = get_response("ASK_NAME")
        # {"key": "ASK_NAME", "path": "...ask_name_ur.wav", "exists": True}
    """
    if response_key not in RESPONSE_MAP:
        logger.warning(f"Unknown response key: '{response_key}'")
        return None

    # Resolve language suffix
    lang = language or cfg.language.default_language
    suffix = cfg.language.audio_suffixes.get(lang, "_ur.wav")

    base_name = RESPONSE_MAP[response_key]
    filename  = f"{base_name}{suffix}"
    full_path = cfg.paths.responses_dir / filename

    result = {
        "key":    response_key,
        "path":   str(full_path),
        "exists": full_path.exists(),
    }

    if not result["exists"]:
        logger.warning(
            f"Audio file missing for key '{response_key}': {full_path}\n"
            f"  Place the file at: {full_path}"
        )

    return result


def get_response_path(
    response_key: str,
    language: Optional[Language] = None,
) -> str:
    """
    Convenience function — returns just the path string.
    Returns "" if key not found or file missing.
    """
    result = get_response(response_key, language)
    if result and result["exists"]:
        return result["path"]
    return ""


def list_responses() -> None:
    """
    Print the status of all response audio files.
    Run to check which WAV files are missing.
    """
    lang   = cfg.language.default_language
    suffix = cfg.language.audio_suffixes.get(lang, "_ur.wav")

    print(f"\nResponse audio files ({lang.value}):")
    print("-" * 55)

    all_present = True
    for key, base in RESPONSE_MAP.items():
        filename  = f"{base}{suffix}"
        full_path = cfg.paths.responses_dir / filename
        status    = "✅" if full_path.exists() else "❌ MISSING"
        print(f"  {status}  {key:<30} {filename}")
        if not full_path.exists():
            all_present = False

    print()
    if all_present:
        print("  All response files present!")
    else:
        print(f"  Place missing files in: {cfg.paths.responses_dir}")
    print()
