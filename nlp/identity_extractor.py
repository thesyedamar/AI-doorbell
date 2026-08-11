"""
nlp/identity_extractor.py — Urdu Name Extractor
=================================================
Extracts visitor names from Urdu speech transcripts.

Two-stage approach:
    Stage 1: Rule-based extraction (fast, no API call)
             Looks for name introduction patterns like:
             "میرا نام احمد ہے" → "احمد"
             "میں علی ہوں"     → "علی"

    Stage 2: Ollama fallback (if rules fail)
             Sends transcript to LLM with targeted prompt
             to extract the name from free-form speech.

This replaces the original pure rule-based approach which
was too brittle for real Urdu speech variations.
"""

import re
import logging
from typing import Optional

from config import cfg

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Urdu name introduction patterns
# ---------------------------------------------------------------------------
# Each pattern tries to capture the name after a common introduction phrase.
# The (?P<name>...) group captures the actual name.

URDU_NAME_PATTERNS = [
    # "میرا نام احمد ہے"
    r"میرا\s+نام\s+(?P<name>[\u0600-\u06FF]+(?:\s+[\u0600-\u06FF]+){0,2})\s*ہے",
    # "میں احمد ہوں"
    r"میں\s+(?P<name>[\u0600-\u06FF]+(?:\s+[\u0600-\u06FF]+){0,2})\s*ہوں",
    # "مجھے احمد کہتے ہیں"
    r"مجھے\s+(?P<name>[\u0600-\u06FF]+(?:\s+[\u0600-\u06FF]+){0,2})\s*کہتے",
    # "نام احمد ہے"
    r"نام\s+(?P<name>[\u0600-\u06FF]+(?:\s+[\u0600-\u06FF]+){0,2})\s*ہے",
    # "میں احمد علی ہوں"
    r"میں\s+(?P<name>[\u0600-\u06FF]+\s+[\u0600-\u06FF]+)\s*ہوں",
    # Standalone Urdu word(s) — last resort rule
    r"^(?P<name>[\u0600-\u06FF]+(?:\s+[\u0600-\u06FF]+){0,2})$",
]


def _is_valid_name(name: str) -> tuple:
    """
    Validate extracted name against rules.

    Returns:
        (is_valid: bool, reason: str)
    """
    if not name:
        return False, "empty"

    name = name.strip()
    words = name.split()

    # Word count check
    if len(words) > cfg.nlp.name_max_words:
        return False, f"too many words ({len(words)})"

    # Total length check
    if len(name) > cfg.nlp.name_max_chars:
        return False, f"too long ({len(name)} chars)"

    # Individual word length check
    for word in words:
        if len(word) < cfg.nlp.name_min_char_length:
            return False, f"word too short: '{word}'"

    # Stopword check — reject common Urdu words that are not names
    for word in words:
        if word in cfg.nlp.urdu_stopwords:
            return False, f"stopword detected: '{word}'"

    # Must contain only Urdu Unicode characters and spaces
    urdu_only = re.sub(r"[\u0600-\u06FF\s]", "", name)
    if urdu_only:
        return False, f"non-Urdu characters: '{urdu_only}'"

    return True, "valid"


def _rule_based_extract(text: str) -> Optional[str]:
    """
    Try all pattern rules to extract a name.
    Returns the name string or None.
    """
    text = text.strip()

    for pattern in URDU_NAME_PATTERNS:
        match = re.search(pattern, text)
        if match:
            name = match.group("name").strip()
            is_valid, reason = _is_valid_name(name)
            if is_valid:
                logger.debug(f"Rule-based match: '{name}' via pattern")
                return name
            else:
                logger.debug(f"Rule match rejected: '{name}' — {reason}")

    return None


def _ollama_extract(text: str) -> Optional[str]:
    """
    Use Ollama LLM to extract name when rules fail.
    More robust for free-form speech and unusual introductions.
    """
    if not cfg.features.ollama_fallback:
        return None

    try:
        from llm.ollama_client import run_ollama_name_extract
        name = run_ollama_name_extract(text)
        if name:
            is_valid, reason = _is_valid_name(name)
            if is_valid:
                logger.info(f"Ollama name extraction: '{name}'")
                return name
            else:
                logger.debug(f"Ollama name rejected: '{name}' — {reason}")
    except Exception as e:
        logger.warning(f"Ollama name extraction failed: {e}")

    return None


def extract_identity(text: str) -> dict:
    """
    Extract visitor name from Urdu transcript.

    Args:
        text: Raw Urdu transcript from Whisper

    Returns:
        dict with keys:
            name:   extracted name or ""
            valid:  True if a valid name was found
            source: "rule" | "ollama" | "none"
            reason: explanation if invalid
    """
    if not text or not text.strip():
        return {"name": "", "valid": False, "source": "none", "reason": "empty transcript"}

    text = text.strip()
    logger.info(f"Extracting identity from: '{text}'")

    # Stage 1 — Rule-based
    name = _rule_based_extract(text)
    if name:
        return {"name": name, "valid": True, "source": "rule", "reason": "valid"}

    # Stage 2 — Ollama fallback
    logger.info("Rules failed — trying Ollama for name extraction.")
    name = _ollama_extract(text)
    if name:
        return {"name": name, "valid": True, "source": "ollama", "reason": "valid"}

    return {
        "name":   "",
        "valid":  False,
        "source": "none",
        "reason": "no name found in transcript",
    }
