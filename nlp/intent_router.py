"""
nlp/intent_router.py — Intent Routing
=======================================
Routes intent detection between ML model and Ollama fallback.

Decision flow:
    1. Try ML model first (fast, local, no API call)
    2. If ML confidence >= threshold → use ML result
    3. If ML confidence < threshold  → try Ollama LLM
    4. If Ollama fails or unavailable → return "unknown"

This two-stage approach gives us:
    - Speed of ML for clear cases
    - Accuracy of LLM for ambiguous cases
    - Graceful degradation if either fails
"""

import logging
from typing import Optional

from config import cfg
from nlp.intent_ml import detect_intent as ml_detect

logger = logging.getLogger(__name__)

# Valid intent labels — anything outside this is forced to "unknown"
VALID_INTENTS = {"greeting", "delivery", "guest", "emergency", "unknown"}


def route_intent(text: str) -> dict:
    """
    Route intent detection through ML then Ollama fallback.

    Args:
        text: Urdu transcript from Whisper

    Returns:
        dict with keys:
            intent:     final intent label
            confidence: confidence score [0.0 - 1.0]
            source:     "ml" | "ollama" | "fallback"
            name:       name extracted by Ollama (if any)
    """
    if not text or not text.strip():
        return _fallback_result("empty transcript")

    text = text.strip()
    logger.info(f"Routing intent for: '{text[:60]}'")

    # ----------------------------------------------------------------
    # Stage 1 — ML classifier
    # ----------------------------------------------------------------
    ml_intent, ml_conf = ml_detect(text)

    if ml_intent != "unknown" and ml_conf >= cfg.nlp.intent_confidence_threshold:
        logger.info(
            f"ML intent accepted | intent={ml_intent} "
            f"confidence={ml_conf:.3f}"
        )
        return {
            "intent":     ml_intent,
            "confidence": ml_conf,
            "source":     "ml",
            "name":       None,
        }

    logger.info(
        f"ML confidence low ({ml_conf:.3f}) — "
        f"routing to Ollama fallback."
    )

    # ----------------------------------------------------------------
    # Stage 2 — Ollama LLM fallback
    # ----------------------------------------------------------------
    if cfg.features.ollama_fallback:
        ollama_result = _try_ollama(text)
        if ollama_result:
            return ollama_result

    # ----------------------------------------------------------------
    # Stage 3 — Final fallback
    # ----------------------------------------------------------------
    return _fallback_result("all sources failed")


def _try_ollama(text: str) -> dict | None:
    """
    Call Ollama for intent detection.
    Returns result dict or None if Ollama is unavailable.
    """
    try:
        from llm.ollama_client import run_ollama_intent

        response = run_ollama_intent(text)
        if response is None:
            logger.warning("Ollama returned None.")
            return None

        intent = response.intent
        if intent not in VALID_INTENTS:
            logger.warning(f"Ollama returned invalid intent: '{intent}' — forcing unknown.")
            intent = "unknown"

        logger.info(
            f"Ollama intent | intent={intent} "
            f"confidence={response.confidence:.3f}"
        )

        return {
            "intent":     intent,
            "confidence": float(response.confidence),
            "source":     "ollama",
            "name":       response.name,
        }

    except Exception as e:
        logger.error(f"Ollama intent routing failed: {e}")
        return None


def _fallback_result(reason: str) -> dict:
    """Return a safe unknown fallback result."""
    logger.info(f"Intent fallback | reason={reason}")
    return {
        "intent":     "unknown",
        "confidence": 0.0,
        "source":     "fallback",
        "name":       None,
    }
