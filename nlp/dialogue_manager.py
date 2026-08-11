"""
nlp/dialogue_manager.py — Dialogue State Machine
==================================================
Controls the full conversation flow with the visitor.

States:
    START       → decide first action based on known/unknown
    ASK_NAME    → collect visitor name (unknown visitors)
    ASK_INTENT  → collect visit reason (all visitors)
    RESPOND     → play final response based on intent
    DONE        → dialogue complete

Flow:
    Known visitor:
        START → ASK_INTENT → RESPOND → DONE

    Unknown visitor:
        START → ASK_NAME → ASK_INTENT → RESPOND → DONE

    On retry exhaustion → FALLBACK → DONE
"""

import logging
from typing import Optional

from config import cfg

logger = logging.getLogger(__name__)

# Valid dialogue states
STATES = {
    "START",
    "ASK_NAME",
    "ASK_INTENT",
    "RESPOND",
    "FALLBACK",
    "DONE",
}

# Maps intent label to response audio key
INTENT_TO_RESPONSE = {
    "delivery":  "DELIVERY_INSTRUCTION",
    "emergency": "EMERGENCY_ALERT",
    "guest":     "GREETING_UNKNOWN",
    "greeting":  "GREETING_KNOWN",
    "unknown":   "FALLBACK_LOW_CONFIDENCE",
}


def handle_dialogue(
    dialogue_state: dict,
    face_status: str,
    face_name: str,
    intent: str = "",
    intent_confidence: float = 0.0,
    entities: dict = None,
    retry_count: int = 0,
) -> dict:
    """
    Process one step of the dialogue state machine.

    Args:
        dialogue_state:    Current state dict {"stage": str, "retry_count": int}
        face_status:       "Known" or "Unknown"
        face_name:         Name from FAISS or "Unknown"
        intent:            Detected intent label (empty if not yet collected)
        intent_confidence: Confidence score for intent
        entities:          Dict with extracted entities e.g. {"name": "احمد"}
        retry_count:       How many retries have happened in current stage

    Returns:
        dict with keys:
            response_key:  Audio key to play (from response_mapper)
            next_state:    Next dialogue stage
            retry_count:   Updated retry count
            done:          True if dialogue is complete
            log_event:     True if this step should be saved to DB
    """
    entities    = entities or {}
    stage       = dialogue_state.get("stage", "START")
    max_retries = cfg.nlp.max_dialogue_retries
    threshold   = cfg.nlp.intent_confidence_threshold

    logger.info(
        f"Dialogue | stage={stage} face={face_status} "
        f"name={face_name} intent={intent} conf={intent_confidence:.2f} "
        f"retry={retry_count}"
    )

    # ----------------------------------------------------------------
    # START
    # ----------------------------------------------------------------
    if stage == "START":
        if face_status == "Known":
            return _result(
                response_key="GREETING_KNOWN",
                next_state="ASK_INTENT",
                retry_count=0,
            )
        else:
            return _result(
                response_key="GREETING_UNKNOWN",
                next_state="ASK_NAME",
                retry_count=0,
            )

    # ----------------------------------------------------------------
    # ASK_NAME
    # ----------------------------------------------------------------
    if stage == "ASK_NAME":
        spoken_name = entities.get("name", "")

        # Name successfully extracted
        if spoken_name:
            logger.info(f"Name captured: '{spoken_name}'")
            return _result(
                response_key="ASK_INTENT",
                next_state="ASK_INTENT",
                retry_count=0,
            )

        # Retry available
        if retry_count < max_retries:
            logger.info(f"Name not found — retry {retry_count + 1}/{max_retries}")
            return _result(
                response_key="ASK_NAME_RETRY",
                next_state="ASK_NAME",
                retry_count=retry_count + 1,
            )

        # Retries exhausted
        logger.warning("Name collection failed after max retries.")
        return _result(
            response_key="FALLBACK_NO_NAME",
            next_state="DONE",
            retry_count=retry_count,
            done=True,
            log_event=True,
        )

    # ----------------------------------------------------------------
    # ASK_INTENT
    # ----------------------------------------------------------------
    if stage == "ASK_INTENT":

        # No intent collected yet — ask for it
        if not intent:
            return _result(
                response_key="ASK_INTENT",
                next_state="ASK_INTENT",
                retry_count=retry_count,
            )

        # Intent collected with sufficient confidence
        if intent != "unknown" and intent_confidence >= threshold:
            response_key = INTENT_TO_RESPONSE.get(intent, "FALLBACK_LOW_CONFIDENCE")
            logger.info(f"Intent accepted: {intent} → {response_key}")
            return _result(
                response_key=response_key,
                next_state="DONE",
                retry_count=retry_count,
                done=True,
                log_event=True,
            )

        # Low confidence — retry
        if retry_count < max_retries:
            logger.info(f"Intent confidence low — retry {retry_count + 1}/{max_retries}")
            return _result(
                response_key="ASK_INTENT_RETRY",
                next_state="ASK_INTENT",
                retry_count=retry_count + 1,
            )

        # Retries exhausted
        logger.warning("Intent collection failed after max retries.")
        return _result(
            response_key="FALLBACK_LOW_CONFIDENCE",
            next_state="DONE",
            retry_count=retry_count,
            done=True,
            log_event=True,
        )

    # ----------------------------------------------------------------
    # Unexpected state — safety net
    # ----------------------------------------------------------------
    logger.error(f"Unexpected dialogue stage: '{stage}' — resetting.")
    return _result(
        response_key="FALLBACK_LOW_CONFIDENCE",
        next_state="DONE",
        retry_count=0,
        done=True,
        log_event=True,
    )


def _result(
    response_key: str,
    next_state: str,
    retry_count: int,
    done: bool = False,
    log_event: bool = False,
) -> dict:
    """Build a standardized dialogue step result."""
    return {
        "response_key": response_key,
        "next_state":   next_state,
        "retry_count":  retry_count,
        "done":         done,
        "log_event":    log_event,
    }


def init_state() -> dict:
    """Return a fresh dialogue state. Call this for every new visitor."""
    return {
        "stage":       "START",
        "retry_count": 0,
    }
