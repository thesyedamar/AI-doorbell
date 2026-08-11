"""
vision/db_tools.py — Database Helper Functions
===============================================
Thin wrapper around Database.insert() and Database.upsert_visitor().
All modules call save_event() instead of touching the DB directly.
This keeps database logic in one place.
"""

import logging
from typing import Optional

from vision.database import db

logger = logging.getLogger(__name__)


def save_event(
    name: str,
    face_index: int,
    image_path: str = "",
    speech_text: str = "",
    intent: str = "unknown",
    spoken_name: str = "",
    conflict: int = 0,
    language: str = "ur",
    duration_ms: int = 0,
    extra: str = "",
) -> int:
    """
    Save a complete visitor interaction to the database.
    Also updates the visitors registry for known faces.

    Args:
        name:        FAISS-matched name or 'Unknown'
        face_index:  FAISS index (-1 for unknown visitors)
        image_path:  Path to the saved face crop image
        speech_text: Raw transcript from Whisper
        intent:      Detected intent (delivery, guest, emergency, etc.)
        spoken_name: Name extracted from speech
        conflict:    1 if FAISS name and spoken name don't match
        language:    'ur' (Urdu) or 'ps' (Pashto)
        duration_ms: Total time taken for the dialogue in milliseconds
        extra:       JSON string for any plugin-specific extra data

    Returns:
        The new event row ID.
    """
    try:
        event_id = db.insert(
            name=name,
            face_index=face_index,
            image_path=image_path,
            speech_text=speech_text,
            intent=intent,
            spoken_name=spoken_name,
            conflict=conflict,
            language=language,
            duration_ms=duration_ms,
            extra=extra,
        )

        # Update visitor registry for known faces
        if face_index >= 0 and name and name != "Unknown":
            db.upsert_visitor(face_index=face_index, name=name)

        logger.info(
            f"Event saved | id={event_id} name={name} "
            f"intent={intent} conflict={conflict}"
        )
        return event_id

    except Exception as e:
        logger.error(f"Failed to save event: {e}")
        return -1
