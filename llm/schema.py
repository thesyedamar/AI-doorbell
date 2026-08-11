"""
llm/schema.py — Pydantic Response Schemas
==========================================
Defines structured response schemas for Ollama LLM outputs.
Uses Pydantic v2 for validation.

Every LLM response is validated against these schemas before
being used by the rest of the system. Invalid responses are
caught here and never reach the dialogue manager.
"""

from typing import Literal, Optional
from pydantic import BaseModel, Field, field_validator


class IntentResponse(BaseModel):
    """
    Schema for Ollama intent detection response.

    Expected JSON from Ollama:
    {
        "intent": "delivery",
        "confidence": 0.92,
        "name": null,
        "raw_meaning": "Visitor is delivering a parcel"
    }
    """

    intent: Literal[
        "greeting",
        "visitor_intro",
        "delivery",
        "guest",
        "emergency",
        "unknown",
    ] = Field(
        default="unknown",
        description="Classified intent of the visitor's speech"
    )

    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Confidence score between 0.0 and 1.0"
    )

    name: Optional[str] = Field(
        default=None,
        description="Visitor name if mentioned in speech, else null"
    )

    raw_meaning: str = Field(
        default="",
        description="Brief English explanation of what the visitor said"
    )

    @field_validator("intent", mode="before")
    @classmethod
    def normalize_intent(cls, v):
        """Normalize intent to lowercase and fallback to unknown."""
        if isinstance(v, str):
            v = v.lower().strip()
            valid = {
                "greeting", "visitor_intro",
                "delivery", "guest",
                "emergency", "unknown",
            }
            return v if v in valid else "unknown"
        return "unknown"

    @field_validator("confidence", mode="before")
    @classmethod
    def clamp_confidence(cls, v):
        """Ensure confidence stays within [0.0, 1.0]."""
        try:
            v = float(v)
            return max(0.0, min(1.0, v))
        except (TypeError, ValueError):
            return 0.0

    @field_validator("name", mode="before")
    @classmethod
    def clean_name(cls, v):
        """Return None for empty or whitespace-only names."""
        if v is None:
            return None
        v = str(v).strip()
        return v if v else None


class NameExtractionResponse(BaseModel):
    """
    Schema for Ollama name extraction response.

    Expected JSON from Ollama:
    {
        "name": "احمد",
        "found": true,
        "confidence": 0.88
    }
    """

    name: Optional[str] = Field(
        default=None,
        description="Extracted Urdu name or null if not found"
    )

    found: bool = Field(
        default=False,
        description="True if a name was successfully extracted"
    )

    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Confidence score between 0.0 and 1.0"
    )

    @field_validator("name", mode="before")
    @classmethod
    def clean_name(cls, v):
        if v is None:
            return None
        v = str(v).strip()
        return v if v else None

    @field_validator("confidence", mode="before")
    @classmethod
    def clamp_confidence(cls, v):
        try:
            return max(0.0, min(1.0, float(v)))
        except (TypeError, ValueError):
            return 0.0
