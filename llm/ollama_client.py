"""
llm/ollama_client.py — Ollama LLM Client
=========================================
Handles all communication with the local Ollama server.
Provides two main functions:

    run_ollama_intent()       — classify visitor intent
    run_ollama_name_extract() — extract visitor name

Both functions:
    - POST to Ollama API
    - Parse and validate JSON response
    - Return typed Pydantic objects
    - Return None gracefully if Ollama is offline

Make sure Ollama is running before starting main.py:
    ollama serve
    ollama pull qwen2.5:3b-instruct
"""

import json
import logging
import requests
from typing import Optional

from config import cfg
from llm.schema import IntentResponse, NameExtractionResponse
from llm.intent_prompt import INTENT_PROMPT, NAME_EXTRACTION_PROMPT

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

def is_ollama_running() -> bool:
    """
    Check if Ollama server is reachable.
    Call this at startup to warn user if Ollama is offline.
    """
    try:
        response = requests.get(
            cfg.ollama.health_url,
            timeout=3,
        )
        return response.status_code == 200
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Core API call
# ---------------------------------------------------------------------------

def _call_ollama(prompt: str, model: str = None) -> Optional[str]:
    """
    Send a prompt to Ollama and return the raw text response.

    Args:
        prompt: Full prompt string (system + user input combined)
        model:  Ollama model name. Defaults to cfg.ollama.model

    Returns:
        Raw response text or None if request failed.
    """
    model = model or cfg.ollama.model

    payload = {
        "model":  model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": cfg.ollama.temperature,
            "top_p":       cfg.ollama.top_p,
        },
    }

    try:
        response = requests.post(
            cfg.ollama.generate_url,
            json=payload,
            timeout=cfg.ollama.timeout_seconds,
        )
        response.raise_for_status()
        data = response.json()
        return data.get("response", "").strip()

    except requests.exceptions.ConnectionError:
        logger.warning(
            "Ollama is not running. Start it with: ollama serve\n"
            "Falling back to unknown intent."
        )
        return None

    except requests.exceptions.Timeout:
        logger.warning(
            f"Ollama timed out after {cfg.ollama.timeout_seconds}s."
        )
        return None

    except Exception as e:
        logger.error(f"Ollama API error: {e}")
        return None


def _parse_json_response(raw: str) -> Optional[dict]:
    """
    Parse JSON from Ollama response.
    Handles cases where Ollama adds extra text around the JSON.
    """
    if not raw:
        return None

    # Try direct parse first
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # Try extracting JSON block from mixed response
    try:
        start = raw.find("{")
        end   = raw.rfind("}") + 1
        if start != -1 and end > start:
            return json.loads(raw[start:end])
    except json.JSONDecodeError:
        pass

    logger.warning(f"Could not parse JSON from Ollama response: '{raw[:100]}'")
    return None


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------

def run_ollama_intent(text: str) -> Optional[IntentResponse]:
    """
    Classify visitor intent using Ollama LLM.

    Args:
        text: Urdu transcript from Whisper

    Returns:
        IntentResponse object or None if Ollama unavailable.
    """
    if not text or not text.strip():
        return None

    prompt = INTENT_PROMPT + text.strip()
    logger.info(f"Sending to Ollama for intent: '{text[:60]}'")

    raw = _call_ollama(prompt, model=cfg.ollama.intent_model)
    if raw is None:
        return None

    data = _parse_json_response(raw)
    if data is None:
        return None

    try:
        result = IntentResponse(**data)
        logger.info(
            f"Ollama intent result | "
            f"intent={result.intent} "
            f"confidence={result.confidence:.3f} "
            f"name={result.name}"
        )
        return result

    except Exception as e:
        logger.error(f"IntentResponse validation failed: {e}")
        return None


def run_ollama_name_extract(text: str) -> Optional[str]:
    """
    Extract visitor name from Urdu speech using Ollama LLM.

    Args:
        text: Urdu transcript from Whisper

    Returns:
        Extracted name string or None if not found / unavailable.
    """
    if not text or not text.strip():
        return None

    prompt = NAME_EXTRACTION_PROMPT + text.strip()
    logger.info(f"Sending to Ollama for name extraction: '{text[:60]}'")

    raw = _call_ollama(prompt, model=cfg.ollama.name_extraction_model)
    if raw is None:
        return None

    data = _parse_json_response(raw)
    if data is None:
        return None

    try:
        result = NameExtractionResponse(**data)
        if result.found and result.name:
            logger.info(f"Ollama extracted name: '{result.name}'")
            return result.name
        return None

    except Exception as e:
        logger.error(f"NameExtractionResponse validation failed: {e}")
        return None
