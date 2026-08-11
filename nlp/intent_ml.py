"""
nlp/intent_ml.py — ML Intent Classifier
=========================================
Classifies visitor intent from Urdu transcript using
a trained TF-IDF + SVM pipeline.

Intents:
    greeting    — visitor just saying hello
    delivery    — parcel / courier delivery
    guest       — visiting someone in the house
    emergency   — urgent help needed
    unknown     — cannot determine intent

Model file: nlp/models/intent_model.pkl
Train with: python train_intent.py

If model file is missing, returns ("unknown", 0.0) gracefully.
"""

import logging
import numpy as np
from typing import Tuple, Optional

from config import cfg

logger = logging.getLogger(__name__)

# Module-level model cache
_intent_model = None


def _get_model():
    """
    Lazy-load intent model on first use.
    Returns None if model file not found.
    """
    global _intent_model

    if _intent_model is not None:
        return _intent_model

    model_path = cfg.paths.intent_model_path

    if not model_path.exists():
        logger.warning(
            f"Intent model not found: {model_path}\n"
            "  Train it with: python train_intent.py\n"
            "  Falling back to Ollama for all intent detection."
        )
        return None

    try:
        import joblib
        _intent_model = joblib.load(str(model_path))
        logger.info(f"Intent model loaded: {model_path}")
        return _intent_model

    except Exception as e:
        logger.error(f"Failed to load intent model: {e}")
        return None





def detect_intent(
    text: str,
    threshold: float = None,
) -> Tuple[str, float]:
    """
    Classify intent from Urdu transcript using ML model.

    Args:
        text:      Urdu transcript string
        threshold: Minimum confidence to trust the result.
                   Below this → caller should use Ollama fallback.
                   Defaults to cfg.nlp.ml_confidence_threshold (0.60)

    Returns:
        Tuple of (intent_label: str, confidence: float)
        Returns ("unknown", 0.0) if model unavailable or below threshold.
    """
    if not text or not text.strip():
        return "unknown", 0.0

    cutoff = threshold if threshold is not None else cfg.nlp.ml_confidence_threshold
    model  = _get_model()

    if model is None:
        return "unknown", 0.0

    try:
        # Get decision scores for all classes
        probs_all   = model.predict_proba([text.strip()])
        probs       = probs_all[0]
        best_idx    = int(np.argmax(probs))
        confidence  = float(probs[best_idx])
        intent      = model.classes_[best_idx]

        logger.debug(
            f"ML intent | text='{text[:40]}' "
            f"intent={intent} confidence={confidence:.3f}"
        )

        if confidence < cutoff:
            logger.info(
                f"ML confidence too low: {confidence:.3f} < {cutoff} "
                f"— will use Ollama fallback."
            )
            return "unknown", confidence

        return intent, confidence

    except Exception as e:
        logger.error(f"Intent classification error: {e}")
        return "unknown", 0.0


def get_all_probabilities(text: str) -> dict:
    """
    Return confidence scores for ALL intent classes.
    Useful for debugging and training data analysis.

    Returns dict: {intent_label: confidence, ...}
    """
    model = _get_model()
    if model is None or not text:
        return {}

    try:
        scores = model.decision_function([text.strip()])
        probs  = _softmax(scores[0])
        return {
            label: round(float(prob), 4)
            for label, prob in zip(model.classes_, probs)
        }
    except Exception as e:
        logger.error(f"get_all_probabilities error: {e}")
        return {}
