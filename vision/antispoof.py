"""
vision/antispoof.py — Anti-Spoofing Detector (Keras MobileNetV2)
=================================================================
Loads the trained MobileNetV2 model from:
    models/antispoofing_model.json
    models/antispoofing_model.h5

Output: sigmoid score
    score >= threshold → REAL person ✅
    score <  threshold → FAKE/spoof ❌

Train the model with:
    python prepare_nuaa.py --nuaa /path/to/nuaa
    python training/train_antispoof.py
"""

import cv2
import logging
import numpy as np
from typing import Tuple

from config import cfg

logger = logging.getLogger(__name__)


class AntiSpoofDetector:

    def __init__(self):
        self.model     = None
        self.enabled   = cfg.features.anti_spoofing
        self.threshold = cfg.vision.antispoof_real_threshold

        if self.enabled:
            self._load_model()
        else:
            logger.warning("Anti-spoofing DISABLED via feature flag.")

    def _load_model(self) -> None:
        json_path    = cfg.paths.antispoof_json
        weights_path = cfg.paths.antispoof_weights

        if not json_path.exists() or not weights_path.exists():
            logger.warning(
                "Anti-spoof model files not found.\n"
                f"  Expected: {json_path}\n"
                f"  Expected: {weights_path}\n"
                "  Run: python prepare_nuaa.py --nuaa /path/to/nuaa\n"
                "  Then: python training/train_antispoof.py\n"
                "  Running WITHOUT anti-spoofing until model is trained."
            )
            self.enabled = False
            return

        try:
            from tensorflow.keras.models import model_from_json
            with open(json_path, "r", encoding="utf-8") as f:
                self.model = model_from_json(f.read())
            self.model.load_weights(str(weights_path))
            logger.info(
                f"Anti-spoof model loaded | threshold={self.threshold}"
            )
        except Exception as e:
            logger.error(f"Failed to load anti-spoof model: {e}")
            self.enabled = False

    def predict(self, face_crop: np.ndarray) -> Tuple[bool, float]:
        """
        Predict whether a face crop is real or spoofed.

        Args:
            face_crop: BGR numpy array of the face region

        Returns:
            (is_real: bool, confidence: float)
            confidence = real probability [0.0 - 1.0]
            Higher = more likely REAL.

        Threshold:
            confidence >= threshold → REAL  ✅
            confidence <  threshold → FAKE  ❌
        """
        if not self.enabled or self.model is None:
            return True, 1.0

        if face_crop is None or face_crop.size == 0:
            return True, 1.0

        try:
            resized    = cv2.resize(face_crop, (224, 224))
            rgb        = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            normalized = rgb.astype(np.float32) / 255.0
            batch      = np.expand_dims(normalized, axis=0)

            score   = float(self.model.predict(batch, verbose=0)[0][0])
            is_real = score >= self.threshold

            logger.debug(
                f"Anti-spoof | score={score:.4f} "
                f"threshold={self.threshold} "
                f"result={'REAL' if is_real else 'FAKE'}"
            )
            return is_real, score

        except Exception as e:
            logger.error(f"Anti-spoof prediction error: {e}")
            return True, 1.0

    def predict_from_frame(
        self,
        frame: np.ndarray,
        bbox: tuple,
    ) -> Tuple[bool, float]:
        """Predict from full frame + bounding box."""
        if not self.enabled:
            return True, 1.0
        x1, y1, x2, y2 = [int(c) for c in bbox]
        h, w = frame.shape[:2]
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(w, x2); y2 = min(h, y2)
        face_crop = frame[y1:y2, x1:x2]
        if face_crop.size == 0:
            return True, 1.0
        return self.predict(face_crop)

    def predict_detailed(self, face_crop: np.ndarray) -> dict:
        """Detailed result for test_antispoof.py."""
        is_real, confidence = self.predict(face_crop)
        return {
            "is_real":       is_real,
            "confidence":    round(confidence, 4),
            "model_1_score": round(confidence, 4),
            "model_2_score": None,
            "threshold":     self.threshold,
            "enabled":       self.enabled,
        }

    def predict_detailed_from_frame(
        self,
        frame: np.ndarray,
        bbox: tuple,
    ) -> dict:
        """Detailed result using full frame."""
        is_real, confidence = self.predict_from_frame(frame, bbox)
        return {
            "is_real":       is_real,
            "confidence":    round(confidence, 4),
            "model_1_score": round(confidence, 4),
            "model_2_score": None,
            "threshold":     self.threshold,
            "enabled":       self.enabled,
        }
