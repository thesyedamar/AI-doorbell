"""
vision/capture.py — Face Camera & Detection
============================================
Handles video capture and face detection using InsightFace buffalo_l.

Quality gates (all must pass before a face is processed):
    1. Blur check       — Laplacian variance >= 35
    2. Landmark check   — eye distance >= 10px, mouth >= 6px
    3. Area check       — face area >= 2500px
    4. Stability gate   — face present for >= 1.5 seconds continuously

Returns:
    (frame, face, status_string)
    face is None if any gate fails
"""

import cv2
import time
import logging
import numpy as np
from typing import Tuple, Optional

from config import cfg

logger = logging.getLogger(__name__)


class FaceCamera:

    def __init__(self):
        self.app             = None
        self.cap             = None
        self._stable_since   = None   # timestamp when face first appeared
        self._last_face_time = None   # timestamp of last detected face

        self._init_insightface()
        self._init_camera()

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def _init_insightface(self) -> None:
        """Load InsightFace buffalo_l model on CPU."""
        try:
            import insightface
            self.app = insightface.app.FaceAnalysis(
                name=cfg.vision.insightface_model,
                providers=["CPUExecutionProvider"],
            )
            self.app.prepare(ctx_id=0, det_size=cfg.vision.det_size)
            logger.info(
                f"InsightFace loaded | model={cfg.vision.insightface_model} "
                f"det_size={cfg.vision.det_size}"
            )
        except Exception as e:
            logger.error(f"Failed to load InsightFace: {e}")
            raise

    def _init_camera(self) -> None:
        """Open webcam or RTSP stream depending on feature flags."""
        if cfg.features.rtsp_camera and cfg.vision.rtsp_url:
            source = cfg.vision.rtsp_url
            logger.info(f"Opening RTSP stream: {source}")
        else:
            source = cfg.vision.camera_index
            logger.info(f"Opening webcam index: {source}")

        self.cap = cv2.VideoCapture(source)

        if not self.cap.isOpened():
            raise RuntimeError(
                f"Cannot open camera source: {source}. "
                "Check CAMERA_INDEX in .env or ensure webcam is connected."
            )

        # Set reasonable defaults
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS,          cfg.system.max_fps)
        logger.info("Camera opened successfully.")

    # ------------------------------------------------------------------
    # Quality gates
    # ------------------------------------------------------------------

    def _check_blur(self, face_crop) -> bool:
        """Return True if image is sharp enough."""
        gray      = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        variance  = cv2.Laplacian(gray, cv2.CV_64F).var()
        passed    = variance >= cfg.vision.blur_threshold
        logger.debug(f"Blur check | variance={variance:.1f} passed={passed}")
        return passed

    def _check_landmarks(self, face) -> bool:
        """Return True if facial landmarks are clearly visible."""
        if face.kps is None:
            return False

        kps = face.kps
        # kps layout: [left_eye, right_eye, nose, left_mouth, right_mouth]
        eye_dist   = np.linalg.norm(kps[0] - kps[1])
        mouth_dist = np.linalg.norm(kps[3] - kps[4])

        passed = (
            eye_dist   >= cfg.vision.min_eye_distance and
            mouth_dist >= cfg.vision.min_mouth_distance
        )
        logger.debug(
            f"Landmark check | eye_dist={eye_dist:.1f} "
            f"mouth_dist={mouth_dist:.1f} passed={passed}"
        )
        return passed

    def _check_area(self, face) -> bool:
        """Return True if face is large enough in frame."""
        x1, y1, x2, y2 = face.bbox
        area   = (x2 - x1) * (y2 - y1)
        passed = area >= cfg.vision.min_face_area
        logger.debug(f"Area check | area={area:.0f} passed={passed}")
        return passed

    def _check_stability(self) -> bool:
        """
        Return True if face has been continuously present
        for at least stability_seconds.
        """
        now = time.time()
        if self._stable_since is None:
            self._stable_since = now
            return False

        elapsed = now - self._stable_since
        passed  = elapsed >= cfg.vision.stability_seconds
        logger.debug(f"Stability check | elapsed={elapsed:.2f}s passed={passed}")
        return passed

    # ------------------------------------------------------------------
    # Main read loop
    # ------------------------------------------------------------------

    def read(self) -> Tuple[Optional[np.ndarray], Optional[object], str]:
        """
        Read one frame and run face detection + quality gates.

        Returns:
            (frame, face, status)

            status values:
                "NO_FRAME"       — camera read failed
                "NO_FACE"        — no face detected
                "LOW_SCORE"      — detection confidence too low
                "BLUR"           — face too blurry
                "LANDMARK"       — landmarks not visible
                "SMALL"          — face area too small
                "STABILIZING"    — face present but not stable yet
                "READY"          — all gates passed, face ready to process
        """
        ret, frame = self.cap.read()
        if not ret or frame is None:
            logger.warning("Camera read failed.")
            self._stable_since = None
            return None, None, "NO_FRAME"

        # Detect faces
        faces = self.app.get(frame)

        if not faces:
            self._stable_since = None
            return frame, None, "NO_FACE"

        # Use face with highest detection score
        face = max(faces, key=lambda f: f.det_score)

        # Detection score gate
        if face.det_score < cfg.vision.min_det_score:
            self._stable_since = None
            return frame, None, "LOW_SCORE"

        # Extract face crop for quality checks
        x1, y1, x2, y2 = [int(c) for c in face.bbox]
        x1 = max(0, x1); y1 = max(0, y1)
        face_crop = frame[y1:y2, x1:x2]

        if face_crop.size == 0:
            self._stable_since = None
            return frame, None, "NO_FACE"

        # Quality gates
        if not self._check_blur(face_crop):
            self._stable_since = None
            return frame, None, "BLUR"

        if not self._check_landmarks(face):
            self._stable_since = None
            return frame, None, "LANDMARK"

        if not self._check_area(face):
            self._stable_since = None
            return frame, None, "SMALL"

        # Stability gate
        if not self._check_stability():
            return frame, None, "STABILIZING"

        # All gates passed
        self._stable_since = None   # reset for next visitor
        return frame, face, "READY"

    def reset_stability(self) -> None:
        """Call this after processing a visitor to reset the stability timer."""
        self._stable_since = None

    def draw_debug(self, frame: np.ndarray, face, status: str) -> np.ndarray:
        """
        Draw detection overlay on frame for the preview window.
        Only active when FEATURE_DEBUG_OVERLAY=1 in .env
        """
        if not cfg.features.debug_overlay:
            # Draw minimal status text even without debug overlay
            cv2.putText(
                frame, f"Status: {status}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 255, 0), 2
            )
            return frame

        # Full debug overlay
        if face is not None:
            x1, y1, x2, y2 = [int(c) for c in face.bbox]
            color = (0, 255, 0) if status == "READY" else (0, 165, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame,
                f"Score:{face.det_score:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
            )

            # Draw landmarks
            if face.kps is not None:
                for kp in face.kps:
                    cv2.circle(frame, (int(kp[0]), int(kp[1])), 3, (255, 0, 0), -1)

        cv2.putText(
            frame, f"Status: {status}",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
            0.7, (0, 255, 0), 2
        )
        return frame

    def release(self) -> None:
        """Release camera resource cleanly."""
        if self.cap and self.cap.isOpened():
            self.cap.release()
            logger.info("Camera released.")
