"""
vision/recognition.py — Face Recognition
=========================================
Bridges the embedder and FAISS index.
Answers two questions:
    1. Is this face known or unknown?
    2. If known, who is it?

Also handles saving new face crops to disk.
"""

import cv2
import logging
import numpy as np
from datetime import datetime
from typing import Tuple

from config import cfg
from vision.embedding import FaceEmbedder
from vision.faiss_store import faiss_index

logger = logging.getLogger(__name__)

# Module-level embedder instance
_embedder = FaceEmbedder()


def recognize_face(
    embedding: np.ndarray,
    threshold: float = None,
) -> Tuple[str, str, int]:
    """
    Query FAISS to identify a face embedding.

    Args:
        embedding:  512-dim float32 numpy array
        threshold:  L2 distance cutoff (default from config)

    Returns:
        Tuple of (status, name, face_index)
        status: "Known" or "Unknown"
        name:   visitor name or "Unknown"
        face_index: FAISS index or -1
    """
    found, idx, name = faiss_index.search(embedding, threshold)

    if found:
        logger.info(f"Face recognized | name={name} idx={idx}")
        return "Known", name, idx

    logger.info("Face not recognized — Unknown visitor.")
    return "Unknown", "Unknown", -1


def add_face_to_db(
    face_crop: np.ndarray,
    embedding: np.ndarray,
    name: str,
) -> int:
    """
    Save a new face crop image and add embedding to FAISS.
    Used during enrollment (manage_faces.py) and
    optionally during runtime for auto-enrollment.

    Args:
        face_crop:  BGR numpy array of the face region
        embedding:  512-dim float32 numpy array
        name:       Visitor name to associate

    Returns:
        FAISS index of the newly added face
    """
    # Save face crop image
    timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name  = name.replace(" ", "_")
    filename   = f"{safe_name}_{timestamp}.jpg"
    save_path  = cfg.paths.faces_dir / filename

    try:
        cv2.imwrite(str(save_path), face_crop)
        logger.info(f"Face crop saved: {save_path}")
    except Exception as e:
        logger.warning(f"Could not save face crop: {e}")

    # Add to FAISS + DB
    idx = faiss_index.add_face(embedding, name)
    logger.info(f"Face added | name={name} idx={idx}")
    return idx


def get_face_crop(frame: np.ndarray, face) -> np.ndarray:
    """
    Extract the face crop from a full frame using the bounding box.

    Args:
        frame: Full BGR frame
        face:  InsightFace face object

    Returns:
        Cropped BGR face image
    """
    x1, y1, x2, y2 = [int(c) for c in face.bbox]
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(frame.shape[1], x2)
    y2 = min(frame.shape[0], y2)
    return frame[y1:y2, x1:x2].copy()


def extract_and_recognize(
    frame: np.ndarray,
    face,
) -> Tuple[str, str, int, np.ndarray, np.ndarray]:
    """
    Convenience function: extract embedding + recognize in one call.

    Returns:
        (status, name, face_index, embedding, face_crop)
    """
    crop      = get_face_crop(frame, face)
    embedding = _embedder.get_embedding_safe(face)

    if embedding is None:
        return "Unknown", "Unknown", -1, None, crop

    status, name, idx = recognize_face(embedding)
    return status, name, idx, embedding, crop
