"""
vision/embedding.py — ArcFace Face Embedder
============================================
Extracts 512-dimensional normalized face embeddings
from InsightFace face objects.

The normed_embedding is already L2-normalized by InsightFace,
which is exactly what FAISS IndexFlatL2 expects.
"""

import logging
import numpy as np

from config import cfg

logger = logging.getLogger(__name__)


class FaceEmbedder:

    def __init__(self):
        logger.info("FaceEmbedder ready | dim=512 (ArcFace normed)")

    def get_embedding(self, face) -> np.ndarray:
        """
        Extract the 512-dim normalized embedding from an InsightFace face object.

        Args:
            face: InsightFace face object (from app.get(frame))

        Returns:
            float32 numpy array of shape (512,)

        Raises:
            ValueError: if the face object has no embedding
        """
        if face is None:
            raise ValueError("FaceEmbedder received None face object.")

        if not hasattr(face, "normed_embedding") or face.normed_embedding is None:
            raise ValueError("Face object has no normed_embedding. Detection may have failed.")

        embedding = np.array(face.normed_embedding, dtype=np.float32)

        if embedding.shape[0] != cfg.recognition.embedding_dim:
            raise ValueError(
                f"Unexpected embedding dim: {embedding.shape[0]} "
                f"(expected {cfg.recognition.embedding_dim})"
            )

        logger.debug(f"Embedding extracted | shape={embedding.shape} norm={np.linalg.norm(embedding):.4f}")
        return embedding

    def get_embedding_safe(self, face):
        """
        Safe version — returns None instead of raising on failure.
        Use this in the main loop where you want graceful fallback.
        """
        try:
            return self.get_embedding(face)
        except Exception as e:
            logger.warning(f"Embedding extraction failed: {e}")
            return None
