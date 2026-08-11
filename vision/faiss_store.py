"""
vision/faiss_store.py — FAISS Vector Database
==============================================
Stores and searches 512-dim ArcFace embeddings.
Persists index to disk so enrolled faces survive restarts.

Storage:
    data/faiss.index  — the FAISS binary index
    SQLite visitors   — maps faiss_idx → name (replaces old pickle approach)
"""

import logging
import numpy as np
from pathlib import Path
from typing import Tuple, Optional

import faiss

from config import cfg
from vision.database import db

logger = logging.getLogger(__name__)


class FaissIndex:

    def __init__(self):
        self.dim        = cfg.recognition.embedding_dim      # 512
        self.threshold  = cfg.recognition.match_threshold    # 0.80 L2
        self.index_path = cfg.paths.faiss_index_path
        self.index      = self._load_or_create()
        logger.info(
            f"FAISS ready | vectors={self.index.ntotal} "
            f"dim={self.dim} threshold={self.threshold}"
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_or_create(self) -> faiss.IndexFlatL2:
        """Load existing index from disk or create a fresh one."""
        if self.index_path.exists():
            try:
                index = faiss.read_index(str(self.index_path))
                logger.info(f"FAISS index loaded from disk: {self.index_path}")
                return index
            except Exception as e:
                logger.warning(f"Could not load FAISS index ({e}). Creating new one.")

        index = faiss.IndexFlatL2(self.dim)
        logger.info("Created new FAISS index.")
        return index

    def _save(self) -> None:
        """Persist index to disk after every write."""
        try:
            faiss.write_index(self.index, str(self.index_path))
            logger.debug(f"FAISS index saved: {self.index_path}")
        except Exception as e:
            logger.error(f"Failed to save FAISS index: {e}")

    def _prepare(self, embedding: np.ndarray) -> np.ndarray:
        """Ensure embedding is float32 and shaped (1, dim) for FAISS."""
        vec = np.array(embedding, dtype=np.float32)
        if vec.ndim == 1:
            vec = vec.reshape(1, -1)
        return vec

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_face(self, embedding: np.ndarray, name: str) -> int:
        """
        Add a face embedding to the index.

        Args:
            embedding: 512-dim float32 numpy array
            name:      Visitor name to associate with this embedding

        Returns:
            The FAISS index position (used as face_index in DB)
        """
        vec = self._prepare(embedding)
        idx = self.index.ntotal          # next available slot
        self.index.add(vec)
        self._save()

        # Register in SQLite visitors table
        db.upsert_visitor(face_index=idx, name=name)

        logger.info(f"Face added to FAISS | name={name} idx={idx}")
        return idx

    def search(
        self,
        embedding: np.ndarray,
        threshold: Optional[float] = None,
    ) -> Tuple[bool, int, str]:
        """
        Search for the closest matching face.

        Args:
            embedding:  512-dim float32 numpy array
            threshold:  L2 distance cutoff (default from config)

        Returns:
            Tuple of (found: bool, face_index: int, name: str)
            If not found → (False, -1, "Unknown")
        """
        if self.index.ntotal == 0:
            logger.debug("FAISS index is empty — no faces enrolled yet.")
            return False, -1, "Unknown"

        cutoff = threshold if threshold is not None else self.threshold
        vec    = self._prepare(embedding)

        distances, indices = self.index.search(vec, k=cfg.recognition.search_k)
        dist = float(distances[0][0])
        idx  = int(indices[0][0])

        logger.debug(f"FAISS search | dist={dist:.4f} idx={idx} cutoff={cutoff}")

        if idx < 0 or dist > cutoff:
            return False, -1, "Unknown"

        # Look up name from SQLite
        name = db.get_visitor_name(idx) or "Unknown"
        return True, idx, name

    def total(self) -> int:
        """Return number of face vectors in the index."""
        return self.index.ntotal

    def rebuild_from_db(self) -> None:
        """
        Placeholder for future: re-embed all face crops and rebuild index.
        Called after a visitor deletion to keep FAISS and DB in sync.
        Currently logs a warning — full implementation requires stored crops.
        """
        logger.warning(
            "rebuild_from_db() called. Full rebuild requires re-embedding "
            "stored face crops. Feature coming in next phase."
        )


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------
faiss_index = FaissIndex()
