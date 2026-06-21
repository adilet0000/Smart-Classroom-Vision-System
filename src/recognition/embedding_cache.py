"""
In-memory cache of face embeddings loaded from the SQLite database.

Motivation:
    The original code called db.get_all_embeddings() inside the recognition
    hot path (every N frames × every active track).  For 25 students at 30 fps
    that is ~75 SQLite SELECT + JSON-decode operations per second — completely
    unnecessary because the embedding table never changes during a session.

Usage:
    cache = EmbeddingCache.load(face_db)   # once at startup
    best_code, score = cache.find_best(query_embedding)  # in the hot path
    cache.reload(face_db)                  # call if new faces are enrolled live
"""
from __future__ import annotations

import numpy as np

from src.recognition.face_database import FaceDatabase


class EmbeddingCache:
    """Pre-loaded, L2-normalized face embeddings for fast cosine similarity search."""

    def __init__(self) -> None:
        # Parallel lists: codes[i] matches normed_embeddings[i]
        self._codes: list[str] = []
        self._normed: list[np.ndarray] = []  # each already unit-norm

    # ── Construction ──────────────────────────────────────────────────────

    @classmethod
    def load(cls, db: FaceDatabase) -> "EmbeddingCache":
        """Build a cache from the current database state."""
        obj = cls()
        obj.reload(db)
        return obj

    def reload(self, db: FaceDatabase) -> None:
        """
        Re-read all embeddings from the database.
        Call this if faces are enrolled while the system is running.
        """
        rows = db.get_all_embeddings()   # list[tuple[str, np.ndarray]]
        codes: list[str] = []
        normed: list[np.ndarray] = []

        for student_code, embedding in rows:
            norm = float(np.linalg.norm(embedding))
            if norm < 1e-6:
                continue  # skip degenerate embeddings
            codes.append(student_code)
            normed.append((embedding / norm).astype(np.float32))

        self._codes = codes
        self._normed = normed

    # ── Search ────────────────────────────────────────────────────────────

    def find_best(
        self,
        query: np.ndarray,
        threshold: float = 0.45,
    ) -> tuple[str, float] | None:
        """
        Return (student_code, similarity) of the best match above threshold,
        or None if no match meets the threshold.

        Query does not need to be pre-normalized.
        When a student has multiple enrolled embeddings, the best score across
        all of their embeddings is used.
        """
        if not self._codes:
            return None

        # Normalize query once
        q_norm = float(np.linalg.norm(query))
        if q_norm < 1e-6:
            return None
        q = (query / q_norm).astype(np.float32)

        # Dot product against all stored embeddings in one BLAS call
        # (faster than a Python loop for > ~10 entries)
        matrix = np.stack(self._normed, axis=0)   # (N, D)
        similarities = matrix @ q                  # (N,)

        # Aggregate per student: keep max similarity across multiple embeddings
        best_per_student: dict[str, float] = {}
        for code, sim in zip(self._codes, similarities.tolist()):
            if sim > best_per_student.get(code, -1.0):
                best_per_student[code] = sim

        if not best_per_student:
            return None

        best_code = max(best_per_student, key=best_per_student.__getitem__)
        best_score = best_per_student[best_code]

        if best_score < threshold:
            return None

        return best_code, float(best_score)

    # ── Diagnostics ───────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._codes)

    def unique_students(self) -> int:
        return len(set(self._codes))
