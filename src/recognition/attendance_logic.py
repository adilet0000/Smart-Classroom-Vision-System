"""
Attendance logic — identity confirmation with N-of-M voting.

The original design queried SQLite on every recognition attempt.
This module now receives the pre-computed (student_code, score) pair
from EmbeddingCache.find_best() and only manages the confirmation state.

The FaceDatabase is still accepted for mark_present() calls only.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from src.recognition.face_database import FaceDatabase


@dataclass
class TrackIdentityState:
    best_student_code: str | None = None
    best_score: float = 0.0
    consecutive_matches: int = 0
    confirmed_student_code: str | None = None


@dataclass
class AttendanceManager:
    similarity_threshold: float = 0.45
    confirm_matches: int = 3
    track_states: dict[int, TrackIdentityState] = field(default_factory=dict)

    def confirm_match(
        self,
        *,
        track_id: int,
        student_code: str,
        score: float,
        db: FaceDatabase,
        session_id: str,
    ) -> str | None:
        """
        Accept a (student_code, score) pair from EmbeddingCache.find_best()
        and apply N-of-M confirmation logic.

        Returns the confirmed student_code once the threshold is reached,
        or None while still accumulating evidence.
        """
        state = self.track_states.setdefault(track_id, TrackIdentityState())

        # Already confirmed — no-op
        if state.confirmed_student_code is not None:
            return state.confirmed_student_code

        if score < self.similarity_threshold:
            state.consecutive_matches = 0
            return None

        if state.best_student_code == student_code:
            state.consecutive_matches += 1
        else:
            state.best_student_code = student_code
            state.consecutive_matches = 1

        state.best_score = score

        if state.consecutive_matches >= self.confirm_matches:
            state.confirmed_student_code = student_code
            db.mark_present(student_code, track_id, session_id)

        return state.confirmed_student_code

    def remove_tracks(self, track_ids: set[int]) -> None:
        for track_id in track_ids:
            self.track_states.pop(track_id, None)
