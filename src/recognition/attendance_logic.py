from __future__ import annotations

from dataclasses import dataclass, field

from src.recognition.face_database import FaceDatabase
from src.recognition.face_embedder import FaceEmbedder


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

   def recognize_embedding(
      self,
      track_id: int,
      embedding,
      db: FaceDatabase,
      embedder: FaceEmbedder,
   ) -> str | None:
      known = db.get_all_embeddings()
      if not known:
         return None

      best_student_code = None
      best_score = -1.0

      for student_code, known_embedding in known:
         score = embedder.cosine_similarity(embedding, known_embedding)
         if score > best_score:
            best_score = score
            best_student_code = student_code

      state = self.track_states.setdefault(track_id, TrackIdentityState())

      if best_student_code is None or best_score < self.similarity_threshold:
         state.consecutive_matches = 0
         return state.confirmed_student_code

      if state.best_student_code == best_student_code:
         state.consecutive_matches += 1
      else:
         state.best_student_code = best_student_code
         state.best_score = best_score
         state.consecutive_matches = 1

      state.best_score = best_score

      if state.consecutive_matches >= self.confirm_matches:
         state.confirmed_student_code = best_student_code
         db.mark_present(best_student_code, track_id)

      return state.confirmed_student_code