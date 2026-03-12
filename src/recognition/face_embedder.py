from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
from insightface.app import FaceAnalysis


@dataclass
class FaceEmbeddingResult:
   embedding: np.ndarray
   bbox: tuple[int, int, int, int]
   det_score: float


class FaceEmbedder:
   def __init__(self) -> None:
      self.app = FaceAnalysis(
         name="buffalo_l",
         providers=["CPUExecutionProvider"],
      )
      self.app.prepare(ctx_id=0, det_size=(640, 640))

   def get_embedding(self, image: np.ndarray) -> Optional[FaceEmbeddingResult]:
      faces = self.app.get(image)

      if not faces:
         return None

      # Берём самое уверенное лицо
      best_face = max(faces, key=lambda f: float(f.det_score))

      x1, y1, x2, y2 = [int(v) for v in best_face.bbox]
      embedding = np.asarray(best_face.embedding, dtype=np.float32)

      return FaceEmbeddingResult(
         embedding=embedding,
         bbox=(x1, y1, x2, y2),
         det_score=float(best_face.det_score),
      )

   @staticmethod
   def crop_head_region(frame: np.ndarray, person_box: tuple[int, int, int, int]) -> np.ndarray:
      x1, y1, x2, y2 = person_box
      h, w = frame.shape[:2]

      x1 = max(0, x1)
      y1 = max(0, y1)
      x2 = min(w, x2)
      y2 = min(h, y2)

      person_crop = frame[y1:y2, x1:x2]
      if person_crop.size == 0:
         return person_crop

      ph, pw = person_crop.shape[:2]

      # Берем верхние 45% человека, там обычно голова + плечи
      head_h = max(1, int(ph * 0.45))
      head_crop = person_crop[:head_h, :pw]

      return head_crop

   @staticmethod
   def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
      a_norm = np.linalg.norm(a)
      b_norm = np.linalg.norm(b)

      if a_norm == 0.0 or b_norm == 0.0:
         return 0.0

      return float(np.dot(a, b) / (a_norm * b_norm))