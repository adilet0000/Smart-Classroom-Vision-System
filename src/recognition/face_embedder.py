from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from insightface.app import FaceAnalysis

from src.core.geometry import crop_head_region


@dataclass
class FaceEmbeddingResult:
   embedding: np.ndarray
   bbox: tuple[int, int, int, int]
   det_score: float


class FaceEmbedder:
   def __init__(self) -> None:
      # Only the detection + recognition models are needed for embeddings;
      # loading genderage/landmark models wastes startup time and memory.
      self.app = FaceAnalysis(
         name="buffalo_l",
         providers=["CPUExecutionProvider"],
         allowed_modules=["detection", "recognition"],
      )
      self.app.prepare(ctx_id=0, det_size=(640, 640))

   def get_embedding(self, image: np.ndarray) -> Optional[FaceEmbeddingResult]:
      faces = self.app.get(image)

      if not faces:
         return None

      best_face = max(faces, key=lambda f: float(f.det_score))

      x1, y1, x2, y2 = [int(v) for v in best_face.bbox]
      embedding = np.asarray(best_face.embedding, dtype=np.float32)

      return FaceEmbeddingResult(
         embedding=embedding,
         bbox=(x1, y1, x2, y2),
         det_score=float(best_face.det_score),
      )

   @staticmethod
   def crop_person_region(
      frame: np.ndarray,
      person_box: tuple[int, int, int, int],
      top_ratio: float = 0.55,
      side_padding_ratio: float = 0.08,
   ) -> np.ndarray:
      # Delegates to the shared, unit-tested geometry helper so cropping and
      # clamping behave identically to the detector.
      return crop_head_region(
         frame, person_box, top_ratio=top_ratio, side_padding_ratio=side_padding_ratio
      )

   @staticmethod
   def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
      a_norm = np.linalg.norm(a)
      b_norm = np.linalg.norm(b)

      if a_norm == 0.0 or b_norm == 0.0:
         return 0.0

      return float(np.dot(a, b) / (a_norm * b_norm))