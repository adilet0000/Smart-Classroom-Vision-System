from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


@dataclass
class FaceBox:
   x1: int
   y1: int
   x2: int
   y2: int
   confidence: float


class FaceDetector:
   """
   Заглушка детектора лиц.
   На следующей неделе подключим MediaPipe или YOLO.
   """

   def __init__(self) -> None:
      pass

   def detect(self, frame: np.ndarray) -> List[FaceBox]:
      # Пока ничего не детектим
      return []
