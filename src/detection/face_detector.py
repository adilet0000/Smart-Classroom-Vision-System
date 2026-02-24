from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
from ultralytics import YOLO


@dataclass
class FaceBox:
   x1: int
   y1: int
   x2: int
   y2: int
   confidence: float


class FaceDetector:
   """
   YOLOv8 person detector.
   Детектим людей (class 0 = person).
   """

   def __init__(self, model_name: str = "yolov8n.pt") -> None:
      self.model = YOLO(model_name)

   def detect(self, frame: np.ndarray) -> List[FaceBox]:
      results = self.model(frame, verbose=False)

      faces: List[FaceBox] = []

      for r in results:
         for box in r.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])

            # COCO class 0 = person
            if cls == 0:
               x1, y1, x2, y2 = box.xyxy[0].tolist()
               faces.append(
                  FaceBox(
                     int(x1),
                     int(y1),
                     int(x2),
                     int(y2),
                     conf,
                  )
               )

      return faces