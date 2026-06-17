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
   class_id: int = 0


class FaceDetector:
   """
   Пока это detector людей через YOLO.
   class 0 в COCO = person.
   """

   def __init__(self, model_name: str = "yolov8n.pt") -> None:
      self.model = YOLO(model_name)

   def detect(self, frame: np.ndarray) -> List[FaceBox]:
      results = self.model(
         frame,
         verbose=False,
         imgsz=640,
         conf=0.25,
         device="mps",
      )

      detections: List[FaceBox] = []

      for result in results:
         for box in result.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])

            if cls != 0:
               continue

            x1, y1, x2, y2 = box.xyxy[0].tolist()

            detections.append(
               FaceBox(
                  x1=int(x1),
                  y1=int(y1),
                  x2=int(x2),
                  y2=int(y2),
                  confidence=conf,
                  class_id=cls,
               )
            )

      return detections