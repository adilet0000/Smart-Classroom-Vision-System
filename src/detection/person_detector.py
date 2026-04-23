from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
from ultralytics import YOLO


@dataclass
class PersonBox:
   x1: int
   y1: int
   x2: int
   y2: int
   confidence: float
   class_id: int = 0


class PersonDetector:
   """
   Detector of people in the frame.
   COCO class 0 = person.
   """

   def __init__(
      self,
      model_name: str = "yolov8n.pt",
      image_size: int = 640,
      confidence_threshold: float = 0.25,
      device: str = "auto",
   ) -> None:
      self.model = YOLO(model_name)
      self.image_size = image_size
      self.confidence_threshold = confidence_threshold
      self.device = None if device == "auto" else device

   def detect(self, frame: np.ndarray) -> List[PersonBox]:
      results = self.model(
         frame,
         verbose=False,
         imgsz=self.image_size,
         conf=self.confidence_threshold,
         device=self.device,
      )

      detections: List[PersonBox] = []

      for result in results:
         for box in result.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])

            if cls != 0:
               continue

            x1, y1, x2, y2 = box.xyxy[0].tolist()

            detections.append(
               PersonBox(
                  x1=int(x1),
                  y1=int(y1),
                  x2=int(x2),
                  y2=int(y2),
                  confidence=conf,
                  class_id=cls,
               )
            )

      return detections
