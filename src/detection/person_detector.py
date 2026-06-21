from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
from ultralytics import YOLO

from src.core.geometry import clamp_box, is_valid_box
from src.core.logger import log_info, log_warn


@dataclass
class PersonBox:
   x1: int
   y1: int
   x2: int
   y2: int
   confidence: float
   class_id: int = 0


def _resolve_device(device: str) -> str:
   """
   Resolve the configured device string into a concrete device for Ultralytics.

   "auto" picks the best available accelerator:
     - Apple Silicon (M1–M4): "mps"
     - NVIDIA GPU:            "cuda"
     - otherwise:             "cpu"
   Any explicit value ("cpu" / "mps" / "cuda" / "0") is passed through unchanged.
   """
   if device != "auto":
      return device
   try:
      import torch

      if torch.backends.mps.is_available():
         return "mps"
      if torch.cuda.is_available():
         return "cuda"
   except Exception as exc:  # torch missing or backend probe failed
      log_warn(f"Device auto-detection failed ({exc}); falling back to CPU.")
   return "cpu"


class PersonDetector:
   """
   Detector of people in the frame.
   COCO class 0 = person.

   Garbage detections (tiny or out-of-frame boxes) are filtered out so the
   tracker and downstream crops only ever see usable boxes.
   """

   def __init__(
      self,
      model_name: str = "yolov8n.pt",
      image_size: int = 640,
      confidence_threshold: float = 0.25,
      device: str = "auto",
      min_box_width: int = 20,
      min_box_height: int = 40,
      min_box_area: int = 0,
   ) -> None:
      self.model = YOLO(model_name)
      self.image_size = image_size
      self.confidence_threshold = confidence_threshold
      self.device = _resolve_device(device)
      self.min_box_width = min_box_width
      self.min_box_height = min_box_height
      self.min_box_area = min_box_area
      log_info(f"PersonDetector: model={model_name} imgsz={image_size} device={self.device}")

   def detect(self, frame: np.ndarray) -> List[PersonBox]:
      h, w = frame.shape[:2]

      results = self.model(
         frame,
         verbose=False,
         imgsz=self.image_size,
         conf=self.confidence_threshold,
         device=self.device,
         classes=[0],   # person only — skip non-person boxes inside YOLO
      )

      detections: List[PersonBox] = []

      for result in results:
         for box in result.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])

            if cls != 0:
               continue

            raw = box.xyxy[0].tolist()
            x1, y1, x2, y2 = clamp_box(
               int(raw[0]), int(raw[1]), int(raw[2]), int(raw[3]), w, h
            )

            # Drop tiny / degenerate boxes that yield useless crops.
            if not is_valid_box(
               x1, y1, x2, y2,
               min_width=self.min_box_width,
               min_height=self.min_box_height,
               min_area=self.min_box_area,
            ):
               continue

            detections.append(
               PersonBox(
                  x1=x1,
                  y1=y1,
                  x2=x2,
                  y2=y2,
                  confidence=conf,
                  class_id=cls,
               )
            )

      return detections
