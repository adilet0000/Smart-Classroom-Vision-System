from __future__ import annotations

import time
from typing import Any, Optional

import cv2


class VideoCapture:
   def __init__(self, source: Any, width: int, height: int, target_fps: int, mirror: bool) -> None:
      self.source = source
      self.width = width
      self.height = height
      self.target_fps = max(1, target_fps)
      self.mirror = mirror

      self.cap = cv2.VideoCapture(source)
      self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
      self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

      self._last_ts = time.time()
      self._fps = 0.0

   @property
   def fps(self) -> float:
      return self._fps

   def read(self) -> tuple[bool, Optional[any]]:
      ok, frame = self.cap.read()
      if not ok or frame is None:
         return False, None

      if self.mirror:
         frame = cv2.flip(frame, 1)

      # FPS estimation
      now = time.time()
      dt = now - self._last_ts
      self._last_ts = now
      if dt > 0:
         instant = 1.0 / dt
         self._fps = 0.9 * self._fps + 0.1 * instant if self._fps > 0 else instant

      return True, frame

   def release(self) -> None:
      self.cap.release()
