from __future__ import annotations

import cv2


def draw_fps(frame, fps: float) -> None:
   cv2.putText(
      frame,
      f"FPS: {fps:.1f}",
      (20, 40),
      cv2.FONT_HERSHEY_SIMPLEX,
      1.0,
      (0, 255, 0),
      2,
      cv2.LINE_AA,
   )


def draw_center_dot(frame) -> None:
   h, w = frame.shape[:2]
   cx, cy = w // 2, h // 2
   cv2.circle(frame, (cx, cy), 4, (0, 255, 255), -1)
