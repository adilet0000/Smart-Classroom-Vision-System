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

def draw_bbox(frame, x1: int, y1: int, x2: int, y2: int, label: str = "") -> None:
   cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

   if label:
      cv2.putText(
         frame,
         label,
         (x1, y1 - 10),
         cv2.FONT_HERSHEY_SIMPLEX,
         0.6,
         (0, 255, 0),
         2,
         cv2.LINE_AA,
      )

def draw_center_dot(frame) -> None:
   h, w = frame.shape[:2]
   cx, cy = w // 2, h // 2
   cv2.circle(frame, (cx, cy), 4, (0, 255, 255), -1)
   
   
def draw_pose(frame, x1: int, y1: int, yaw: float, pitch: float, roll: float) -> None:
   text = f"Y:{yaw:.1f} P:{pitch:.1f} R:{roll:.1f}"
   cv2.putText(
      frame,
      text,
      (x1, y1 - 30),
      cv2.FONT_HERSHEY_SIMPLEX,
      0.55,
      (255, 255, 0),
      2,
      cv2.LINE_AA,
   )
   

def draw_engagement(frame, x1: int, y1: int, label: str, score: float) -> None:
   text = f"{label} ({score:.2f})"
   cv2.putText(
      frame,
      text,
      (x1, y1 - 55),
      cv2.FONT_HERSHEY_SIMPLEX,
      0.6,
      (0, 200, 255),
      2,
      cv2.LINE_AA,
   )

def draw_engagement_stats(
   frame,
   x1: int,
   y1: int,
   attentive_ratio: float,
   avg_score: float,
) -> None:
   text = f"A:{attentive_ratio:.2f} S:{avg_score:.2f}"
   cv2.putText(
      frame,
      text,
      (x1, y1 - 80),
      cv2.FONT_HERSHEY_SIMPLEX,
      0.55,
      (255, 180, 0),
      2,
      cv2.LINE_AA,
   )
   
def draw_class_stats(frame, avg_attention: float, avg_score: float) -> None:
   text = f"CLASS A:{avg_attention:.2f} S:{avg_score:.2f}"
   cv2.putText(
      frame,
      text,
      (20, 80),
      cv2.FONT_HERSHEY_SIMPLEX,
      0.8,
      (0, 255, 255),
      2,
      cv2.LINE_AA,
   )