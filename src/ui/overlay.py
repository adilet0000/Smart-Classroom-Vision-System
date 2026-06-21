"""
OpenCV overlay drawing utilities.

Each function draws directly onto the provided frame (in-place).
"""
from __future__ import annotations

import cv2
import numpy as np

from src.engagement.labels import LABEL_COLOURS


# ── Per-person overlays ────────────────────────────────────────────────────

def draw_fps(frame: np.ndarray, fps: float) -> None:
   cv2.putText(
      frame, f"FPS: {fps:.1f}",
      (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
      (0, 255, 0), 2, cv2.LINE_AA,
   )


def draw_center_dot(frame: np.ndarray) -> None:
   h, w = frame.shape[:2]
   cv2.circle(frame, (w // 2, h // 2), 4, (0, 255, 255), -1)


def draw_bbox(
   frame: np.ndarray,
   x1: int, y1: int, x2: int, y2: int,
   label: str = "",
) -> None:
   cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
   if label:
      cv2.putText(
         frame, label,
         (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
         (0, 255, 0), 2, cv2.LINE_AA,
      )


def draw_pose(
   frame: np.ndarray,
   x1: int, y1: int,
   yaw: float, pitch: float, roll: float,
) -> None:
   text = f"Y:{yaw:+.1f} P:{pitch:+.1f} R:{roll:+.1f}"
   cv2.putText(
      frame, text,
      (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.50,
      (255, 255, 0), 1, cv2.LINE_AA,
   )


def draw_engagement(
   frame: np.ndarray,
   x1: int, y1: int,
   label: str, score: float,
) -> None:
   colour = LABEL_COLOURS.get(label, (200, 200, 200))
   text = f"{label} ({score:.2f})"
   cv2.putText(
      frame, text,
      (x1, y1 - 52), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
      colour, 2, cv2.LINE_AA,
   )


def draw_engagement_stats(
   frame: np.ndarray,
   x1: int, y1: int,
   attentive_ratio: float,
   avg_score: float,
) -> None:
   text = f"A:{attentive_ratio:.2f} S:{avg_score:.2f}"
   cv2.putText(
      frame, text,
      (x1, y1 - 74), cv2.FONT_HERSHEY_SIMPLEX, 0.50,
      (255, 180, 0), 1, cv2.LINE_AA,
   )


def draw_feature_stats(
   frame: np.ndarray,
   x1: int, y1: int,
   ear: float,
   gaze_score: float,
   body_tilt: float | None,
) -> None:
   body_text = "--" if body_tilt is None else f"{body_tilt:.0f}"
   text = f"EAR:{ear:.2f} G:{gaze_score:.2f} B:{body_text}"
   cv2.putText(
      frame, text,
      (x1, y1 - 94), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
      (180, 220, 255), 1, cv2.LINE_AA,
   )


def draw_class_stats(
   frame: np.ndarray,
   avg_attention: float,
   avg_score: float,
) -> None:
   text = f"CLASS  A:{avg_attention:.2f}  S:{avg_score:.2f}"
   cv2.putText(
      frame, text,
      (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
      (0, 255, 255), 2, cv2.LINE_AA,
   )


# ── Debug panel ───────────────────────────────────────────────────────────

def draw_debug_panel(
   frame: np.ndarray,
   fps: float,
   active_tracks: int,
   recognized_count: int,
   class_avg_score: float,
   class_attentive_ratio: float,
) -> None:
   """
   Compact debug info panel drawn in the bottom-left corner.
   Enabled via config.debug.show_debug_panel.

   Displays:
     FPS            — current pipeline frame rate
     Tracks         — number of active ByteTrack tracks this frame
     Recognized     — number of identity-confirmed tracks this session
     Cls Score      — class-average engagement score [0, 1]
     Cls Attentive  — fraction of class currently labelled attentive [0, 1]
   """
   lines = [
      f"FPS:           {fps:.1f}",
      f"Tracks:        {active_tracks}",
      f"Recognized:    {recognized_count}",
      f"Cls Score:     {class_avg_score:.2f}",
      f"Cls Attentive: {class_attentive_ratio:.2f}",
   ]

   h, w = frame.shape[:2]
   line_h = 22
   padding = 10
   panel_h = len(lines) * line_h + padding * 2
   panel_w = 240

   x0 = 10
   y0 = h - panel_h - 10

   # Semi-transparent dark background
   overlay = frame.copy()
   cv2.rectangle(overlay, (x0, y0), (x0 + panel_w, y0 + panel_h), (20, 20, 20), -1)
   cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

   # Text rows
   for i, line in enumerate(lines):
      cy = y0 + padding + (i + 1) * line_h - 4
      cv2.putText(
         frame, line,
         (x0 + 8, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.48,
         (210, 230, 210), 1, cv2.LINE_AA,
      )
