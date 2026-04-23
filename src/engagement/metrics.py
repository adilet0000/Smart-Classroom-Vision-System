from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EngagementFeatures:
   yaw: float
   pitch: float
   roll: float
   eye_aspect_ratio: float
   eyes_closed: bool
   gaze_x: float
   gaze_y: float
   gaze_score: float
   body_tilt: float | None = None
   body_visible: bool = False
