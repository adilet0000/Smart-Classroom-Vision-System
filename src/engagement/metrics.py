from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EngagementFeatures:
   yaw: float = 0.0
   pitch: float = 0.0
   blink_rate: float = 0.0
   gaze_score: float = 0.0
