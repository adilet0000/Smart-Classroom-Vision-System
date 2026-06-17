from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EngagementState:
   label: str
   score: float


def classify_engagement(yaw: float, pitch: float) -> EngagementState:
   abs_yaw = abs(yaw)

   if pitch < -6:
      return EngagementState(label="down", score=0.3)

   if abs_yaw > 15:
      return EngagementState(label="away", score=0.4)

   return EngagementState(label="attentive", score=1.0)