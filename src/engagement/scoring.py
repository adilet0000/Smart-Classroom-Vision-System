from __future__ import annotations

from src.engagement.metrics import EngagementFeatures


def compute_engagement_score(features: EngagementFeatures) -> float:
   """
   Простая формула-заглушка.
   Потом заменим на реальную.
   """

   attentive = max(0.0, 1.0 - abs(features.yaw) / 90.0)
   drowsy_penalty = features.blink_rate * 0.1

   score = attentive - drowsy_penalty
   return max(0.0, min(1.0, score))
