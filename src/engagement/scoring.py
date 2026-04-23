from __future__ import annotations

from dataclasses import dataclass

from src.engagement.metrics import EngagementFeatures


@dataclass
class EngagementState:
   label: str
   score: float
   reason: str = ""


@dataclass(frozen=True)
class EngagementScoringConfig:
   attentive_score_threshold: float = 0.68
   yaw_away_threshold: float = 25.0
   pitch_down_threshold: float = -12.0
   roll_tilt_threshold: float = 25.0
   gaze_away_threshold: float = 0.24
   gaze_down_threshold: float = 0.20
   body_tilt_threshold: float = 22.0


def classify_engagement(
   features: EngagementFeatures,
   cfg: EngagementScoringConfig | None = None,
) -> EngagementState:
   cfg = cfg or EngagementScoringConfig()

   head_score = _head_score(features.yaw, features.pitch, cfg)
   gaze_score = features.gaze_score
   eye_score = 0.15 if features.eyes_closed else 1.0
   body_score = _body_score(features.body_tilt, features.body_visible, cfg)

   score = (
      0.42 * head_score
      + 0.28 * gaze_score
      + 0.18 * eye_score
      + 0.12 * body_score
   )

   if features.eyes_closed:
      return EngagementState(label="eyes_closed", score=min(score, 0.35), reason="eyes")

   if features.pitch < cfg.pitch_down_threshold or features.gaze_y > cfg.gaze_down_threshold:
      return EngagementState(label="looking_down", score=min(score, 0.55), reason="down")

   if abs(features.yaw) > cfg.yaw_away_threshold or abs(features.gaze_x) > cfg.gaze_away_threshold:
      return EngagementState(label="looking_away", score=min(score, 0.6), reason="away")

   if abs(features.roll) > cfg.roll_tilt_threshold:
      return EngagementState(label="head_tilted", score=min(score, 0.65), reason="roll")

   if features.body_visible and features.body_tilt is not None:
      if abs(features.body_tilt) > cfg.body_tilt_threshold:
         return EngagementState(label="leaning", score=min(score, 0.7), reason="body")

   if score >= cfg.attentive_score_threshold:
      return EngagementState(label="attentive", score=score, reason="features")

   return EngagementState(label="distracted", score=score, reason="low_score")


def _head_score(yaw: float, pitch: float, cfg: EngagementScoringConfig) -> float:
   yaw_penalty = abs(yaw) / max(cfg.yaw_away_threshold, 1.0)
   down_penalty = max(0.0, cfg.pitch_down_threshold - pitch) / max(abs(cfg.pitch_down_threshold), 1.0)
   return max(0.0, 1.0 - max(yaw_penalty, down_penalty))


def _body_score(
   body_tilt: float | None,
   body_visible: bool,
   cfg: EngagementScoringConfig,
) -> float:
   if not body_visible or body_tilt is None:
      return 0.75

   return max(0.0, 1.0 - abs(body_tilt) / max(cfg.body_tilt_threshold, 1.0))
