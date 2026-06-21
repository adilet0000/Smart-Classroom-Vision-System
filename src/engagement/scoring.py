"""
Engagement scoring — calibrated for real head-pose degree values.

Background:
    After fixing the head_pose.py bug (removing the erroneous *360 multiplier),
    yaw / pitch / roll are now in their true geometric degree ranges:
        yaw   ≈ ±0–45° for a seated student (0 = straight at camera)
        pitch ≈ -20° to +10°  (negative = looking down)
        roll  ≈ ±0–20°

    All thresholds below are calibrated to these corrected ranges.

Scoring philosophy:
    1. Hard-rule pass for clear negative signals (eyes closed, strong head turn).
    2. For ambiguous cases, compute a normalized score [0, 1] from weighted features.
    3. Score threshold decides attentive vs distracted.

    The weights (head 45%, gaze 30%, eye 15%, body 10%) are still heuristic —
    they should be replaced with a trained classifier once ground-truth labels
    are collected.  Document them honestly.
"""
from __future__ import annotations

from dataclasses import dataclass

from src.engagement.labels import (
    LABEL_ATTENTIVE,
    LABEL_DISTRACTED,
    LABEL_EYES_CLOSED,
    LABEL_HEAD_TILTED,
    LABEL_LEANING,
    LABEL_LOOKING_AWAY,
    LABEL_LOOKING_DOWN,
)
from src.engagement.metrics import EngagementFeatures


@dataclass
class EngagementState:
    label: str
    score: float
    reason: str = ""


@dataclass(frozen=True)
class EngagementScoringConfig:
    # ── Head pose thresholds (degrees, real units after bug fix) ──────────
    # A student is "looking away" if |yaw| > yaw_away_threshold.
    # Calibration note: at 30 cm–5 m from camera, yaw ≈ 15° = ~30% off-axis;
    # 25° is a comfortable "clearly not watching the board" cut-off.
    yaw_away_threshold: float = 25.0

    # A student is "looking down" if pitch < pitch_down_threshold (negative = down).
    # -8° is roughly "chin tilted noticeably toward chest/desk".
    pitch_down_threshold: float = -8.0

    # Roll: head tilting sideways — rarely engagement-breaking, higher threshold.
    roll_tilt_threshold: float = 20.0

    # ── Gaze thresholds (normalized iris position, 0 = centered) ─────────
    # gaze_x / gaze_y range approximately [-0.5, +0.5].
    # 0.25 ≈ iris displaced 25% of the eye width from center.
    gaze_away_threshold: float = 0.25
    gaze_down_threshold: float = 0.22

    # ── Body pose threshold (degrees) ─────────────────────────────────────
    body_tilt_threshold: float = 22.0

    # ── Composite score threshold ──────────────────────────────────────────
    # A student is labelled ATTENTIVE when all hard rules pass AND score >= this.
    attentive_score_threshold: float = 0.60


def classify_engagement(
    features: EngagementFeatures,
    cfg: EngagementScoringConfig | None = None,
) -> EngagementState:
    """
    Classify a single-frame engagement state from the feature vector.

    Returns an EngagementState with:
        label  — one of the canonical strings from engagement.labels
        score  — continuous value in [0, 1] (higher = more engaged)
        reason — short diagnostic string for logging
    """
    cfg = cfg or EngagementScoringConfig()

    # ── 1. Compute continuous sub-scores ──────────────────────────────────
    head_sc = _head_score(features.yaw, features.pitch, cfg)
    gaze_sc = features.gaze_score          # already in [0, 1] from HeadPoseEstimator
    eye_sc  = 0.15 if features.eyes_closed else 1.0
    body_sc = _body_score(features.body_tilt, features.body_visible, cfg)

    # Weighted composite — weights are heuristic; replace with classifier output
    # once ground-truth labels are available.
    score = (
        0.45 * head_sc
        + 0.30 * gaze_sc
        + 0.15 * eye_sc
        + 0.10 * body_sc
    )
    score = max(0.0, min(1.0, score))

    # ── 2. Hard-rule overrides (ordered from most-to-least severe) ────────
    # Each rule caps the score so it stays consistent with the label.

    if features.eyes_closed:
        return EngagementState(
            label=LABEL_EYES_CLOSED,
            score=min(score, 0.25),
            reason="eyes_closed",
        )

    if features.pitch < cfg.pitch_down_threshold or features.gaze_y > cfg.gaze_down_threshold:
        return EngagementState(
            label=LABEL_LOOKING_DOWN,
            score=min(score, 0.45),
            reason=f"pitch={features.pitch:.1f} gaze_y={features.gaze_y:.2f}",
        )

    if abs(features.yaw) > cfg.yaw_away_threshold or abs(features.gaze_x) > cfg.gaze_away_threshold:
        return EngagementState(
            label=LABEL_LOOKING_AWAY,
            score=min(score, 0.50),
            reason=f"yaw={features.yaw:.1f} gaze_x={features.gaze_x:.2f}",
        )

    if abs(features.roll) > cfg.roll_tilt_threshold:
        return EngagementState(
            label=LABEL_HEAD_TILTED,
            score=min(score, 0.60),
            reason=f"roll={features.roll:.1f}",
        )

    if features.body_visible and features.body_tilt is not None:
        if abs(features.body_tilt) > cfg.body_tilt_threshold:
            return EngagementState(
                label=LABEL_LEANING,
                score=min(score, 0.65),
                reason=f"body_tilt={features.body_tilt:.1f}",
            )

    # ── 3. Score-threshold decision ───────────────────────────────────────
    if score >= cfg.attentive_score_threshold:
        return EngagementState(label=LABEL_ATTENTIVE, score=score, reason="ok")

    return EngagementState(label=LABEL_DISTRACTED, score=score, reason="low_score")


# ── Helpers ───────────────────────────────────────────────────────────────

def _head_score(yaw: float, pitch: float, cfg: EngagementScoringConfig) -> float:
    """
    Penalize head rotation away from front.

    With corrected angles (real degrees):
    - yaw=0, pitch=0  → score 1.0  (perfectly frontal)
    - yaw=25, pitch=0 → score ≈ 0.0 (exactly at threshold)
    - yaw=12, pitch=0 → score ≈ 0.52
    """
    yaw_penalty = abs(yaw) / max(cfg.yaw_away_threshold, 1.0)

    # Pitch: only penalize looking *down* (negative pitch).
    # Looking slightly up (positive pitch) does not indicate disengagement.
    if pitch < cfg.pitch_down_threshold:
        pitch_penalty = (cfg.pitch_down_threshold - pitch) / max(abs(cfg.pitch_down_threshold), 1.0)
    else:
        pitch_penalty = 0.0

    return max(0.0, 1.0 - max(yaw_penalty, pitch_penalty))


def _body_score(
    body_tilt: float | None,
    body_visible: bool,
    cfg: EngagementScoringConfig,
) -> float:
    """
    Returns 0.75 when body is not visible (neutral assumption, not penalised).
    Returns [0, 1] based on torso tilt when visible.
    """
    if not body_visible or body_tilt is None:
        return 0.75
    return max(0.0, 1.0 - abs(body_tilt) / max(cfg.body_tilt_threshold, 1.0))
