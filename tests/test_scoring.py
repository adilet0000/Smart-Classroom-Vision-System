"""Engagement scorer: label correctness and score ranges (pure python)."""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.engagement.labels import (  # noqa: E402
    LABEL_ATTENTIVE,
    LABEL_DISTRACTED,
    LABEL_EYES_CLOSED,
    LABEL_HEAD_TILTED,
    LABEL_LEANING,
    LABEL_LOOKING_AWAY,
    LABEL_LOOKING_DOWN,
    is_valid_label,
)
from src.engagement.metrics import EngagementFeatures
from src.engagement.scoring import EngagementScoringConfig, classify_engagement

CFG = EngagementScoringConfig()


def _features(**overrides) -> EngagementFeatures:
    base = dict(
        yaw=0.0,
        pitch=0.0,
        roll=0.0,
        eye_aspect_ratio=0.30,
        eyes_closed=False,
        gaze_x=0.0,
        gaze_y=0.0,
        gaze_score=1.0,
        body_tilt=None,
        body_visible=False,
    )
    base.update(overrides)
    return EngagementFeatures(**base)


def test_frontal_is_attentive_and_high_score():
    state = classify_engagement(_features(), CFG)
    assert state.label == LABEL_ATTENTIVE
    assert state.score >= CFG.attentive_score_threshold
    assert 0.0 <= state.score <= 1.0


def test_eyes_closed_label_and_capped_score():
    state = classify_engagement(_features(eyes_closed=True, eye_aspect_ratio=0.10), CFG)
    assert state.label == LABEL_EYES_CLOSED
    assert state.score <= 0.25


def test_strong_yaw_is_looking_away():
    state = classify_engagement(_features(yaw=40.0), CFG)
    assert state.label == LABEL_LOOKING_AWAY
    assert state.score <= 0.50


def test_gaze_x_offset_is_looking_away():
    state = classify_engagement(_features(gaze_x=0.4), CFG)
    assert state.label == LABEL_LOOKING_AWAY


def test_pitch_down_is_looking_down():
    state = classify_engagement(_features(pitch=-25.0), CFG)
    assert state.label == LABEL_LOOKING_DOWN
    assert state.score <= 0.45


def test_roll_tilt_is_head_tilted():
    state = classify_engagement(_features(roll=35.0), CFG)
    assert state.label == LABEL_HEAD_TILTED


def test_body_lean_is_leaning():
    state = classify_engagement(_features(body_tilt=40.0, body_visible=True), CFG)
    assert state.label == LABEL_LEANING
    assert state.score <= 0.65


def test_borderline_no_hard_rule_is_distracted():
    # yaw 15 (< 25, no hard rule) + mediocre gaze -> score below threshold.
    state = classify_engagement(_features(yaw=15.0, gaze_score=0.5, gaze_x=0.1), CFG)
    assert state.label == LABEL_DISTRACTED
    assert state.score < CFG.attentive_score_threshold


def test_eyes_closed_takes_priority_over_other_signals():
    # Even with a head turn, eyes-closed is the most severe and wins.
    state = classify_engagement(_features(eyes_closed=True, yaw=40.0, pitch=-30.0), CFG)
    assert state.label == LABEL_EYES_CLOSED


def test_missing_body_does_not_break_or_penalize_to_distracted():
    # No body data (body_visible=False) must still allow an attentive result.
    state = classify_engagement(_features(body_visible=False, body_tilt=None), CFG)
    assert state.label == LABEL_ATTENTIVE


def test_all_emitted_labels_are_canonical():
    for f in [
        _features(),
        _features(eyes_closed=True),
        _features(yaw=40.0),
        _features(pitch=-25.0),
        _features(roll=35.0),
        _features(body_tilt=40.0, body_visible=True),
        _features(yaw=15.0, gaze_score=0.5, gaze_x=0.1),
    ]:
        assert is_valid_label(classify_engagement(f, CFG).label)


def test_attentive_scores_higher_than_looking_away():
    attentive = classify_engagement(_features(), CFG).score
    away = classify_engagement(_features(yaw=40.0), CFG).score
    assert attentive > away


if __name__ == "__main__":
    import sys

    from tests.run_tests import run_module

    sys.exit(0 if run_module(sys.modules[__name__]) else 1)
