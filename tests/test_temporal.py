"""Temporal sliding-window aggregation: ratios, averages, eviction, cleanup."""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.engagement.labels import (  # noqa: E402
    LABEL_ATTENTIVE,
    LABEL_LOOKING_AWAY,
    LABEL_LOOKING_DOWN,
)
from src.engagement.temporal import (  # noqa: E402
    EngagementTracker,
    TrackEngagementMemory,
)


def test_attentive_ratio_basic():
    mem = TrackEngagementMemory(window_frames=4)
    mem.update(LABEL_ATTENTIVE, 0.9)
    mem.update(LABEL_ATTENTIVE, 0.8)
    mem.update(LABEL_LOOKING_AWAY, 0.4)
    mem.update(LABEL_LOOKING_DOWN, 0.3)
    assert mem.attentive_ratio() == 0.5


def test_label_ratio_specific_label():
    mem = TrackEngagementMemory(window_frames=4)
    for lbl in (LABEL_LOOKING_AWAY, LABEL_LOOKING_AWAY, LABEL_ATTENTIVE, LABEL_ATTENTIVE):
        mem.update(lbl, 0.5)
    assert mem.label_ratio(LABEL_LOOKING_AWAY) == 0.5


def test_window_eviction_respects_maxlen():
    mem = TrackEngagementMemory(window_frames=3)
    for _ in range(5):
        mem.update(LABEL_LOOKING_AWAY, 0.4)
    mem.update(LABEL_ATTENTIVE, 1.0)  # 6th push, window holds last 3
    assert len(mem) == 3
    # Last three are [away, away, attentive] -> 1/3 attentive.
    assert abs(mem.attentive_ratio() - (1 / 3)) < 1e-9


def test_average_score():
    mem = TrackEngagementMemory(window_frames=10)
    mem.update(LABEL_ATTENTIVE, 1.0)
    mem.update(LABEL_ATTENTIVE, 0.0)
    assert mem.average_score() == 0.5


def test_empty_memory_returns_zero():
    mem = TrackEngagementMemory(window_frames=5)
    assert mem.attentive_ratio() == 0.0
    assert mem.average_score() == 0.0
    assert mem.label_ratio(LABEL_ATTENTIVE) == 0.0


def test_label_distribution_sums_to_one_when_populated():
    mem = TrackEngagementMemory(window_frames=4)
    for lbl in (LABEL_ATTENTIVE, LABEL_ATTENTIVE, LABEL_LOOKING_AWAY, LABEL_LOOKING_DOWN):
        mem.update(lbl, 0.5)
    dist = mem.label_distribution()
    assert abs(sum(dist.values()) - 1.0) < 1e-9
    assert dist[LABEL_ATTENTIVE] == 0.5


def test_tracker_class_averages():
    tracker = EngagementTracker(window_frames=10)
    tracker.update(1, LABEL_ATTENTIVE, 1.0)
    tracker.update(2, LABEL_LOOKING_AWAY, 0.0)
    # Track 1 fully attentive, track 2 not -> class attentive ratio 0.5.
    assert tracker.class_average_attentive_ratio([1, 2]) == 0.5
    assert tracker.class_average_score([1, 2]) == 0.5


def test_tracker_ignores_unknown_tracks_in_average():
    tracker = EngagementTracker(window_frames=10)
    tracker.update(1, LABEL_ATTENTIVE, 1.0)
    # Track 99 has no memory; it should not drag the average down.
    assert tracker.class_average_score([1, 99]) == 1.0
    assert tracker.class_average_attentive_ratio([1, 99]) == 1.0


def test_tracker_empty_class_returns_zero():
    tracker = EngagementTracker(window_frames=10)
    assert tracker.class_average_score([]) == 0.0
    assert tracker.class_average_attentive_ratio([]) == 0.0


def test_remove_tracks_cleans_state():
    tracker = EngagementTracker(window_frames=10)
    tracker.update(1, LABEL_ATTENTIVE, 1.0)
    tracker.update(2, LABEL_ATTENTIVE, 1.0)
    tracker.remove_tracks({1})
    assert 1 not in tracker.memory
    assert 2 in tracker.memory


def test_custom_window_size_is_applied():
    mem = TrackEngagementMemory(window_frames=2)
    assert mem.labels.maxlen == 2
    assert mem.scores.maxlen == 2


if __name__ == "__main__":
    import sys

    from tests.run_tests import run_module

    sys.exit(0 if run_module(sys.modules[__name__]) else 1)
