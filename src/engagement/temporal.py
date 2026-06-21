"""
Per-track and class-level engagement aggregation over a sliding window.

Uses ON_TASK_LABELS from engagement.labels as the single source of truth
for what counts as "attentive". This replaces the previous hardcoded
string comparison to "attentive" that lived in this file.

The previous away_ratio() / down_ratio() methods checked for label strings
"away" and "down" — labels that the scorer never actually emitted.
They have been replaced with a general label_ratio() method that works
for any canonical label.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from src.engagement.labels import ALL_LABELS, ON_TASK_LABELS


@dataclass
class TrackEngagementMemory:
    """Sliding-window history of engagement labels and scores for one track."""

    window_frames: int = 90
    labels: deque[str] = field(default_factory=lambda: deque(maxlen=90))
    scores: deque[float] = field(default_factory=lambda: deque(maxlen=90))

    def __post_init__(self) -> None:
        # Recreate with the configured window size (dataclass fields set before __post_init__)
        if self.labels.maxlen != self.window_frames:
            self.labels = deque(self.labels, maxlen=self.window_frames)
            self.scores = deque(self.scores, maxlen=self.window_frames)

    def update(self, label: str, score: float) -> None:
        self.labels.append(label)
        self.scores.append(score)

    # ── Ratio helpers ─────────────────────────────────────────────────────

    def attentive_ratio(self) -> float:
        """Fraction of window frames where the label is in ON_TASK_LABELS."""
        if not self.labels:
            return 0.0
        return sum(1 for lbl in self.labels if lbl in ON_TASK_LABELS) / len(self.labels)

    def label_ratio(self, label: str) -> float:
        """Fraction of window frames with a specific label."""
        if not self.labels:
            return 0.0
        return sum(1 for lbl in self.labels if lbl == label) / len(self.labels)

    def label_distribution(self) -> dict[str, float]:
        """
        Returns fraction of window for every canonical label.
        Labels not observed in the window are included with value 0.0.
        """
        if not self.labels:
            return {lbl: 0.0 for lbl in ALL_LABELS}
        n = len(self.labels)
        counts: dict[str, int] = {lbl: 0 for lbl in ALL_LABELS}
        for lbl in self.labels:
            if lbl in counts:
                counts[lbl] += 1
        return {lbl: cnt / n for lbl, cnt in counts.items()}

    # ── Score helpers ─────────────────────────────────────────────────────

    def average_score(self) -> float:
        if not self.scores:
            return 0.0
        return sum(self.scores) / len(self.scores)

    def __len__(self) -> int:
        return len(self.labels)


class EngagementTracker:
    """Manages per-track engagement memory and computes class-level aggregates."""

    def __init__(self, window_frames: int = 90) -> None:
        self._window_frames = window_frames
        self.memory: dict[int, TrackEngagementMemory] = {}

    def update(self, track_id: int, label: str, score: float) -> TrackEngagementMemory:
        if track_id not in self.memory:
            self.memory[track_id] = TrackEngagementMemory(window_frames=self._window_frames)
        state = self.memory[track_id]
        state.update(label, score)
        return state

    def remove_tracks(self, track_ids: set[int]) -> None:
        for track_id in track_ids:
            self.memory.pop(track_id, None)

    def class_average_score(self, active_track_ids: list[int]) -> float:
        values = [
            self.memory[tid].average_score()
            for tid in active_track_ids
            if tid in self.memory and self.memory[tid].scores
        ]
        return sum(values) / len(values) if values else 0.0

    def class_average_attentive_ratio(self, active_track_ids: list[int]) -> float:
        values = [
            self.memory[tid].attentive_ratio()
            for tid in active_track_ids
            if tid in self.memory and self.memory[tid].labels
        ]
        return sum(values) / len(values) if values else 0.0
