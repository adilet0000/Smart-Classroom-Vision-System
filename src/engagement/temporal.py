from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field


@dataclass
class TrackEngagementMemory:
   labels: deque[str] = field(default_factory=lambda: deque(maxlen=75))
   scores: deque[float] = field(default_factory=lambda: deque(maxlen=75))

   def update(self, label: str, score: float) -> None:
      self.labels.append(label)
      self.scores.append(score)

   def attentive_ratio(self) -> float:
      if not self.labels:
         return 0.0
      return sum(1 for x in self.labels if x == "attentive") / len(self.labels)

   def away_ratio(self) -> float:
      if not self.labels:
         return 0.0
      return sum(1 for x in self.labels if x == "away") / len(self.labels)

   def down_ratio(self) -> float:
      if not self.labels:
         return 0.0
      return sum(1 for x in self.labels if x == "down") / len(self.labels)

   def average_score(self) -> float:
      if not self.scores:
         return 0.0
      return sum(self.scores) / len(self.scores)


class EngagementTracker:
   def __init__(self) -> None:
      self.memory: dict[int, TrackEngagementMemory] = {}

   def update(self, track_id: int, label: str, score: float) -> TrackEngagementMemory:
      state = self.memory.setdefault(track_id, TrackEngagementMemory())
      state.update(label, score)
      return state

   def remove_tracks(self, track_ids: set[int]) -> None:
      for track_id in track_ids:
         self.memory.pop(track_id, None)
   
   def class_average_score(self, active_track_ids: list[int]) -> float:
      if not active_track_ids:
         return 0.0

      values = []
      for track_id in active_track_ids:
         state = self.memory.get(track_id)
         if state is not None and state.scores:
            values.append(state.average_score())

      if not values:
         return 0.0

      return sum(values) / len(values)

   def class_average_attentive_ratio(self, active_track_ids: list[int]) -> float:
      if not active_track_ids:
         return 0.0

      values = []
      for track_id in active_track_ids:
         state = self.memory.get(track_id)
         if state is not None and state.labels:
            values.append(state.attentive_ratio())

      if not values:
         return 0.0

      return sum(values) / len(values)
