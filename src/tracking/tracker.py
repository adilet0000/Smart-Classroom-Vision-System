from __future__ import annotations

from dataclasses import dataclass
from typing import List

from src.detection.face_detector import FaceBox


@dataclass
class Track:
   track_id: int
   bbox: FaceBox


class Tracker:
   """
   Заглушка трекера.
   Позже подключим ByteTrack или DeepSORT.
   """

   def __init__(self) -> None:
      self._next_id = 0

   def update(self, detections: List[FaceBox]) -> List[Track]:
      tracks = []
      for det in detections:
         tracks.append(Track(track_id=self._next_id, bbox=det))
         self._next_id += 1
      return tracks
