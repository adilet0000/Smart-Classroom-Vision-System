from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import supervision as sv

from src.detection.person_detector import PersonBox


@dataclass
class Track:
   track_id: int
   bbox: PersonBox


class Tracker:
   def __init__(
      self,
      fps: int = 30,
      track_activation_threshold: float = 0.25,
      lost_track_buffer: int = 30,
      minimum_matching_threshold: float = 0.8,
      minimum_consecutive_frames: int = 2,
   ) -> None:
      # All tuning knobs are config-driven so tracking can be stabilised per
      # scene without code changes:
      #   track_activation_threshold  — higher = fewer spurious new tracks
      #   lost_track_buffer           — frames an occluded track survives before
      #                                 a re-appearance gets a fresh id
      #   minimum_matching_threshold  — IoU gate for associating detections
      #   minimum_consecutive_frames  — frames before a track is confirmed
      #                                 (suppresses one-frame id flicker)
      self.tracker = sv.ByteTrack(
         track_activation_threshold=track_activation_threshold,
         lost_track_buffer=lost_track_buffer,
         minimum_matching_threshold=minimum_matching_threshold,
         frame_rate=fps,
         minimum_consecutive_frames=minimum_consecutive_frames,
      )

   def update(self, detections: List[PersonBox]) -> List[Track]:
      if not detections:
         tracked = self.tracker.update_with_detections(sv.Detections.empty())
         return []

      xyxy = np.array(
         [[d.x1, d.y1, d.x2, d.y2] for d in detections],
         dtype=np.float32,
      )
      confidence = np.array(
         [d.confidence for d in detections],
         dtype=np.float32,
      )
      class_id = np.array(
         [d.class_id for d in detections],
         dtype=int,
      )

      sv_detections = sv.Detections(
         xyxy=xyxy,
         confidence=confidence,
         class_id=class_id,
      )

      tracked = self.tracker.update_with_detections(sv_detections)

      output: List[Track] = []

      if tracked.tracker_id is None:
         return output

      for i in range(len(tracked.xyxy)):
         x1, y1, x2, y2 = tracked.xyxy[i].astype(int).tolist()
         conf = float(tracked.confidence[i]) if tracked.confidence is not None else 0.0
         tid = int(tracked.tracker_id[i])

         # В новых версиях track_id может быть -1,
         # если трек еще не подтвержден minimum_consecutive_frames
         if tid < 0:
            continue

         output.append(
            Track(
               track_id=tid,
               bbox=PersonBox(
                  x1=x1,
                  y1=y1,
                  x2=x2,
                  y2=y2,
                  confidence=conf,
                  class_id=0,
               ),
            )
         )

      return output
