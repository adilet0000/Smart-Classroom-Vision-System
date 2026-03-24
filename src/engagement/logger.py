from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path


class EngagementCSVLogger:
   def __init__(self, log_path: str = "data/engagement_log.csv") -> None:
      self.log_path = Path(log_path)
      self.log_path.parent.mkdir(parents=True, exist_ok=True)

      self._ensure_file()

   def _ensure_file(self) -> None:
      if self.log_path.exists():
         return

      with self.log_path.open("w", newline="", encoding="utf-8") as f:
         writer = csv.writer(f)
         writer.writerow(
            [
               "timestamp",
               "track_id",
               "name",
               "label",
               "score",
               "attentive_ratio",
               "class_avg_attention",
               "class_avg_score",
            ]
         )

   def log(
      self,
      track_id: int,
      name: str,
      label: str,
      score: float,
      attentive_ratio: float,
      class_avg_attention: float,
      class_avg_score: float,
   ) -> None:
      timestamp = datetime.now().isoformat(timespec="seconds")

      with self.log_path.open("a", newline="", encoding="utf-8") as f:
         writer = csv.writer(f)
         writer.writerow(
            [
               timestamp,
               track_id,
               name,
               label,
               f"{score:.4f}",
               f"{attentive_ratio:.4f}",
               f"{class_avg_attention:.4f}",
               f"{class_avg_score:.4f}",
            ]
         )