from __future__ import annotations

import csv
import os
from datetime import datetime


class EngagementCSVLogger:
    def __init__(self, logs_dir: str = "logs") -> None:
        os.makedirs(logs_dir, exist_ok=True)

        # Уникальное имя файла на каждую сессию
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.file_path = os.path.join(logs_dir, f"session_{timestamp}.csv")

        # Открываем файл
        self.file = open(self.file_path, mode="w", newline="", encoding="utf-8")
        self.writer = csv.writer(self.file)

        # Пишем заголовок
        self.writer.writerow([
            "timestamp",
            "track_id",
            "name",
            "label",
            "score",
            "attentive_ratio",
            "class_avg_attention",
            "class_avg_score",
        ])

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
        timestamp = datetime.now().isoformat()

        self.writer.writerow([
            timestamp,
            track_id,
            name,
            label,
            score,
            attentive_ratio,
            class_avg_attention,
            class_avg_score,
        ])

        # важно: сразу сбрасывать в файл
        self.file.flush()

    def close(self) -> None:
        self.file.close()