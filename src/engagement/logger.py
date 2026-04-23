from __future__ import annotations

import csv
import os
from datetime import datetime


class EngagementCSVLogger:
    def __init__(self, session_id: str, logs_dir: str = "logs") -> None:
        self.session_id = session_id
        os.makedirs(logs_dir, exist_ok=True)

        # Уникальное имя файла на каждую сессию
        self.file_path = os.path.join(logs_dir, f"session_{session_id}.csv")

        # Открываем файл
        self.file = open(self.file_path, mode="w", newline="", encoding="utf-8")
        self.writer = csv.writer(self.file)

        # Пишем заголовок
        self.writer.writerow([
            "timestamp",
            "session_id",
            "track_id",
            "student_code",
            "name",
            "person_confidence",
            "face_confidence",
            "label",
            "score",
            "reason",
            "attentive_ratio",
            "class_avg_attention",
            "class_avg_score",
            "yaw",
            "pitch",
            "roll",
            "eye_aspect_ratio",
            "eyes_closed",
            "gaze_x",
            "gaze_y",
            "gaze_score",
            "body_tilt",
            "body_visible",
        ])

    def log(
        self,
        track_id: int,
        student_code: str,
        name: str,
        person_confidence: float,
        face_confidence: float | None,
        label: str,
        score: float,
        reason: str,
        attentive_ratio: float,
        class_avg_attention: float,
        class_avg_score: float,
        yaw: float,
        pitch: float,
        roll: float,
        eye_aspect_ratio: float,
        eyes_closed: bool,
        gaze_x: float,
        gaze_y: float,
        gaze_score: float,
        body_tilt: float | None,
        body_visible: bool,
    ) -> None:
        timestamp = datetime.now().isoformat()

        self.writer.writerow([
            timestamp,
            self.session_id,
            track_id,
            student_code,
            name,
            person_confidence,
            "" if face_confidence is None else face_confidence,
            label,
            score,
            reason,
            attentive_ratio,
            class_avg_attention,
            class_avg_score,
            yaw,
            pitch,
            roll,
            eye_aspect_ratio,
            int(eyes_closed),
            gaze_x,
            gaze_y,
            gaze_score,
            "" if body_tilt is None else body_tilt,
            int(body_visible),
        ])

        # важно: сразу сбрасывать в файл
        self.file.flush()

    def close(self) -> None:
        self.file.close()
