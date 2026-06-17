from __future__ import annotations

import cv2

from src.core.config import load_config
from src.core.logger import log_info, log_warn
from src.video.capture import VideoCapture
from src.ui.overlay import (
    draw_fps,
    draw_center_dot,
    draw_bbox,
    draw_pose,
    draw_engagement,
    draw_engagement_stats,
    draw_class_stats
)

from src.detection.face_detector import FaceDetector
from src.tracking.tracker import Tracker
from src.recognition.face_database import FaceDatabase
from src.recognition.face_embedder import FaceEmbedder
from src.recognition.attendance_logic import AttendanceManager

from src.features.head_pose import HeadPoseEstimator

from src.engagement.scoring import classify_engagement
from src.engagement.temporal import EngagementTracker
from src.engagement.logger import EngagementCSVLogger


def run() -> None:
    cfg = load_config()

    log_info(f"Video source: {cfg.video.source}")
    log_info(f"Resolution: {cfg.video.width}x{cfg.video.height} | target_fps={cfg.video.target_fps}")

    cap = VideoCapture(
        source=cfg.video.source,
        width=cfg.video.width,
        height=cfg.video.height,
        target_fps=cfg.video.target_fps,
        mirror=cfg.video.mirror,
    )

    detector = FaceDetector()
    tracker = Tracker(fps=cfg.video.target_fps)

    face_db = FaceDatabase()
    face_embedder = FaceEmbedder()
    attendance_manager = AttendanceManager()
    head_pose_estimator = HeadPoseEstimator()
    engagement_tracker = EngagementTracker()
    engagement_logger = EngagementCSVLogger()

    frame_index = 0

    cv2.namedWindow(cfg.app.window_name, cv2.WINDOW_NORMAL)

    track_identity = {}

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            log_warn("Frame read failed.")
            break

        detections = detector.detect(frame)
        tracks = tracker.update(detections)

        frame_index += 1

        frame_records = []

        for track in tracks:
            box = track.bbox

            # Базовый label для отображения
            label = track_identity.get(
                track.track_id,
                f"track:{track.track_id} {box.confidence:.2f}"
            )

            # --- Face recognition ---
            if track.track_id not in track_identity and frame_index % 10 == 0:
                person_crop = face_embedder.crop_person_region(
                    frame,
                    (box.x1, box.y1, box.x2, box.y2),
                )

                if person_crop.size != 0:
                    face_result = face_embedder.get_embedding(person_crop)

                    if face_result is not None:
                        student_code = attendance_manager.recognize_embedding(
                            track_id=track.track_id,
                            embedding=face_result.embedding,
                            db=face_db,
                            embedder=face_embedder,
                        )

                        if student_code is not None:
                            name = face_db.get_student_name(student_code)

                            # защита от мусорных значений
                            if name and name != "name":
                                track_identity[track.track_id] = name

            # обновляем label если уже распознан
            if track.track_id in track_identity:
                label = track_identity[track.track_id]

            draw_bbox(frame, box.x1, box.y1, box.x2, box.y2, label)

            person_crop = face_embedder.crop_person_region(
                frame,
                (box.x1, box.y1, box.x2, box.y2),
            )

            if person_crop.size != 0:
                pose = head_pose_estimator.estimate(person_crop)

                if pose is not None:
                    draw_pose(frame, box.x1, box.y1, pose.yaw, pose.pitch, pose.roll)

                    engagement = classify_engagement(pose.yaw, pose.pitch)

                    draw_engagement(frame, box.x1, box.y1, engagement.label, engagement.score)

                    memory = engagement_tracker.update(
                        track.track_id,
                        engagement.label,
                        engagement.score,
                    )

                    draw_engagement_stats(
                        frame,
                        box.x1,
                        box.y1,
                        memory.attentive_ratio(),
                        memory.average_score(),
                    )

                    # ✅ ЛОГИРУЕМ ТОЛЬКО ЕСЛИ УЧЕНИК РАСПОЗНАН
                    name = track_identity.get(track.track_id)

                    if name is not None:
                        frame_records.append({
                            "track_id": track.track_id,
                            "name": name,
                            "label": engagement.label,
                            "score": engagement.score,
                            "attentive_ratio": memory.attentive_ratio(),
                        })

        active_track_ids = [track.track_id for track in tracks]

        class_avg_attention = engagement_tracker.class_average_attentive_ratio(active_track_ids)
        class_avg_score = engagement_tracker.class_average_score(active_track_ids)

        draw_class_stats(frame, class_avg_attention, class_avg_score)

        # логируем периодически
        if frame_index % 30 == 0:
            for record in frame_records:
                if record["name"] is None:
                    continue

                engagement_logger.log(
                    track_id=record["track_id"],
                    name=record["name"],
                    label=record["label"],
                    score=record["score"],
                    attentive_ratio=record["attentive_ratio"],
                    class_avg_attention=class_avg_attention,
                    class_avg_score=class_avg_score,
                )

        if cfg.debug.draw_center_dot:
            draw_center_dot(frame)

        if cfg.app.show_fps:
            draw_fps(frame, cap.fps)

        cv2.imshow(cfg.app.window_name, frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            break

    cap.release()
    cv2.destroyAllWindows()
    engagement_logger.close()

    log_info("Stopped.")


if __name__ == "__main__":
    run()
    
    # python -m scripts.analyze_engagement logs/session_2026-03-31_17-09-18.csv - сохранение отчета в папке reports/engagement_report_2026-03-31_17-09-18.png
    # python -m src.app.main
    # python -m scripts.register_face --image photos/me.jpg --code CS21_001 --name "Adilet Orozaliev"