from __future__ import annotations

from datetime import datetime

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
    draw_feature_stats,
    draw_class_stats
)

from src.detection.person_detector import PersonDetector
from src.tracking.tracker import Tracker
from src.recognition.face_database import FaceDatabase
from src.recognition.face_embedder import FaceEmbedder
from src.recognition.attendance_logic import AttendanceManager

from src.features.head_pose import HeadPoseEstimator
from src.features.body_pose import BodyPoseEstimator

from src.engagement.metrics import EngagementFeatures
from src.engagement.scoring import EngagementScoringConfig, classify_engagement
from src.engagement.temporal import EngagementTracker
from src.engagement.logger import EngagementCSVLogger


def run() -> None:
    cfg = load_config()

    log_info(f"Video source: {cfg.video.source}")
    log_info(f"Resolution: {cfg.video.width}x{cfg.video.height} | target_fps={cfg.video.target_fps}")

    session_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_info(f"Session ID: {session_id}")

    cap = VideoCapture(
        source=cfg.video.source,
        width=cfg.video.width,
        height=cfg.video.height,
        target_fps=cfg.video.target_fps,
        mirror=cfg.video.mirror,
    )

    detector = PersonDetector(
        model_name=cfg.detection.model_name,
        image_size=cfg.detection.image_size,
        confidence_threshold=cfg.detection.confidence_threshold,
        device=cfg.detection.device,
    )
    tracker = Tracker(fps=cfg.video.target_fps)

    face_db = FaceDatabase()
    face_embedder = FaceEmbedder()
    attendance_manager = AttendanceManager(
        similarity_threshold=cfg.recognition.similarity_threshold,
        confirm_matches=cfg.recognition.confirm_matches,
    )
    head_pose_estimator = HeadPoseEstimator(
        eye_closed_threshold=cfg.engagement.eye_closed_threshold,
        gaze_center_tolerance_x=cfg.engagement.gaze_center_tolerance_x,
        gaze_center_tolerance_y=cfg.engagement.gaze_center_tolerance_y,
    )
    body_pose_estimator = BodyPoseEstimator()
    scoring_config = EngagementScoringConfig(
        attentive_score_threshold=cfg.engagement.attentive_score_threshold,
        yaw_away_threshold=cfg.engagement.yaw_away_threshold,
        pitch_down_threshold=cfg.engagement.pitch_down_threshold,
        roll_tilt_threshold=cfg.engagement.roll_tilt_threshold,
        gaze_away_threshold=cfg.engagement.gaze_away_threshold,
        gaze_down_threshold=cfg.engagement.gaze_down_threshold,
        body_tilt_threshold=cfg.engagement.body_tilt_threshold,
    )
    engagement_tracker = EngagementTracker()
    engagement_logger = EngagementCSVLogger(
        session_id=session_id,
        logs_dir=cfg.logging.logs_dir,
    )

    frame_index = 0

    cv2.namedWindow(cfg.app.window_name, cv2.WINDOW_NORMAL)

    track_identity: dict[int, str] = {}
    track_student_code: dict[int, str] = {}
    track_face_confidence: dict[int, float] = {}
    missing_track_frames: dict[int, int] = {}

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            log_warn("Frame read failed.")
            break

        detections = detector.detect(frame)
        tracks = tracker.update(detections)
        active_track_ids = {track.track_id for track in tracks}
        known_track_ids = (
            set(track_identity)
            | set(track_student_code)
            | set(track_face_confidence)
            | set(missing_track_frames)
            | set(attendance_manager.track_states)
            | set(engagement_tracker.memory)
        )
        expired_track_ids: set[int] = set()

        for track_id in known_track_ids:
            if track_id in active_track_ids:
                missing_track_frames.pop(track_id, None)
                continue

            missing_track_frames[track_id] = missing_track_frames.get(track_id, 0) + 1

            if missing_track_frames[track_id] >= cfg.tracking.cleanup_after_missing_frames:
                missing_track_frames.pop(track_id, None)
                track_identity.pop(track_id, None)
                track_student_code.pop(track_id, None)
                track_face_confidence.pop(track_id, None)
                expired_track_ids.add(track_id)

        attendance_manager.remove_tracks(expired_track_ids)
        engagement_tracker.remove_tracks(expired_track_ids)

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
            if (
                track.track_id not in track_identity
                and frame_index % cfg.recognition.interval_frames == 0
            ):
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
                            session_id=session_id,
                        )

                        if student_code is not None:
                            name = face_db.get_student_name(student_code)

                            # защита от мусорных значений
                            if name and name != "name":
                                track_identity[track.track_id] = name
                                track_student_code[track.track_id] = student_code
                                track_face_confidence[track.track_id] = face_result.det_score

            # обновляем label если уже распознан
            if track.track_id in track_identity:
                label = track_identity[track.track_id]

            draw_bbox(frame, box.x1, box.y1, box.x2, box.y2, label)

            full_person_crop = frame[
                max(0, box.y1):min(frame.shape[0], box.y2),
                max(0, box.x1):min(frame.shape[1], box.x2),
            ]
            person_crop = face_embedder.crop_person_region(
                frame,
                (box.x1, box.y1, box.x2, box.y2),
            )

            if person_crop.size != 0:
                pose = head_pose_estimator.estimate(person_crop)

                if pose is not None:
                    draw_pose(frame, box.x1, box.y1, pose.yaw, pose.pitch, pose.roll)

                    body_pose = None
                    if full_person_crop.size != 0:
                        body_pose = body_pose_estimator.estimate(full_person_crop)

                    features = EngagementFeatures(
                        yaw=pose.yaw,
                        pitch=pose.pitch,
                        roll=pose.roll,
                        eye_aspect_ratio=pose.eye_aspect_ratio,
                        eyes_closed=pose.eyes_closed,
                        gaze_x=pose.gaze_x,
                        gaze_y=pose.gaze_y,
                        gaze_score=pose.gaze_score,
                        body_tilt=body_pose.torso_tilt if body_pose is not None and body_pose.visible else None,
                        body_visible=body_pose.visible if body_pose is not None else False,
                    )

                    engagement = classify_engagement(features, scoring_config)

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
                    draw_feature_stats(
                        frame,
                        box.x1,
                        box.y1,
                        features.eye_aspect_ratio,
                        features.gaze_score,
                        features.body_tilt,
                    )

                    # ✅ ЛОГИРУЕМ ТОЛЬКО ЕСЛИ УЧЕНИК РАСПОЗНАН
                    name = track_identity.get(track.track_id)
                    student_code = track_student_code.get(track.track_id)

                    if name is not None and student_code is not None:
                        frame_records.append({
                            "track_id": track.track_id,
                            "student_code": student_code,
                            "name": name,
                            "person_confidence": box.confidence,
                            "face_confidence": track_face_confidence.get(track.track_id),
                            "label": engagement.label,
                            "score": engagement.score,
                            "reason": engagement.reason,
                            "attentive_ratio": memory.attentive_ratio(),
                            "features": features,
                        })

        active_track_id_list = list(active_track_ids)

        class_avg_attention = engagement_tracker.class_average_attentive_ratio(active_track_id_list)
        class_avg_score = engagement_tracker.class_average_score(active_track_id_list)

        draw_class_stats(frame, class_avg_attention, class_avg_score)

        # логируем периодически
        if frame_index % cfg.logging.interval_frames == 0:
            for record in frame_records:
                if record["name"] is None:
                    continue

                engagement_logger.log(
                    track_id=record["track_id"],
                    student_code=record["student_code"],
                    name=record["name"],
                    person_confidence=record["person_confidence"],
                    face_confidence=record["face_confidence"],
                    label=record["label"],
                    score=record["score"],
                    reason=record["reason"],
                    attentive_ratio=record["attentive_ratio"],
                    class_avg_attention=class_avg_attention,
                    class_avg_score=class_avg_score,
                    yaw=record["features"].yaw,
                    pitch=record["features"].pitch,
                    roll=record["features"].roll,
                    eye_aspect_ratio=record["features"].eye_aspect_ratio,
                    eyes_closed=record["features"].eyes_closed,
                    gaze_x=record["features"].gaze_x,
                    gaze_y=record["features"].gaze_y,
                    gaze_score=record["features"].gaze_score,
                    body_tilt=record["features"].body_tilt,
                    body_visible=record["features"].body_visible,
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

    # Использование:
    # .venv/bin/python -m scripts.analyze_engagement logs/session_2026-03-31_17-09-18.csv

    # Можно настроить размер временного окна:
    # .venv/bin/python -m scripts.analyze_engagement logs/session_2026-03-31_17-09-18.csv --bin-seconds 10
