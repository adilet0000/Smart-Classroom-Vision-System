"""
Smart Classroom — main pipeline entry point.

Pipeline stages (executed per frame):
  1. capture        — read frame from camera / video
  2. detection      — YOLO person detection
  3. tracking       — ByteTrack multi-person tracking + stale-track cleanup
  4. recognition    — InsightFace embedding → attendance confirmation
  5. pose           — MediaPipe FaceMesh head pose + body pose (one crop per track)
  6. engagement     — rule-based scoring
  7. temporal       — sliding-window aggregation
  8. rendering      — draw bounding boxes / labels / debug panel
  9. logging        — write CSV row every N frames
"""
from __future__ import annotations

from datetime import datetime
from typing import NamedTuple

import cv2
import numpy as np

from src.core.config import load_config
from src.core.logger import log_info, log_warn

from src.video.capture import VideoCapture

from src.detection.person_detector import PersonDetector
from src.tracking.tracker import Tracker

from src.recognition.face_database import FaceDatabase
from src.recognition.face_embedder import FaceEmbedder
from src.recognition.embedding_cache import EmbeddingCache
from src.recognition.attendance_logic import AttendanceManager

from src.features.head_pose import HeadPoseEstimator
from src.features.body_pose import BodyPoseEstimator

from src.engagement.metrics import EngagementFeatures
from src.engagement.scoring import EngagementScoringConfig, classify_engagement
from src.engagement.temporal import EngagementTracker
from src.engagement.logger import EngagementCSVLogger

from src.ui.overlay import (
    draw_fps,
    draw_center_dot,
    draw_bbox,
    draw_pose,
    draw_engagement,
    draw_engagement_stats,
    draw_feature_stats,
    draw_class_stats,
    draw_debug_panel,
)


# ── Per-track identity state ───────────────────────────────────────────────

class TrackState(NamedTuple):
    """All identity info for a confirmed track, stored in one place."""
    name: str
    student_code: str
    face_confidence: float


# ── Main pipeline ──────────────────────────────────────────────────────────

def run() -> None:
    cfg = load_config()

    log_info(f"Video source: {cfg.video.source}")
    log_info(f"Resolution: {cfg.video.width}x{cfg.video.height} | target_fps={cfg.video.target_fps}")

    session_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_info(f"Session ID: {session_id}")

    # ── Initialise components ──────────────────────────────────────────────

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
        min_box_width=cfg.detection.min_box_width,
        min_box_height=cfg.detection.min_box_height,
        min_box_area=cfg.detection.min_box_area,
    )
    tracker = Tracker(
        fps=cfg.video.target_fps,
        track_activation_threshold=cfg.tracking.track_activation_threshold,
        lost_track_buffer=cfg.tracking.lost_track_buffer,
        minimum_matching_threshold=cfg.tracking.minimum_matching_threshold,
        minimum_consecutive_frames=cfg.tracking.minimum_consecutive_frames,
    )

    face_db      = FaceDatabase()
    face_embedder = FaceEmbedder()

    # ── T3: preload all embeddings once at startup ─────────────────────────
    emb_cache = EmbeddingCache.load(face_db)
    log_info(
        f"Embedding cache loaded: {len(emb_cache)} vectors "
        f"({emb_cache.unique_students()} students)"
    )

    attendance_manager = AttendanceManager(
        similarity_threshold=cfg.recognition.similarity_threshold,
        confirm_matches=cfg.recognition.confirm_matches,
    )

    head_pose_estimator = HeadPoseEstimator(
        eye_closed_threshold=cfg.engagement.eye_closed_threshold,
        gaze_center_tolerance_x=cfg.engagement.gaze_center_tolerance_x,
        gaze_center_tolerance_y=cfg.engagement.gaze_center_tolerance_y,
    )
    body_pose_estimator = BodyPoseEstimator() if cfg.engagement.use_body_pose else None

    scoring_config = EngagementScoringConfig(
        attentive_score_threshold=cfg.engagement.attentive_score_threshold,
        yaw_away_threshold=cfg.engagement.yaw_away_threshold,
        pitch_down_threshold=cfg.engagement.pitch_down_threshold,
        roll_tilt_threshold=cfg.engagement.roll_tilt_threshold,
        gaze_away_threshold=cfg.engagement.gaze_away_threshold,
        gaze_down_threshold=cfg.engagement.gaze_down_threshold,
        body_tilt_threshold=cfg.engagement.body_tilt_threshold,
    )

    engagement_tracker = EngagementTracker(window_frames=cfg.engagement.temporal_window_frames)
    engagement_logger  = EngagementCSVLogger(
        session_id=session_id,
        logs_dir=cfg.logging.logs_dir,
    )

    # ── Session-level state ────────────────────────────────────────────────

    # T7: consolidated identity map — single dict replaces 4 parallel dicts
    confirmed_tracks: dict[int, TrackState] = {}
    missing_track_frames: dict[int, int]    = {}

    frame_index = 0
    cv2.namedWindow(cfg.app.window_name, cv2.WINDOW_NORMAL)

    # ── Frame loop ─────────────────────────────────────────────────────────

    while True:

        # ── Stage 1: Capture ───────────────────────────────────────────────
        ok, frame = cap.read()
        if not ok or frame is None:
            log_warn("Frame read failed — stopping.")
            break

        frame_index += 1

        # ── Stage 2: Detection ─────────────────────────────────────────────
        detections = detector.detect(frame)

        # ── Stage 3: Tracking + stale-track cleanup ────────────────────────
        tracks = tracker.update(detections)
        active_ids = {t.track_id for t in tracks}

        known_ids = (
            set(confirmed_tracks)
            | set(missing_track_frames)
            | set(attendance_manager.track_states)
            | set(engagement_tracker.memory)
        )
        expired_ids: set[int] = set()

        for tid in known_ids:
            if tid in active_ids:
                missing_track_frames.pop(tid, None)
                continue
            missing_track_frames[tid] = missing_track_frames.get(tid, 0) + 1
            if missing_track_frames[tid] >= cfg.tracking.cleanup_after_missing_frames:
                missing_track_frames.pop(tid, None)
                confirmed_tracks.pop(tid, None)
                expired_ids.add(tid)

        attendance_manager.remove_tracks(expired_ids)
        engagement_tracker.remove_tracks(expired_ids)

        active_id_list = list(active_ids)
        frame_records: list[dict] = []

        # ── Stages 4–7: per-track pipeline ────────────────────────────────
        for track in tracks:
            box = track.bbox
            tid = track.track_id

            # ── T2: compute crop ONCE — reuse for recognition and pose ─────
            person_crop = face_embedder.crop_person_region(
                frame, (box.x1, box.y1, box.x2, box.y2)
            )
            # Full-height crop for body pose (shoulders + hips must be visible)
            h0, w0 = frame.shape[:2]
            full_crop = frame[
                max(0, box.y1): min(h0, box.y2),
                max(0, box.x1): min(w0, box.x2),
            ]

            # ── Stage 4: Recognition ───────────────────────────────────────
            _maybe_run_recognition(
                frame_index=frame_index,
                track_id=tid,
                person_crop=person_crop,
                frame=frame,
                box=box,
                face_embedder=face_embedder,
                emb_cache=emb_cache,
                face_db=face_db,
                attendance_manager=attendance_manager,
                confirmed_tracks=confirmed_tracks,
                session_id=session_id,
                cfg=cfg,
            )

            # Display label: name if confirmed, else track ID
            if tid in confirmed_tracks:
                label = confirmed_tracks[tid].name
            else:
                label = f"track:{tid} {box.confidence:.2f}"

            draw_bbox(frame, box.x1, box.y1, box.x2, box.y2, label)

            # ── Stage 5: Pose estimation ───────────────────────────────────
            if person_crop.size == 0:
                continue

            pose = head_pose_estimator.estimate(person_crop)
            if pose is None:
                continue

            draw_pose(frame, box.x1, box.y1, pose.yaw, pose.pitch, pose.roll)

            body_pose = None
            if body_pose_estimator is not None and full_crop.size > 0:
                body_pose = body_pose_estimator.estimate(full_crop)

            # ── Stage 6: Engagement scoring ────────────────────────────────
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

            # ── Stage 7: Temporal aggregation ──────────────────────────────
            memory = engagement_tracker.update(tid, engagement.label, engagement.score)

            draw_engagement_stats(frame, box.x1, box.y1, memory.attentive_ratio(), memory.average_score())
            draw_feature_stats(frame, box.x1, box.y1, features.eye_aspect_ratio, features.gaze_score, features.body_tilt)

            # Accumulate frame record for logging (only if identity confirmed)
            if tid in confirmed_tracks:
                ts = confirmed_tracks[tid]
                frame_records.append({
                    "track_id":    tid,
                    "student_code": ts.student_code,
                    "name":         ts.name,
                    "person_confidence": box.confidence,
                    "face_confidence":   ts.face_confidence,
                    "label":   engagement.label,
                    "score":   engagement.score,
                    "reason":  engagement.reason,
                    "attentive_ratio": memory.attentive_ratio(),
                    "features": features,
                })

        # ── Stage 8: Class-level rendering ────────────────────────────────
        class_avg_attention = engagement_tracker.class_average_attentive_ratio(active_id_list)
        class_avg_score     = engagement_tracker.class_average_score(active_id_list)

        draw_class_stats(frame, class_avg_attention, class_avg_score)

        if cfg.debug.draw_center_dot:
            draw_center_dot(frame)
        if cfg.app.show_fps:
            draw_fps(frame, cap.fps)
        if cfg.debug.show_debug_panel:
            draw_debug_panel(
                frame,
                fps=cap.fps,
                active_tracks=len(tracks),
                recognized_count=len(confirmed_tracks),
                class_avg_score=class_avg_score,
                class_attentive_ratio=class_avg_attention,
                device=detector.device,
            )

        # ── Stage 9: CSV logging ───────────────────────────────────────────
        if frame_index % cfg.logging.interval_frames == 0:
            for rec in frame_records:
                engagement_logger.log(
                    track_id=rec["track_id"],
                    student_code=rec["student_code"],
                    name=rec["name"],
                    person_confidence=rec["person_confidence"],
                    face_confidence=rec["face_confidence"],
                    label=rec["label"],
                    score=rec["score"],
                    reason=rec["reason"],
                    attentive_ratio=rec["attentive_ratio"],
                    class_avg_attention=class_avg_attention,
                    class_avg_score=class_avg_score,
                    yaw=rec["features"].yaw,
                    pitch=rec["features"].pitch,
                    roll=rec["features"].roll,
                    eye_aspect_ratio=rec["features"].eye_aspect_ratio,
                    eyes_closed=rec["features"].eyes_closed,
                    gaze_x=rec["features"].gaze_x,
                    gaze_y=rec["features"].gaze_y,
                    gaze_score=rec["features"].gaze_score,
                    body_tilt=rec["features"].body_tilt,
                    body_visible=rec["features"].body_visible,
                )

        # ── Display ────────────────────────────────────────────────────────
        cv2.imshow(cfg.app.window_name, frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            break

    # ── Shutdown ───────────────────────────────────────────────────────────
    cap.release()
    cv2.destroyAllWindows()
    engagement_logger.close()
    log_info("Stopped.")


# ── Recognition helper ─────────────────────────────────────────────────────

def _maybe_run_recognition(
    *,
    frame_index: int,
    track_id: int,
    person_crop: np.ndarray,
    frame: np.ndarray,
    box,
    face_embedder: FaceEmbedder,
    emb_cache: EmbeddingCache,
    face_db: FaceDatabase,
    attendance_manager: AttendanceManager,
    confirmed_tracks: dict[int, TrackState],
    session_id: str,
    cfg,
) -> None:
    """
    Run InsightFace recognition for unconfirmed tracks on every Nth frame.
    Uses the pre-loaded EmbeddingCache instead of querying SQLite each time.
    On success, writes the identity into confirmed_tracks and marks attendance.
    """
    if track_id in confirmed_tracks:
        return  # already confirmed — don't re-run
    if frame_index % cfg.recognition.interval_frames != 0:
        return  # not our turn
    if person_crop.size == 0:
        return

    face_result = face_embedder.get_embedding(person_crop)
    if face_result is None:
        return
    if face_result.det_score < cfg.recognition.min_face_det_score:
        return  # low quality detection — skip

    # T3: cache search instead of DB read
    match = emb_cache.find_best(face_result.embedding, cfg.recognition.similarity_threshold)
    if match is None:
        return

    best_code, best_score = match

    student_code = attendance_manager.confirm_match(
        track_id=track_id,
        student_code=best_code,
        score=best_score,
        db=face_db,
        session_id=session_id,
    )

    if student_code is not None:
        name = face_db.get_student_name(student_code)
        if name and name != "name":   # guard against corrupt DB entries
            confirmed_tracks[track_id] = TrackState(
                name=name,
                student_code=student_code,
                face_confidence=face_result.det_score,
            )


if __name__ == "__main__":
    run()

    # Useful commands:
    # python -m src.app.main
    # python -m scripts.register_face --image photos/me.jpg --code CS21_001 --name "Adilet Orozaliev"
    # python -m scripts.analyze_engagement logs/session_<timestamp>.csv
