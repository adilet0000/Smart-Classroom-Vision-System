from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import os
import yaml
from dotenv import load_dotenv


def _deep_merge(base: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
   result = dict(base)
   for k, v in extra.items():
      if isinstance(v, dict) and isinstance(result.get(k), dict):
         result[k] = _deep_merge(result[k], v)
      else:
         result[k] = v
   return result


@dataclass(frozen=True)
class AppConfig:
   window_name: str = "Engagement Detection"
   show_fps: bool = True


@dataclass(frozen=True)
class VideoConfig:
   source: Any = 0
   width: int = 1280
   height: int = 720
   target_fps: int = 30
   mirror: bool = True


@dataclass(frozen=True)
class DetectionConfig:
   model_name: str = "yolov8n.pt"
   image_size: int = 640
   confidence_threshold: float = 0.25
   device: str = "auto"


@dataclass(frozen=True)
class RecognitionConfig:
   similarity_threshold: float = 0.45
   confirm_matches: int = 3
   # How many frames between recognition attempts per unconfirmed track.
   # Higher = cheaper; lower = faster first-recognition.
   interval_frames: int = 10
   # Minimum face detection score from InsightFace to use an embedding.
   min_face_det_score: float = 0.70


@dataclass(frozen=True)
class TrackingConfig:
   cleanup_after_missing_frames: int = 60


@dataclass(frozen=True)
class LoggingConfig:
   # Write a CSV row every N frames for each recognized student.
   interval_frames: int = 30
   logs_dir: str = "logs"


@dataclass(frozen=True)
class EngagementConfig:
   # ── Head pose thresholds (degrees — calibrated for corrected solvePnP output) ──
   # yaw: 0 = facing camera, positive = turning right, negative = turning left.
   # A student is "looking away" when |yaw| > yaw_away_threshold.
   yaw_away_threshold: float = 25.0

   # pitch: 0 = level, negative = looking down, positive = looking up.
   # A student is "looking down" when pitch < pitch_down_threshold.
   pitch_down_threshold: float = -8.0

   # roll: 0 = upright, ± = head tilt. Rarely engagement-breaking.
   roll_tilt_threshold: float = 20.0

   # ── Gaze thresholds (normalized iris offset, 0 = centered) ─────────────
   gaze_away_threshold: float = 0.25
   gaze_down_threshold: float = 0.22

   # ── Body pose ──────────────────────────────────────────────────────────
   body_tilt_threshold: float = 22.0

   # ── Eye state ──────────────────────────────────────────────────────────
   eye_closed_threshold: float = 0.19
   gaze_center_tolerance_x: float = 0.22
   gaze_center_tolerance_y: float = 0.18

   # ── Composite score ────────────────────────────────────────────────────
   attentive_score_threshold: float = 0.60

   # ── Temporal window ────────────────────────────────────────────────────
   # Number of frames in the sliding deque for per-track history.
   # At 30 fps: 90 frames ≈ 3 seconds of smoothing.
   temporal_window_frames: int = 90


@dataclass(frozen=True)
class DebugConfig:
   draw_center_dot: bool = True
   # Show the compact debug panel (FPS, tracks, IDs, class score, etc.)
   show_debug_panel: bool = True


@dataclass(frozen=True)
class Config:
   app: AppConfig
   video: VideoConfig
   detection: DetectionConfig
   recognition: RecognitionConfig
   tracking: TrackingConfig
   logging: LoggingConfig
   engagement: EngagementConfig
   debug: DebugConfig


def load_config(config_path: str = "configs/default.yaml") -> Config:
   load_dotenv(override=False)

   path = Path(config_path)
   data: dict[str, Any] = {}
   if path.exists():
      data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

   env_overrides: dict[str, Any] = {}
   video_source = os.getenv("VIDEO_SOURCE")
   if video_source:
      env_overrides = {"video": {"source": video_source}}

   merged = _deep_merge(data, env_overrides)

   app        = merged.get("app", {})
   video      = merged.get("video", {})
   detection  = merged.get("detection", {})
   recognition = merged.get("recognition", {})
   tracking   = merged.get("tracking", {})
   logging_   = merged.get("logging", {})
   engagement = merged.get("engagement", {})
   debug      = merged.get("debug", {})

   return Config(
      app=AppConfig(
         window_name=str(app.get("window_name", "Engagement Detection")),
         show_fps=bool(app.get("show_fps", True)),
      ),
      video=VideoConfig(
         source=video.get("source", 0),
         width=int(video.get("width", 1280)),
         height=int(video.get("height", 720)),
         target_fps=int(video.get("target_fps", 30)),
         mirror=bool(video.get("mirror", True)),
      ),
      detection=DetectionConfig(
         model_name=str(detection.get("model_name", "yolov8n.pt")),
         image_size=int(detection.get("image_size", 640)),
         confidence_threshold=float(detection.get("confidence_threshold", 0.25)),
         device=str(detection.get("device", "auto")),
      ),
      recognition=RecognitionConfig(
         similarity_threshold=float(recognition.get("similarity_threshold", 0.45)),
         confirm_matches=int(recognition.get("confirm_matches", 3)),
         interval_frames=int(recognition.get("interval_frames", 10)),
         min_face_det_score=float(recognition.get("min_face_det_score", 0.70)),
      ),
      tracking=TrackingConfig(
         cleanup_after_missing_frames=int(tracking.get("cleanup_after_missing_frames", 60)),
      ),
      logging=LoggingConfig(
         interval_frames=int(logging_.get("interval_frames", 30)),
         logs_dir=str(logging_.get("logs_dir", "logs")),
      ),
      engagement=EngagementConfig(
         yaw_away_threshold=float(engagement.get("yaw_away_threshold", 25.0)),
         pitch_down_threshold=float(engagement.get("pitch_down_threshold", -8.0)),
         roll_tilt_threshold=float(engagement.get("roll_tilt_threshold", 20.0)),
         gaze_away_threshold=float(engagement.get("gaze_away_threshold", 0.25)),
         gaze_down_threshold=float(engagement.get("gaze_down_threshold", 0.22)),
         body_tilt_threshold=float(engagement.get("body_tilt_threshold", 22.0)),
         eye_closed_threshold=float(engagement.get("eye_closed_threshold", 0.19)),
         gaze_center_tolerance_x=float(engagement.get("gaze_center_tolerance_x", 0.22)),
         gaze_center_tolerance_y=float(engagement.get("gaze_center_tolerance_y", 0.18)),
         attentive_score_threshold=float(engagement.get("attentive_score_threshold", 0.60)),
         temporal_window_frames=int(engagement.get("temporal_window_frames", 90)),
      ),
      debug=DebugConfig(
         draw_center_dot=bool(debug.get("draw_center_dot", True)),
         show_debug_panel=bool(debug.get("show_debug_panel", True)),
      ),
   )
