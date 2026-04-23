from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np

from src.core.logger import log_warn


@dataclass
class BodyPoseResult:
   torso_tilt: float
   shoulder_slope: float
   visibility: float

   @property
   def visible(self) -> bool:
      return self.visibility >= 0.5


class BodyPoseEstimator:
   def __init__(
      self,
      min_detection_confidence: float = 0.5,
      min_tracking_confidence: float = 0.5,
   ) -> None:
      self.mp_pose = mp.solutions.pose
      model_path = (
         Path(mp.__file__).resolve().parent
         / "modules"
         / "pose_landmark"
         / "pose_landmark_lite.tflite"
      )
      if not model_path.exists():
         log_warn(f"MediaPipe Pose model not found: {model_path}")
         self.pose = None
         return

      try:
         self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=0,
            enable_segmentation=False,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
         )
      except Exception as exc:
         log_warn(f"MediaPipe Pose unavailable: {exc}")
         self.pose = None

   def estimate(self, image: np.ndarray) -> Optional[BodyPoseResult]:
      if image.size == 0 or self.pose is None:
         return None

      rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
      results = self.pose.process(rgb)

      if not results.pose_landmarks:
         return None

      landmarks = results.pose_landmarks.landmark
      left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
      right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
      left_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP]
      right_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP]

      keypoints = [left_shoulder, right_shoulder, left_hip, right_hip]
      visibility = float(np.mean([point.visibility for point in keypoints]))

      if visibility < 0.35:
         return BodyPoseResult(torso_tilt=0.0, shoulder_slope=0.0, visibility=visibility)

      shoulder_mid = np.array(
         [
            (left_shoulder.x + right_shoulder.x) / 2.0,
            (left_shoulder.y + right_shoulder.y) / 2.0,
         ],
         dtype=np.float64,
      )
      hip_mid = np.array(
         [
            (left_hip.x + right_hip.x) / 2.0,
            (left_hip.y + right_hip.y) / 2.0,
         ],
         dtype=np.float64,
      )
      torso_vector = shoulder_mid - hip_mid

      torso_tilt = float(np.degrees(np.arctan2(torso_vector[0], -torso_vector[1])))
      shoulder_vector = np.array(
         [right_shoulder.x - left_shoulder.x, right_shoulder.y - left_shoulder.y],
         dtype=np.float64,
      )
      shoulder_slope = float(np.degrees(np.arctan2(shoulder_vector[1], shoulder_vector[0])))

      return BodyPoseResult(
         torso_tilt=torso_tilt,
         shoulder_slope=shoulder_slope,
         visibility=visibility,
      )
