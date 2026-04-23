from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np

from src.core.logger import log_warn


@dataclass
class HeadPoseResult:
   yaw: float
   pitch: float
   roll: float
   eye_aspect_ratio: float
   eyes_closed: bool
   gaze_x: float
   gaze_y: float
   gaze_score: float


class HeadPoseEstimator:
   LEFT_EYE = [33, 160, 158, 133, 153, 144]
   RIGHT_EYE = [362, 385, 387, 263, 373, 380]
   LEFT_IRIS = [468, 469, 470, 471, 472]
   RIGHT_IRIS = [473, 474, 475, 476, 477]

   # Approximate canonical 3D face model points in millimeters.
   LANDMARK_IDS = [33, 263, 1, 61, 291, 199]
   MODEL_POINTS = np.array(
      [
         [-225.0, 170.0, -135.0],  # left eye outer corner
         [225.0, 170.0, -135.0],   # right eye outer corner
         [0.0, 0.0, 0.0],          # nose tip
         [-150.0, -150.0, -125.0], # left mouth corner
         [150.0, -150.0, -125.0],  # right mouth corner
         [0.0, -330.0, -65.0],     # chin / lower face
      ],
      dtype=np.float64,
   )

   def __init__(
      self,
      eye_closed_threshold: float = 0.19,
      gaze_center_tolerance_x: float = 0.22,
      gaze_center_tolerance_y: float = 0.18,
   ) -> None:
      self.eye_closed_threshold = eye_closed_threshold
      self.gaze_center_tolerance_x = gaze_center_tolerance_x
      self.gaze_center_tolerance_y = gaze_center_tolerance_y
      self.mp_face_mesh = mp.solutions.face_mesh
      try:
         self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
         )
      except Exception as exc:
         log_warn(f"MediaPipe FaceMesh unavailable: {exc}")
         self.face_mesh = None

   def estimate(self, image: np.ndarray) -> Optional[HeadPoseResult]:
      if image.size == 0 or self.face_mesh is None:
         return None

      rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
      results = self.face_mesh.process(rgb)

      if not results.multi_face_landmarks:
         return None

      face_landmarks = results.multi_face_landmarks[0]
      h, w = image.shape[:2]

      face_2d = []

      for idx in self.LANDMARK_IDS:
         lm = face_landmarks.landmark[idx]
         x, y = int(lm.x * w), int(lm.y * h)
         face_2d.append([x, y])

      face_2d = np.array(face_2d, dtype=np.float64)

      focal_length = w
      cam_matrix = np.array(
         [
            [focal_length, 0, w / 2],
            [0, focal_length, h / 2],
            [0, 0, 1],
         ],
         dtype=np.float64,
      )

      dist_matrix = np.zeros((4, 1), dtype=np.float64)

      success, rot_vec, trans_vec = cv2.solvePnP(
         self.MODEL_POINTS,
         face_2d,
         cam_matrix,
         dist_matrix,
         flags=cv2.SOLVEPNP_ITERATIVE,
      )

      if not success:
         return None

      rmat, _ = cv2.Rodrigues(rot_vec)
      angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)

      pitch = float(angles[0] * 360)
      yaw = float(angles[1] * 360)
      roll = float(angles[2] * 360)
      ear = self._eye_aspect_ratio(face_landmarks, w, h)
      gaze_x, gaze_y, gaze_score = self._gaze(face_landmarks, w, h)

      return HeadPoseResult(
         yaw=yaw,
         pitch=pitch,
         roll=roll,
         eye_aspect_ratio=ear,
         eyes_closed=ear < self.eye_closed_threshold,
         gaze_x=gaze_x,
         gaze_y=gaze_y,
         gaze_score=gaze_score,
      )

   def _point(self, face_landmarks, idx: int, w: int, h: int) -> np.ndarray:
      lm = face_landmarks.landmark[idx]
      return np.array([lm.x * w, lm.y * h], dtype=np.float64)

   def _eye_aspect_ratio(self, face_landmarks, w: int, h: int) -> float:
      left = [self._point(face_landmarks, idx, w, h) for idx in self.LEFT_EYE]
      right = [self._point(face_landmarks, idx, w, h) for idx in self.RIGHT_EYE]
      return float((self._single_eye_aspect_ratio(left) + self._single_eye_aspect_ratio(right)) / 2.0)

   @staticmethod
   def _single_eye_aspect_ratio(eye: list[np.ndarray]) -> float:
      a = np.linalg.norm(eye[1] - eye[5])
      b = np.linalg.norm(eye[2] - eye[4])
      c = np.linalg.norm(eye[0] - eye[3])
      if c == 0.0:
         return 0.0
      return float((a + b) / (2.0 * c))

   def _gaze(self, face_landmarks, w: int, h: int) -> tuple[float, float, float]:
      left = [self._point(face_landmarks, idx, w, h) for idx in self.LEFT_EYE]
      right = [self._point(face_landmarks, idx, w, h) for idx in self.RIGHT_EYE]
      left_iris = [self._point(face_landmarks, idx, w, h) for idx in self.LEFT_IRIS]
      right_iris = [self._point(face_landmarks, idx, w, h) for idx in self.RIGHT_IRIS]

      left_ratio = self._iris_ratio(left, left_iris)
      right_ratio = self._iris_ratio(right, right_iris)

      gaze_x = float(((left_ratio[0] + right_ratio[0]) / 2.0) - 0.5)
      gaze_y = float(((left_ratio[1] + right_ratio[1]) / 2.0) - 0.5)

      x_penalty = abs(gaze_x) / self.gaze_center_tolerance_x
      y_penalty = abs(gaze_y) / self.gaze_center_tolerance_y
      gaze_score = max(0.0, 1.0 - max(x_penalty, y_penalty))

      return gaze_x, gaze_y, float(gaze_score)

   @staticmethod
   def _iris_ratio(eye: list[np.ndarray], iris: list[np.ndarray]) -> tuple[float, float]:
      eye_points = np.array(eye)
      iris_center = np.mean(np.array(iris), axis=0)

      min_xy = np.min(eye_points, axis=0)
      max_xy = np.max(eye_points, axis=0)
      span = np.maximum(max_xy - min_xy, 1.0)
      ratio = (iris_center - min_xy) / span
      return float(ratio[0]), float(ratio[1])
