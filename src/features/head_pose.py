from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np


@dataclass
class HeadPoseResult:
   yaw: float
   pitch: float
   roll: float


class HeadPoseEstimator:
   def __init__(self) -> None:
      self.mp_face_mesh = mp.solutions.face_mesh
      self.face_mesh = self.mp_face_mesh.FaceMesh(
         static_image_mode=False,
         max_num_faces=1,
         refine_landmarks=True,
         min_detection_confidence=0.5,
         min_tracking_confidence=0.5,
      )

   def estimate(self, image: np.ndarray) -> Optional[HeadPoseResult]:
      if image.size == 0:
         return None

      rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
      results = self.face_mesh.process(rgb)

      if not results.multi_face_landmarks:
         return None

      face_landmarks = results.multi_face_landmarks[0]
      h, w = image.shape[:2]

      # Ключевые точки MediaPipe
      landmark_ids = [33, 263, 1, 61, 291, 199]

      face_2d = []
      face_3d = []

      for idx in landmark_ids:
         lm = face_landmarks.landmark[idx]
         x, y = int(lm.x * w), int(lm.y * h)

         face_2d.append([x, y])
         face_3d.append([x, y, lm.z])

      face_2d = np.array(face_2d, dtype=np.float64)
      face_3d = np.array(face_3d, dtype=np.float64)

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
         face_3d,
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

      return HeadPoseResult(
         yaw=yaw,
         pitch=pitch,
         roll=roll,
      )