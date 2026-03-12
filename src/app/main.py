from __future__ import annotations

import cv2

from src.core.config import load_config
from src.core.logger import log_info, log_warn
from src.video.capture import VideoCapture
from src.ui.overlay import draw_fps, draw_center_dot, draw_bbox

from src.detection.face_detector import FaceDetector
from src.tracking.tracker import Tracker
from src.recognition.face_database import FaceDatabase
from src.recognition.face_embedder import FaceEmbedder
from src.recognition.attendance_logic import AttendanceManager

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

      for track in tracks:
         box = track.bbox
         label = track_identity.get(
            track.track_id,
            f"track:{track.track_id} {box.confidence:.2f}"
         )

         # Распознаем не каждый кадр, чтобы не убить FPS
         if frame_index % 10 == 0:
            head_crop = face_embedder.crop_head_region(
               frame,
               (box.x1, box.y1, box.x2, box.y2),
            )

            if head_crop.size != 0:
               face_result = face_embedder.get_embedding(head_crop)

               if face_result is not None:
                  student_code = attendance_manager.recognize_embedding(
                     track_id=track.track_id,
                     embedding=face_result.embedding,
                     db=face_db,
                     embedder=face_embedder,
                  )

                  if student_code is not None:
                     name = face_db.get_student_name(student_code)
                     if name:
                        track_identity[track.track_id] = name
                        label = name

         draw_bbox(frame, box.x1, box.y1, box.x2, box.y2, label)


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
   log_info("Stopped.")


if __name__ == "__main__":
   run()
