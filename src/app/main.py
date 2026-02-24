from __future__ import annotations

import cv2

from src.core.config import load_config
from src.core.logger import log_info, log_warn
from src.video.capture import VideoCapture
from src.ui.overlay import draw_fps, draw_center_dot, draw_bbox

from src.detection.face_detector import FaceDetector
from src.tracking.tracker import Tracker

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
   tracker = Tracker()


   cv2.namedWindow(cfg.app.window_name, cv2.WINDOW_NORMAL)

   while True:
      ok, frame = cap.read()
      if not ok or frame is None:
         log_warn("Frame read failed.")
         break

      detections = detector.detect(frame)
      tracks = tracker.update(detections)

      for track in tracks:
         box = track.bbox
         label = f"id:{track.track_id} {box.confidence:.2f}"
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
