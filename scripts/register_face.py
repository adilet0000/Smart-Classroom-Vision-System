from __future__ import annotations

import argparse
import cv2

from src.recognition.face_database import FaceDatabase
from src.recognition.face_embedder import FaceEmbedder


def main() -> None:
   parser = argparse.ArgumentParser()
   parser.add_argument("--image", required=True, help="Путь к фото студента")
   parser.add_argument("--code", required=True, help="Уникальный код, например CS21_001")
   parser.add_argument("--name", required=True, help="Полное имя")
   args = parser.parse_args()

   image = cv2.imread(args.image)
   if image is None:
      raise ValueError("Не удалось прочитать изображение")

   db = FaceDatabase()
   embedder = FaceEmbedder()

   result = embedder.get_embedding(image)
   if result is None:
      raise ValueError("Лицо на фото не найдено")

   db.add_student(args.code, args.name)
   db.add_embedding(args.code, result.embedding)

   print(f"OK: зарегистрирован {args.name} ({args.code})")


if __name__ == "__main__":
   main()