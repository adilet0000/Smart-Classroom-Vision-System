from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Optional

import numpy as np


class FaceDatabase:
   def __init__(self, db_path: str = "data/faces.db") -> None:
      self.db_path = db_path
      Path("data").mkdir(parents=True, exist_ok=True)
      self.conn = sqlite3.connect(self.db_path)
      self._create_tables()

   def _create_tables(self) -> None:
      cursor = self.conn.cursor()

      cursor.execute(
         """
         CREATE TABLE IF NOT EXISTS students (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_code TEXT UNIQUE NOT NULL,
            full_name TEXT NOT NULL
         )
         """
      )

      cursor.execute(
         """
         CREATE TABLE IF NOT EXISTS face_embeddings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_code TEXT NOT NULL,
            embedding_json TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
         )
         """
      )

      cursor.execute(
         """
         CREATE TABLE IF NOT EXISTS attendance_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_code TEXT NOT NULL,
            track_id INTEGER NOT NULL,
            first_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            status TEXT NOT NULL
         )
         """
      )

      self.conn.commit()

   def add_student(self, student_code: str, full_name: str) -> None:
      cursor = self.conn.cursor()
      cursor.execute(
         """
         INSERT OR IGNORE INTO students (student_code, full_name)
         VALUES (?, ?)
         """,
         (student_code, full_name),
      )
      self.conn.commit()

   def add_embedding(self, student_code: str, embedding: np.ndarray) -> None:
      cursor = self.conn.cursor()
      embedding_json = json.dumps(embedding.astype(float).tolist())

      cursor.execute(
         """
         INSERT INTO face_embeddings (student_code, embedding_json)
         VALUES (?, ?)
         """,
         (student_code, embedding_json),
      )
      self.conn.commit()

   def get_all_embeddings(self) -> list[tuple[str, np.ndarray]]:
      cursor = self.conn.cursor()
      cursor.execute(
         """
         SELECT student_code, embedding_json
         FROM face_embeddings
         """
      )

      rows = cursor.fetchall()
      output: list[tuple[str, np.ndarray]] = []

      for student_code, embedding_json in rows:
         vec = np.asarray(json.loads(embedding_json), dtype=np.float32)
         output.append((student_code, vec))

      return output

   def get_student_name(self, student_code: str) -> Optional[str]:
      cursor = self.conn.cursor()
      cursor.execute(
         """
         SELECT full_name
         FROM students
         WHERE student_code = ?
         """,
         (student_code,),
      )

      row = cursor.fetchone()
      if row is None:
         return None

      return str(row[0])

   def mark_present(self, student_code: str, track_id: int) -> None:
      cursor = self.conn.cursor()

      cursor.execute(
         """
         SELECT id
         FROM attendance_logs
         WHERE student_code = ? AND track_id = ? AND status = 'present'
         """,
         (student_code, track_id),
      )

      row = cursor.fetchone()

      if row is None:
         cursor.execute(
            """
            INSERT INTO attendance_logs (student_code, track_id, status)
            VALUES (?, ?, 'present')
            """,
            (student_code, track_id),
         )
      else:
         cursor.execute(
            """
            UPDATE attendance_logs
            SET last_seen = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            (row[0],),
         )

      self.conn.commit()