# Smart Classroom — Real-time Engagement Analytics

Real-time computer-vision pipeline that watches a classroom through a single
camera and produces **attendance** and **engagement** analytics for reports.

> ⚠️ **Engagement here is a proxy metric, not ground truth.** The system infers
> "attention" from head pose, gaze direction, eye state and body posture using
> hand-tuned rules. It approximates whether a student is *oriented toward the
> front of the room* — it does **not** measure actual cognitive engagement.
> Treat the numbers as a coarse signal, not an objective verdict on a student.

---

## What it does

1. **Detect** people in the frame (YOLOv8, COCO `person` class).
2. **Track** each person across frames (ByteTrack via `supervision`).
3. **Recognize** students by face for attendance (InsightFace embeddings vs. an
   enrolled database).
4. **Estimate engagement** from head pose (MediaPipe FaceMesh → yaw/pitch/roll),
   gaze, eye-aspect-ratio and optional body pose (MediaPipe Pose).
5. **Aggregate** per-student engagement over a sliding time window and compute a
   class-wide average.
6. **Log** everything to a per-session CSV for offline reporting.

## Pipeline (per frame)

```
camera
  → YOLO person detection            (src/detection/person_detector.py)
  → ByteTrack tracking               (src/tracking/tracker.py)
  → person crop (once per track)     (src/core/geometry.py)
  → InsightFace embedding            (src/recognition/face_embedder.py)
  → attendance recognition           (src/recognition/attendance_logic.py + embedding_cache.py)
  → MediaPipe FaceMesh head pose     (src/features/head_pose.py)
  → engagement scoring (rules)       (src/engagement/scoring.py)
  → temporal smoothing               (src/engagement/temporal.py)
  → class average metrics            (src/engagement/temporal.py)
  → CSV logging                      (src/engagement/logger.py)
```

---

## Installation

Requires **Python 3.11**.

```bash
python3.11 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

On first run, Ultralytics downloads `yolov8n.pt` (if absent) and InsightFace
downloads the `buffalo_l` model pack into `~/.insightface`.

**Apple Silicon (M1–M4):** set `detection.device: "auto"` (the default) and YOLO
runs on the Metal (MPS) backend automatically. InsightFace runs on CPU
(`CPUExecutionProvider`) for stability.

---

## Register a face (enrollment)

Add a student to the face database from a clear, front-facing photo:

```bash
python -m scripts.register_face --image photos/me.jpg --code CS21_001 --name "Adilet Orozaliev"
```

- `--code` is a unique student id (e.g. `CS21_001`).
- You can register multiple photos for the same `--code`; recognition uses the
  best match across all of a student's embeddings.
- Data is stored in `data/faces.db` (SQLite). The running app loads all
  embeddings into memory once at startup.

---

## Run real-time

```bash
python -m src.app.main
```

Run on a recorded video instead of a webcam (no camera needed — handy for
testing) by overriding the source via env var or `configs/default.yaml`:

```bash
VIDEO_SOURCE=path/to/clip.mp4 python -m src.app.main
```

### Hotkeys

| Key        | Action |
|------------|--------|
| `q` / `Esc`| Quit   |

---

## Reports

Generate per-student attendance + engagement dashboards from a session log:

```bash
python -m scripts.analyze_engagement logs/session_<timestamp>.csv
```

Outputs (CSV summaries, a timeline, a PNG dashboard and a Markdown report) are
written to `reports/`.

---

## Where things live

| Path                       | Contents |
|----------------------------|----------|
| `configs/default.yaml`     | All tunable parameters |
| `data/faces.db`            | Enrolled students + face embeddings + attendance (SQLite) |
| `logs/session_*.csv`       | Per-session engagement logs |
| `reports/`                 | Generated dashboards & reports |
| `photos/`                  | Enrollment photos (not tracked in git) |

`data/`, `logs/*.csv`, `reports/`, `photos/` and model weights are
git-ignored — they are regenerated/local artifacts.

---

## Configuration

Everything tunable lives in `configs/default.yaml`. Key sections:

- **`app`** — window name, FPS overlay.
- **`video`** — source, resolution, target FPS, mirror.
- **`detection`** — YOLO model, `image_size`, `confidence_threshold`, `device`,
  and a garbage-box filter (`min_box_width` / `min_box_height` / `min_box_area`).
- **`tracking`** — `cleanup_after_missing_frames` plus ByteTrack tuning
  (`track_activation_threshold`, `lost_track_buffer`,
  `minimum_matching_threshold`, `minimum_consecutive_frames`).
- **`recognition`** — `similarity_threshold`, `confirm_matches`,
  `interval_frames` (recognize every N frames per unconfirmed track),
  `min_face_det_score`.
- **`engagement`** — pose/gaze/body thresholds, `attentive_score_threshold`,
  `use_body_pose` (disable for a big FPS win), `temporal_window_frames`.
- **`logging`** — `interval_frames` (CSV write cadence), `logs_dir`.
- **`debug`** — `draw_center_dot`, `show_debug_panel`.

---

## Engagement labels & scoring

Single-frame labels (see `src/engagement/labels.py`):

| Label          | Meaning                              | Score cap |
|----------------|--------------------------------------|-----------|
| `attentive`    | Oriented toward the front, eyes open | up to 1.0 |
| `distracted`   | No hard rule fired but score is low  | < 0.60    |
| `looking_away` | Strong yaw / off-axis gaze           | ≤ 0.50    |
| `looking_down` | Pitch down / gaze down (phone, desk) | ≤ 0.45    |
| `eyes_closed`  | Eye-aspect-ratio below threshold     | ≤ 0.25    |
| `head_tilted`  | Large roll                           | ≤ 0.60    |
| `leaning`      | Large torso tilt                     | ≤ 0.65    |

The composite score is a weighted blend (head 45%, gaze 30%, eye 15%, body 10%)
— **heuristic weights**, documented honestly in `scoring.py`. `attentive_ratio`
counts only `attentive` frames over the temporal window. Tracks with no
detectable face (no pose) are simply excluded from the class average rather than
counted against it.

---

## Tests / sanity checks

Pure-Python sanity tests (no camera, GPU, or ML model downloads needed):

```bash
python tests/run_tests.py        # no pytest required
# or, if pytest is installed:
pytest tests/
```

Covers: engagement scorer labels/scores, temporal ratios, bbox clamp/crop edge
cases, and the embedding-cache loader/search. Heavy ML inference is intentionally
**not** unit-tested.

---

## Limitations

- **Proxy metric.** Engagement is inferred from geometry, not understanding.
- **Rules, not ML.** Scoring weights/thresholds are hand-tuned. No trained
  classifier yet (would require a labelled dataset).
- **Head pose** uses `solvePnP` on a generic 3D face model; absolute
  yaw/pitch/roll are approximate and benefit from per-camera calibration.
- **Recognition** depends on enrollment photo quality, lighting and resolution;
  small/blurry faces fail `min_face_det_score` and are skipped.
- **Crowds / occlusion** degrade tracking; `track_id` may change after long
  occlusions, which restarts a student's temporal window.
- **Single camera, single view.** No multi-camera fusion.
