# CV Smart Classroom Assistant

## Что было до улучшений

### 1. Детекция и трекинг

- `YOLOv8n` использовался как базовая модель для детекции людей в кадре.
- `ByteTrack` отслеживал найденных людей между кадрами и выдавал стабильный `track_id`.
- Компонент детекции назывался `FaceDetector`, хотя по факту находил людей (`COCO class 0 = person`), из-за чего архитектура читалась неоднозначно.

### 2. Распознавание лиц

- `InsightFace buffalo_l` использовался для получения face embeddings.
- Сравнение embeddings с `data/faces.db` позволяло связать трек с зарегистрированным студентом.
- Для распознавания использовалась верхняя часть `person bbox`, где обычно находится лицо.

### 3. Engagement baseline

- Вовлеченность определялась простыми правилами по `yaw` и `pitch`.
- Такой baseline был понятным и быстрым, но фактически измерял в основном направление головы.
- Eye state, gaze и body pose не были полноценно интегрированы в расчет.

### 4. Логирование и отчеты

- В CSV сохранялись в основном итоговые значения: `label`, `score`, `attentive_ratio`, class averages.
- Сырые признаки почти не сохранялись, поэтому данные были ограниченно полезны для дальнейшего ML.
- Отчет строил простой bar chart по проценту внимания.

### 5. Инженерные ограничения

- `device` для YOLO мог быть захардкожен.
- Не было полноценного `session_id` в CSV и SQLite attendance.
- Старые состояния треков/identity/engagement не очищались явно.
- `requirements.txt` был сохранен в неудобной кодировке.

## Что стало сейчас

### 1. Детекция людей

- `YOLOv8n` сохранен как основной detector для людей.
- Добавлен отдельный `PersonDetector`, который явно работает с классом `person`.
- Старый `FaceDetector` оставлен только как compatibility alias, чтобы старые импорты не ломались.
- `device`, `image_size` и confidence threshold вынесены в `configs/default.yaml`.

### 2. Трекинг и состояние

- `ByteTrack` продолжает отвечать за `track_id`.
- Для потерянных треков добавлена очистка связанных runtime-состояний:
  - identity;
  - student code;
  - face confidence;
  - attendance recognition state;
  - engagement memory.
- Это уменьшает риск, что имя или engagement history останутся привязанными к уже исчезнувшему треку.

### 3. Распознавание лиц

- `InsightFace buffalo_l` остается основным инструментом face recognition.
- Распознавание по-прежнему запускается на верхней части `person crop`, а не на отдельной глобальной стадии `face detector -> face/person association`.
- `InsightFace` внутри crop находит лицо и возвращает `det_score`; этот `face_confidence` теперь сохраняется в лог.
- После подтверждения identity система сохраняет не только имя, но и `student_code`.

### 4. Head pose, eyes и gaze

- `MediaPipe FaceMesh` теперь используется шире:
  - `yaw`;
  - `pitch`;
  - `roll`;
  - `eye_aspect_ratio`;
  - `eyes_closed`;
  - rough `gaze_x`;
  - rough `gaze_y`;
  - `gaze_score`.
- Head pose считается через фиксированную приближенную 3D face model и `solvePnP`, а не через псевдо-3D точки из тех же 2D landmarks.
- Важно: признаки лица все еще извлекаются из верхней части `person crop`. Это лучше прежнего baseline, но еще не полноценный face-bbox pipeline.

### 5. Body pose

- Добавлен `BodyPoseEstimator` на MediaPipe Pose.
- Он оценивает:
  - `torso_tilt`;
  - `shoulder_slope`;
  - visibility плеч/бедер.
- Body pose работает как optional feature: если MediaPipe Pose или модель недоступны, приложение не падает, а продолжает работать без body features.

### 6. Engagement scoring

- Rule-based baseline сохранен, но стал feature-based.
- Теперь scorer учитывает:
  - направление головы;
  - rough gaze;
  - закрытые глаза;
  - наклон головы;
  - наклон корпуса, если корпус виден.
- Возможные labels стали подробнее:
  - `attentive`;
  - `looking_away`;
  - `looking_down`;
  - `eyes_closed`;
  - `head_tilted`;
  - `leaning`;
  - `distracted`.
- Это все еще не обученная ML-модель engagement, а честный baseline для сбора данных и дальнейшей валидации.

### 7. Логирование

- CSV-лог теперь содержит `session_id`.
- В новые логи сохраняются:
  - `track_id`;
  - `student_code`;
  - `name`;
  - `person_confidence`;
  - `face_confidence`;
  - `label`;
  - `score`;
  - `reason`;
  - `attentive_ratio`;
  - `class_avg_attention`;
  - `class_avg_score`;
  - `yaw`;
  - `pitch`;
  - `roll`;
  - `eye_aspect_ratio`;
  - `eyes_closed`;
  - `gaze_x`;
  - `gaze_y`;
  - `gaze_score`;
  - `body_tilt`;
  - `body_visible`.
- Такой формат уже подходит для последующего анализа, калибровки thresholds и обучения baseline ML-моделей.

### 8. Сессионная структура

- `session_id` используется в CSV.
- `session_id` добавлен в SQLite `attendance_logs`.
- Для старой базы есть migration: если колонки `session_id` нет, она добавляется автоматически.
- Это отделяет разные запуски системы друг от друга.

### 9. Конфигурация

- В `configs/default.yaml` вынесены:
  - video settings;
  - detector model/device/confidence;
  - recognition similarity threshold;
  - recognition interval;
  - tracking cleanup timeout;
  - logging interval;
  - engagement thresholds для yaw/pitch/roll/gaze/body/eyes.
- Проект стало проще калибровать под разные камеры, аудитории и машины.

### 10. Отчеты

- `scripts/analyze_engagement.py` теперь генерирует не один простой график, а пакет отчета:
  - dashboard PNG;
  - student summary CSV;
  - timeline CSV;
  - label distribution CSV;
  - Markdown report.
- Attention считается по времени между timestamps, а не просто по количеству строк.
- В отчетах есть:
  - attendance status;
  - observed minutes;
  - attentive minutes;
  - attention percent;
  - average score;
  - class timeline;
  - behavior distribution по labels.
- Если доступен roster из `data/faces.db`, отсутствующие студенты могут попасть в summary как `absent`.
- Пустые сессии теперь обрабатываются корректно и не ломают analyzer.

## Что пока не реализовано полностью

### 1. Полный face-bbox pipeline

- Пока нет отдельного глобального этапа:
  - найти все лица в кадре;
  - связать face bbox с person track;
  - использовать именно face bbox для recognition/head pose/gaze.
- Сейчас система использует верхнюю часть `person crop`, а внутри нее InsightFace/FaceMesh ищут лицо.
- Это проще и быстрее, но менее точно при дальних лицах, перекрытиях и сложных ракурсах.

### 2. Обученная ML-модель engagement

- Logistic Regression / Random Forest / MLP пока не обучены.
- Текущий scorer остается rule-based baseline.
- Зато новые CSV уже содержат признаки, на которых можно обучать первую модель.

### 3. Валидация качества

- Thresholds для gaze, EAR, yaw/pitch/roll и similarity еще требуют калибровки на реальных видео из аудитории.
- Нужен размеченный dataset с ground truth labels.
- Без этого проценты внимания нельзя считать объективной педагогической метрикой.

### 4. Надежная работа MediaPipe в headless окружении

- В GUI-запуске с камерой pipeline рассчитан на работу штатно.
- В headless/sandbox окружениях MediaPipe может не создать OpenGL context; код обрабатывает это через fallback и не падает.

## Итоговое сравнение

### Было

- Рабочий real-time baseline на `YOLOv8n + ByteTrack + InsightFace + MediaPipe FaceMesh`.
- Engagement оценивался почти только по `yaw/pitch`.
- Логи были удобны для демонстрации, но слабо подходили для дальнейшего ML.
- Отчеты были простыми.

### Стало

- Архитектура стала чище: `PersonDetector`, отдельные feature modules, расширенный scorer, отдельный report generator.
- Engagement стал feature-based: head pose + eyes + rough gaze + optional body pose.
- Логи стали богаче и включают raw features, confidence и `student_code`.
- Появился `session_id` в CSV и SQLite.
- Отчеты стали сессионными, временными и более полезными для анализа класса.
- Проект лучше подготовлен к следующему этапу: сбору размеченных данных и обучению первой ML-модели engagement.

## Краткий вывод

Проект стал сильнее не за счет полной замены технологий, а за счет правильного усиления существующего pipeline. Текущий стек остается практичным для real-time прототипа, а новые логи и отчеты уже создают основу для валидации и ML. Главный следующий шаг: собрать размеченные данные и проверить, насколько rule-based labels совпадают с реальным поведением студентов.
