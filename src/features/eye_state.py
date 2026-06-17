import numpy as np

LEFT_EYE = [33, 34, 35, 36, 37, 38]
RIGHT_EYE = [39, 40, 41, 42, 43, 44]

class EyeStateDetector:
    def __init__(self, threshold: float = 0.2, min_frames_closed: int = 3):
        self.threshold = threshold
        self.min_frames_closed = min_frames_closed
        self.closed_counter = {}

    def _euclidean(self, p1, p2):
        return np.linalg.norm(np.array(p1) - np.array(p2))

    def compute_ear(self, eye):
        # eye = 6 точек глаза
        A = self._euclidean(eye[1], eye[5])
        B = self._euclidean(eye[2], eye[4])
        C = self._euclidean(eye[0], eye[3])
        return (A + B) / (2.0 * C)

    def is_blinking(self, track_id, left_eye, right_eye):
        left_ear = self.compute_ear(left_eye)
        right_ear = self.compute_ear(right_eye)

        ear = (left_ear + right_ear) / 2.0

        if track_id not in self.closed_counter:
            self.closed_counter[track_id] = 0

        if ear < self.threshold:
            self.closed_counter[track_id] += 1
        else:
            self.closed_counter[track_id] = 0

        return self.closed_counter[track_id] >= self.min_frames_closed, ear