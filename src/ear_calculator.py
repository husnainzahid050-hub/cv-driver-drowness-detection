"""
src/ear_calculator.py
─────────────────────
Eye Aspect Ratio (EAR) calculation.

The EAR formula was introduced by Soukupová & Čech (2016):

         ||p2 − p6|| + ||p3 − p5||
  EAR = ─────────────────────────────
               2 · ||p1 − p4||

Where p1…p6 are the 6 key landmark points around the eye:
  p1 = outer corner  p4 = inner corner
  p2,p3 = upper lid  p5,p6 = lower lid

When the eye is OPEN  → EAR ≈ 0.25 – 0.35
When the eye is CLOSED → EAR ≈ 0.0  – 0.15

A normal blink lasts 2–4 video frames.
Sustained closure > 20 frames indicates drowsiness.
"""

import numpy as np
from scipy.spatial import distance as dist
from typing import List, Tuple


def compute_ear(eye_landmarks: List[Tuple[float, float]]) -> float:
    """
    Compute the Eye Aspect Ratio for one eye.

    Args:
        eye_landmarks: List of 6 (x, y) tuples in order:
                       [outer_corner, upper1, upper2,
                        inner_corner, lower1, lower2]

    Returns:
        EAR value (float). Lower = more closed.
    """
    # Vertical distances: ||p2-p6|| and ||p3-p5||
    A = dist.euclidean(eye_landmarks[1], eye_landmarks[5])
    B = dist.euclidean(eye_landmarks[2], eye_landmarks[4])

    # Horizontal distance: ||p1-p4||
    C = dist.euclidean(eye_landmarks[0], eye_landmarks[3])

    # Guard against division by zero
    if C == 0:
        return 0.0

    ear = (A + B) / (2.0 * C)
    return round(float(ear), 4)


def average_ear(left_ear: float, right_ear: float) -> float:
    """Return the average of left and right EAR values."""
    return round((left_ear + right_ear) / 2.0, 4)


def extract_eye_coords(
    landmarks,
    indices: List[int],
    frame_width: int,
    frame_height: int
) -> List[Tuple[float, float]]:
    """
    Extract pixel coordinates for the given landmark indices.

    Args:
        landmarks: MediaPipe face mesh landmark object
        indices: List of landmark indices to extract
        frame_width: Frame width in pixels
        frame_height: Frame height in pixels

    Returns:
        List of (x, y) pixel coordinate tuples
    """
    coords = []
    for idx in indices:
        lm = landmarks.landmark[idx]
        x = int(lm.x * frame_width)
        y = int(lm.y * frame_height)
        coords.append((x, y))
    return coords


def get_eye_outline_coords(
    landmarks,
    indices: List[int],
    frame_width: int,
    frame_height: int
) -> np.ndarray:
    """
    Extract eye outline coordinates as a NumPy array for drawing.

    Returns:
        numpy array of shape (N, 1, 2) suitable for cv2.polylines
    """
    coords = extract_eye_coords(landmarks, indices, frame_width, frame_height)
    return np.array(coords, dtype=np.int32).reshape((-1, 1, 2))


class EARSmoother:
    """
    Applies a simple moving average to EAR values to reduce noise
    from jitter in facial landmark detection.
    """

    def __init__(self, window_size: int = 5):
        self.window_size = window_size
        self._history: List[float] = []

    def smooth(self, ear: float) -> float:
        """Add new EAR value and return smoothed result."""
        self._history.append(ear)
        if len(self._history) > self.window_size:
            self._history.pop(0)
        return round(float(np.mean(self._history)), 4)

    def reset(self):
        self._history.clear()

    @property
    def history(self) -> List[float]:
        return list(self._history)
