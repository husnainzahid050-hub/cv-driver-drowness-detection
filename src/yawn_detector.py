"""
src/yawn_detector.py
─────────────────────
Yawn detection using Mouth Aspect Ratio (MAR).

Yawning is one of the earliest and most reliable signs of drowsiness.
It is caused by the brain trying to increase oxygen intake when tired.

How MAR works (same principle as EAR for eyes):

         ||p2 − p6|| + ||p3 − p5||
  MAR = ─────────────────────────────
               2 · ||p1 − p4||

  p1 = left corner     p4 = right corner   (horizontal span)
  p2 = upper outer     p6 = lower outer    (vertical span)
  p3 = upper inner     p5 = lower inner

  Normal closed mouth  → MAR ≈ 0.0 – 0.3
  Normal speech        → MAR ≈ 0.3 – 0.5
  Yawn (wide open)     → MAR > 0.6

MediaPipe Face Mesh landmark indices used:
  Left corner  : 61
  Upper outer  : 82
  Upper inner  : 13
  Right corner : 291
  Lower inner  : 14
  Lower outer  : 87
"""

import numpy as np
from scipy.spatial import distance as dist
from typing import List, Tuple
from config.settings import Settings


# Mouth landmark indices (MediaPipe Face Mesh)
MOUTH_INDICES = [61, 82, 13, 291, 14, 87]

# Full mouth outline for visualization
MOUTH_OUTLINE_INDICES = [
    61, 185, 40, 39, 37, 0, 267,
    269, 270, 409, 291, 375, 321,
    405, 314, 17, 84, 181, 91, 146, 61
]


def compute_mar(mouth_landmarks: List[Tuple[float, float]]) -> float:
    """
    Compute the Mouth Aspect Ratio for yawn detection.

    Args:
        mouth_landmarks: List of 6 (x, y) tuples:
                         [left_corner, upper_outer, upper_inner,
                          right_corner, lower_inner, lower_outer]

    Returns:
        MAR value. Higher = more open. Normal speech ~0.3, yawn > 0.6
    """
    # Vertical: ||p2-p6|| and ||p3-p5||
    A = dist.euclidean(mouth_landmarks[1], mouth_landmarks[5])
    B = dist.euclidean(mouth_landmarks[2], mouth_landmarks[4])

    # Horizontal: ||p1-p4||
    C = dist.euclidean(mouth_landmarks[0], mouth_landmarks[3])

    if C == 0:
        return 0.0

    mar = (A + B) / (2.0 * C)
    return round(float(mar), 4)


def extract_mouth_coords(
    landmarks,
    frame_width: int,
    frame_height: int
) -> List[Tuple[float, float]]:
    """Extract the 6 mouth landmark pixel coordinates."""
    coords = []
    for idx in MOUTH_INDICES:
        lm = landmarks.landmark[idx]
        x = int(lm.x * frame_width)
        y = int(lm.y * frame_height)
        coords.append((x, y))
    return coords


def get_mouth_outline_coords(landmarks, frame_width: int, frame_height: int):
    """Return mouth outline as numpy array for cv2.polylines."""
    coords = []
    for idx in MOUTH_OUTLINE_INDICES:
        lm = landmarks.landmark[idx]
        x = int(lm.x * frame_width)
        y = int(lm.y * frame_height)
        coords.append((x, y))
    return np.array(coords, dtype=np.int32).reshape((-1, 1, 2))


class YawnDetector:
    """
    Tracks mouth opening over time and detects yawn events.
    Uses a moving average to reduce noise.
    """

    def __init__(self):
        self.consec_frames = 0          # Consecutive frames mouth is open
        self.total_yawns = 0
        self.yawn_active = False
        self._mar_history: List[float] = []
        self._smoothing_window = 5

    def update(self, mar: float) -> bool:
        """
        Update with new MAR value. Returns True if yawn is active.
        """
        # Smooth MAR
        self._mar_history.append(mar)
        if len(self._mar_history) > self._smoothing_window:
            self._mar_history.pop(0)
        smooth_mar = float(np.mean(self._mar_history))

        if smooth_mar >= Settings.MAR_THRESHOLD:
            self.consec_frames += 1
            if self.consec_frames >= Settings.YAWN_CONSEC_FRAMES:
                if not self.yawn_active:
                    self.yawn_active = True
                    self.total_yawns += 1
                    print(f"\n  😮  [Yawn #{self.total_yawns}] "
                          f"YAWN DETECTED | MAR={smooth_mar:.3f}")
                return True
        else:
            if self.yawn_active:
                print(f"  ✔  [Yawn] Mouth closed | MAR={smooth_mar:.3f}")
            self.yawn_active = False
            self.consec_frames = 0

        return False

    @property
    def smooth_mar(self) -> float:
        if not self._mar_history:
            return 0.0
        return round(float(np.mean(self._mar_history)), 4)

    def reset(self):
        self.consec_frames = 0
        self.yawn_active = False
        self._mar_history.clear()
