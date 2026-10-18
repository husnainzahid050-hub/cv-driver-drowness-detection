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
  Yawn (wide open)     → MAR > 0.5, held for ~1 second

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

    A wide-open mouth is NOT always a yawn — a shocked / surprised driver
    also opens the mouth. The two are told apart by:

      Cue                  Yawn                 Surprise ("😮")
      ───────────────────  ───────────────────  ───────────────────
      Opening speed        slow (~0.5–1 s)      sudden (few frames)
      Head                 tilts up / back      slightly forward
      Eyes                 lazy / half-closed   wide open

    Each cue votes (eyes count double). More yawn votes → yawn.
    On a tie it is a yawn unless the eyes are wide open.
    """

    def __init__(self):
        self.consec_frames = 0          # Consecutive frames mouth is open
        self.total_yawns = 0
        self.yawn_active = False
        self.surprise_active = False    # Mouth wide open, judged as surprise
        self._mar_history: List[float] = []
        self._smoothing_window = 5
        self._gap_frames = 0            # Consecutive frames mouth is closed

        # Yawn vs. surprise cues
        self._rise_frames = 0           # Frames the mouth has been opening
        self._onset_frames = None       # How fast this opening happened
        self._base_ear = None           # Driver's normal EAR (mouth closed)
        self._base_pitch = None         # Driver's normal pitch (mouth closed)
        self._open_ears: List[float] = []   # EAR while mouth wide open
        self._min_pitch = None          # Most "head up" pitch while open
        self._max_pitch = None          # Most "head forward" pitch while open
        self.last_cues = ""             # Last yawn/surprise decision, readable
        self._sudden_limit = float(Settings.YAWN_SUDDEN_ONSET_FRAMES)

    def update(self, mar: float, pitch: float = None, ear: float = None,
               fps: float = None) -> bool:
        """
        Update with new MAR value. Returns True if yawn is active.

        Frames count towards a yawn only while MAR >= MAR_THRESHOLD.
        The candidate yawn ends only after MAR stays below
        MAR_EXIT_THRESHOLD for more than YAWN_GAP_FRAMES frames.

        pitch (head pose, degrees) and ear (eye aspect ratio) are optional;
        when given they are used to reject surprise / shock mouth openings.
        fps (real camera FPS) scales the "sudden opening" frame limit,
        which is defined for Settings.TARGET_FPS.
        """
        self._sudden_limit = Settings.YAWN_SUDDEN_ONSET_FRAMES
        if fps and fps > 0:
            self._sudden_limit *= fps / Settings.TARGET_FPS

        # Smooth MAR
        self._mar_history.append(mar)
        if len(self._mar_history) > self._smoothing_window:
            self._mar_history.pop(0)
        smooth_mar = float(np.mean(self._mar_history))

        if self.consec_frames == 0:
            # Opening speed — uses raw MAR, smoothing would hide a sudden jump
            if mar < Settings.MAR_OPEN_START:
                self._rise_frames = 0
                self._onset_frames = None
            elif self._onset_frames is None:
                if mar >= Settings.MAR_THRESHOLD:
                    self._onset_frames = self._rise_frames
                else:
                    self._rise_frames += 1

            # Learn the driver's normal face while the mouth is closed
            if smooth_mar < Settings.MAR_EXIT_THRESHOLD:
                self._update_baseline(pitch, ear)

        if smooth_mar >= Settings.MAR_THRESHOLD:
            self.consec_frames += 1
            self._gap_frames = 0
            if ear is not None:
                self._open_ears.append(ear)
            if pitch is not None:
                self._min_pitch = pitch if self._min_pitch is None else min(self._min_pitch, pitch)
                self._max_pitch = pitch if self._max_pitch is None else max(self._max_pitch, pitch)
        elif smooth_mar >= Settings.MAR_EXIT_THRESHOLD:
            # Between thresholds: keep current state, don't count
            self._gap_frames = 0
        elif self.consec_frames > 0:
            self._gap_frames += 1
            if self._gap_frames > Settings.YAWN_GAP_FRAMES:
                self._end_opening(smooth_mar)

        # Re-checked every frame: a yawn's head tilt / squint may come late
        if self.consec_frames >= Settings.YAWN_CONSEC_FRAMES and not self.yawn_active:
            if self._looks_like_yawn():
                self.yawn_active = True
                self.surprise_active = False
                self.total_yawns += 1
                print(f"\n  😮  [Yawn #{self.total_yawns}] "
                      f"YAWN DETECTED | MAR={smooth_mar:.3f}")
                print(f"       {self.last_cues}")
            elif not self.surprise_active:
                self.surprise_active = True
                print(f"\n  [Mouth] Wide opening looks like SHOCK/SURPRISE, "
                      f"not a yawn | MAR={smooth_mar:.3f}")
                print(f"       {self.last_cues}")

        return self.yawn_active

    def _update_baseline(self, pitch: float, ear: float):
        """Slow running average of the driver's normal pitch and EAR."""
        alpha = 0.05
        if pitch is not None:
            self._base_pitch = pitch if self._base_pitch is None else \
                (1 - alpha) * self._base_pitch + alpha * pitch
        # Ignore blinks only — some drivers have naturally narrow eyes
        # (relaxed EAR near EAR_THRESHOLD), so keep those in the baseline
        if ear is not None and ear >= Settings.EAR_THRESHOLD * 0.7:
            self._base_ear = ear if self._base_ear is None else \
                (1 - alpha) * self._base_ear + alpha * ear

    def _looks_like_yawn(self) -> bool:
        """
        Vote between yawn and surprise using opening speed, head, eyes.
        Eyes are the strongest cue (2 votes) and also break ties:
        wide-open eyes never count as a yawn on a tie.
        Sets self.last_cues to a readable summary of what was seen.
        """
        yawn_votes = surprise_votes = 0
        speed = head = eyes = "?"

        # 1. Opening speed: yawn opens slowly, surprise suddenly
        if self._onset_frames is not None:
            if self._onset_frames <= self._sudden_limit:
                surprise_votes += 1
                speed = f"sudden({self._onset_frames}f)"
            else:
                yawn_votes += 1
                speed = f"slow({self._onset_frames}f)"

        # 2. Head: yawn tilts up/back (pitch decreases),
        #    surprise jerks slightly forward (pitch increases)
        if self._base_pitch is not None and self._min_pitch is not None:
            back = self._base_pitch - self._min_pitch
            fwd = self._max_pitch - self._base_pitch
            head = "still"
            if back >= Settings.YAWN_HEAD_TILT_DEG and back >= fwd:
                yawn_votes += 1
                head = f"back {back:.0f}deg"
            elif fwd >= Settings.SURPRISE_HEAD_FORWARD_DEG:
                surprise_votes += 1
                head = f"forward {fwd:.0f}deg"

        # 3. Eyes (strongest cue): wide open in surprise,
        #    lazy / half-closed in a drowsy yawn
        eyes_wide = False
        if self._base_ear and self._open_ears:
            ratio = float(np.mean(self._open_ears)) / self._base_ear
            eyes = "normal"
            if ratio > 1 + Settings.YAWN_EYE_CHANGE_RATIO:
                surprise_votes += 2
                eyes_wide = True
                eyes = f"wide x{ratio:.2f}"
            elif ratio < 1 - Settings.YAWN_EYE_CHANGE_RATIO:
                yawn_votes += 2
                eyes = f"sleepy x{ratio:.2f}"

        self.last_cues = (f"speed={speed} head={head} eyes={eyes} "
                          f"(yawn {yawn_votes} : surprise {surprise_votes})")

        if yawn_votes == surprise_votes:
            return not eyes_wide
        return yawn_votes > surprise_votes

    def _end_opening(self, smooth_mar: float):
        """Mouth has closed — clear the current yawn / surprise episode."""
        if self.yawn_active:
            print(f"  ✔  [Yawn] Mouth closed | MAR={smooth_mar:.3f}")
        self.yawn_active = False
        self.surprise_active = False
        self.consec_frames = 0
        self._gap_frames = 0
        self._rise_frames = 0
        self._onset_frames = None
        self._open_ears.clear()
        self._min_pitch = None
        self._max_pitch = None

    @property
    def smooth_mar(self) -> float:
        if not self._mar_history:
            return 0.0
        return round(float(np.mean(self._mar_history)), 4)

    def reset(self):
        self._end_opening(0.0)
        self._mar_history.clear()
        self._base_ear = None
        self._base_pitch = None
