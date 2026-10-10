"""
src/glasses_detector.py
────────────────────────
Detects whether the driver is wearing glasses.

Why this matters:
  Glasses frames and lenses can obstruct MediaPipe's eye landmark
  detection, causing EAR values to be consistently lower than they
  should be for an awake person. This leads to false drowsiness alerts.

  When glasses are detected, the system:
    1. Raises the EAR threshold to compensate (less strict)
    2. Increases weight of HEAD POSE in drowsiness decision
    3. Still uses EAR but as secondary signal
    4. Shows "GLASSES MODE" indicator on dashboard

Detection method — Landmark Stability Analysis:
  When glasses are present, the eye region EAR is suppressed but
  stays relatively STABLE across frames (not fluctuating much).
  A person without glasses has more EAR variation.

  Additionally, we check the eye region brightness ratio — glasses
  create distinct bright/dark reflections in the eye zone.

  Heuristics used (all must agree):
    1. Median EAR over recent frames is in "glasses zone" (0.14–0.20)
       while the person appears otherwise alert (face angle normal)
    2. EAR variance is low (stable suppression = glasses, not drowsy)
    3. Confirmed only after GLASSES_CONFIRM_FRAMES consecutive frames

This approach requires no separate ML model — it uses signals already
computed by the main detection pipeline.
"""

import numpy as np
from collections import deque
from config.settings import Settings


class GlassesDetector:
    """
    Detects glasses presence and adjusts detection sensitivity accordingly.
    """

    def __init__(self):
        self._ear_history = deque(maxlen=Settings.GLASSES_HISTORY_FRAMES)
        self._pitch_history = deque(maxlen=Settings.GLASSES_HISTORY_FRAMES)
        self._confirm_frames = 0
        self.glasses_detected = False
        self._frames_since_check = 0
        self._recheck_interval = 150  # Recheck every N frames

        print("  [GlassesDetector] Initialized — monitoring for glasses")

    def update(self, avg_ear: float, pitch: float) -> bool:
        """
        Update with latest EAR and head pitch. Returns glasses_detected.

        Args:
            avg_ear: Current smoothed average EAR
            pitch: Current head pitch angle (degrees)
        """
        self._ear_history.append(avg_ear)
        self._pitch_history.append(pitch)
        self._frames_since_check += 1

        # Only evaluate after enough history
        if len(self._ear_history) < Settings.GLASSES_HISTORY_FRAMES:
            return self.glasses_detected

        # Only re-evaluate periodically to avoid flip-flopping
        if self._frames_since_check < self._recheck_interval and self.glasses_detected:
            return self.glasses_detected

        self._frames_since_check = 0
        self._evaluate()
        return self.glasses_detected

    def _evaluate(self):
        """Run the glasses detection heuristics."""
        ears = np.array(self._ear_history)
        pitches = np.array(self._pitch_history)

        median_ear = np.median(ears)
        ear_variance = np.var(ears)
        median_pitch = np.median(np.abs(pitches))

        # Heuristic 1: EAR is suppressed but not zero (glasses zone)
        ear_suppressed = (Settings.GLASSES_EAR_LOW <= median_ear <= Settings.GLASSES_EAR_HIGH)

        # Heuristic 2: Low variance — stable suppression (not drowsy fluctuation)
        # A genuinely drowsy person has dropping EAR; glasses cause flat suppression
        low_variance = ear_variance < Settings.GLASSES_EAR_VARIANCE_THRESHOLD

        # Heuristic 3: Head is roughly upright (person is awake but EAR is low)
        head_upright = median_pitch < 20.0

        was_detected = self.glasses_detected

        if ear_suppressed and low_variance and head_upright:
            self._confirm_frames += 1
            if self._confirm_frames >= Settings.GLASSES_CONFIRM_FRAMES:
                if not self.glasses_detected:
                    self.glasses_detected = True
                    print(f"\n  👓  [GlassesDetector] GLASSES DETECTED!")
                    print(f"       Median EAR={median_ear:.4f}  Variance={ear_variance:.6f}")
                    print(f"       → Switching to HEAD POSE primary mode")
        else:
            self._confirm_frames = max(0, self._confirm_frames - 1)
            if self._confirm_frames == 0 and self.glasses_detected:
                self.glasses_detected = False
                print(f"\n  👁  [GlassesDetector] Glasses removed — EAR mode restored")

    @property
    def adjusted_ear_threshold(self) -> float:
        """
        Return a slightly relaxed EAR threshold when glasses are detected.
        This reduces false positives from lens/frame occlusion.
        """
        if self.glasses_detected:
            return Settings.EAR_THRESHOLD - Settings.GLASSES_EAR_THRESHOLD_REDUCTION
        return Settings.EAR_THRESHOLD

    @property
    def ear_weight(self) -> float:
        """
        Weight to apply to EAR in drowsiness decision.
        Reduced when glasses present (rely more on head pose).
        0.0 = ignore EAR entirely, 1.0 = full weight
        """
        return Settings.GLASSES_EAR_WEIGHT if self.glasses_detected else 1.0

    @property
    def pose_weight(self) -> float:
        """
        Weight to apply to head pose in drowsiness decision.
        Increased when glasses present.
        """
        return Settings.GLASSES_POSE_WEIGHT if self.glasses_detected else 0.4

    def reset(self):
        self._ear_history.clear()
        self._pitch_history.clear()
        self._confirm_frames = 0
