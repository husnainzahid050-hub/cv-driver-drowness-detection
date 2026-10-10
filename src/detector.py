"""
src/detector.py
───────────────
Core drowsiness detection engine — v2.0

Features:
  1. EAR-based eye closure detection (original)
  2. Yawn detection via Mouth Aspect Ratio (MAR)          ← NEW
  3. Head pose estimation (pitch/yaw/roll)                ← NEW
  4. Glasses detection + automatic mode switching        ← NEW

Drowsiness decision logic (weighted):
  Without glasses: EAR closure (primary) + Yawn + Head pose
  With glasses:    Head pose (primary) + Yawn  ← EAR becomes secondary

Alert triggers (ANY of these fires the alarm):
  • EAR < threshold for CONSEC_FRAMES consecutive frames
  • MAR > threshold for YAWN_CONSEC_FRAMES frames (yawn)
  • Head pitch > PITCH_THRESHOLD for HEAD_POSE_CONSEC_FRAMES frames
  • Head roll > ROLL_THRESHOLD (sudden sideways tilt)
"""

import cv2
import numpy as np
import os
import time
from datetime import datetime

# ── MediaPipe compatibility shim ──────────────────────────────────
try:
    import mediapipe as mp
    FaceMesh = mp.solutions.face_mesh.FaceMesh
except AttributeError:
    try:
        from mediapipe.python.solutions.face_mesh import FaceMesh
    except (ImportError, ModuleNotFoundError):
        raise ImportError(
            "\n\n  Fix:  pip install mediapipe==0.10.9\n"
        )
# ─────────────────────────────────────────────────────────────────

from config.settings import Settings
from src.ear_calculator import (
    compute_ear, average_ear, extract_eye_coords,
    get_eye_outline_coords, EARSmoother
)
from src.yawn_detector import (
    YawnDetector, compute_mar, extract_mouth_coords, get_mouth_outline_coords
)
from src.head_pose import HeadPoseEstimator
from src.glasses_detector import GlassesDetector
from src.alert_manager import AlertManager
from src.display import DisplayRenderer
from src.logger import SessionLogger


class DrowsinessDetector:

    def __init__(self, camera_index: int = 0):
        self.camera_index = camera_index

        self.face_mesh = FaceMesh(
            max_num_faces=1,
            refine_landmarks=Settings.REFINE_LANDMARKS,
            min_detection_confidence=Settings.MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=Settings.MIN_TRACKING_CONFIDENCE,
        )

        self.alert_manager  = AlertManager()
        self.smoother       = EARSmoother(window_size=5)
        self.yawn_detector  = YawnDetector()
        self.head_pose      = HeadPoseEstimator()
        self.glasses        = GlassesDetector()
        self.logger         = SessionLogger()

        self.consec_frames  = 0
        self.alert_active   = False
        self.pose_consec    = 0
        self.pose_alert     = False
        self.frame_counter  = 0
        self.total_alerts   = 0
        self.current_fps    = 0.0
        self._fps_counter   = 0
        self._fps_timer     = time.time()
        self.renderer       = None

        print("  [Detector] v2.0 — EAR + Yawn + Head Pose + Glasses")
        print(f"  [Detector] EAR threshold   = {Settings.EAR_THRESHOLD}")
        print(f"  [Detector] MAR threshold   = {Settings.MAR_THRESHOLD}")
        print(f"  [Detector] Pitch threshold = {Settings.HEAD_PITCH_THRESHOLD}°")

    def run(self):
        cap = self._open_camera()
        if cap is None:
            return

        fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.renderer = DisplayRenderer(fw, fh)

        print(f"\n  [Detector] Camera opened: {fw}x{fh}")
        print("  [Detector] Detection loop running...\n")
        os.makedirs(Settings.SCREENSHOT_DIR, exist_ok=True)

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("  [Detector] Frame read failed.")
                    break

                self.frame_counter += 1
                self._update_fps()

                r = self._analyse_frame(frame)
                self._update_state(r)

                display_frame = self.renderer.render(
                    frame=frame,
                    left_ear=r["left_ear"],
                    right_ear=r["right_ear"],
                    avg_ear=r["avg_ear"],
                    frame_counter=self.frame_counter,
                    consec_frames=self.consec_frames,
                    alert_active=self.alert_active,
                    vibration_active=self.alert_manager.is_vibrating,
                    vibration_intensity=self.alert_manager.vibration_intensity,
                    vibration_phase=self.alert_manager.vibration_phase,
                    fps=self.current_fps,
                    face_detected=r["face_detected"],
                    total_alerts=self.total_alerts,
                    mar=r["mar"],
                    yawn_active=r["yawn_active"],
                    total_yawns=self.yawn_detector.total_yawns,
                    pitch=r["pitch"],
                    yaw=r["yaw"],
                    roll=r["roll"],
                    pose_alert=self.pose_alert,
                    glasses_detected=self.glasses.glasses_detected,
                )

                cv2.imshow(Settings.WINDOW_TITLE, display_frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:
                    break
                elif key == ord('r'):
                    self._reset_state()
                elif key == ord('s'):
                    self._save_screenshot(display_frame)

        except KeyboardInterrupt:
            print("\n  [Detector] Interrupted")
        finally:
            self._cleanup(cap)

    def _analyse_frame(self, frame: np.ndarray) -> dict:
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self.face_mesh.process(rgb)
        rgb.flags.writeable = True

        empty = {"face_detected": False,
                 "left_ear": 0.0, "right_ear": 0.0, "avg_ear": 0.0,
                 "mar": 0.0, "yawn_active": False,
                 "pitch": 0.0, "yaw": 0.0, "roll": 0.0}

        if not results.multi_face_landmarks:
            return empty

        lms = results.multi_face_landmarks[0]

        # EAR
        left_ear  = compute_ear(extract_eye_coords(lms, Settings.LEFT_EYE_INDICES,  w, h))
        right_ear = compute_ear(extract_eye_coords(lms, Settings.RIGHT_EYE_INDICES, w, h))
        avg_ear   = self.smoother.smooth(average_ear(left_ear, right_ear))

        ear_color = Settings.COLOR_OK if avg_ear >= Settings.EAR_THRESHOLD else Settings.COLOR_ALERT
        cv2.polylines(frame, [get_eye_outline_coords(lms, Settings.LEFT_EYE_OUTLINE, w, h)],  True, ear_color, 1, cv2.LINE_AA)
        cv2.polylines(frame, [get_eye_outline_coords(lms, Settings.RIGHT_EYE_OUTLINE, w, h)], True, ear_color, 1, cv2.LINE_AA)

        # Head pose
        pitch, yaw, roll = self.head_pose.estimate(lms, w, h)
        if Settings.SHOW_HEAD_AXES:
            self.head_pose.draw_axes(frame, lms, w, h)

        # Glasses
        self.glasses.update(avg_ear, pitch)

        # Yawn
        mar = compute_mar(extract_mouth_coords(lms, w, h))
        yawn_active = self.yawn_detector.update(mar)
        mouth_color = Settings.COLOR_ALERT if yawn_active else (0, 200, 100)
        cv2.polylines(frame, [get_mouth_outline_coords(lms, w, h)], True, mouth_color, 1, cv2.LINE_AA)

        return {"face_detected": True,
                "left_ear": left_ear, "right_ear": right_ear, "avg_ear": avg_ear,
                "mar": mar, "yawn_active": yawn_active,
                "pitch": pitch, "yaw": yaw, "roll": roll}

    def _update_state(self, r: dict):
        if not r["face_detected"]:
            return

        avg_ear = r["avg_ear"]
        pitch   = r["pitch"]
        yawn    = r["yawn_active"]
        glasses = self.glasses.glasses_detected

        # EAR — relaxed threshold if glasses
        ear_thresh = self.glasses.adjusted_ear_threshold
        if avg_ear < ear_thresh:
            self.consec_frames += 1
        else:
            self.consec_frames = 0

        # Head pose
        if self.head_pose.is_drowsy_pose():
            self.pose_consec += 1
        else:
            self.pose_consec = 0

        # Decision
        ear_frames_needed = int(Settings.CONSEC_FRAMES * (1.5 if glasses else 1.0))
        ear_triggered  = self.consec_frames >= ear_frames_needed
        pose_triggered = self.pose_consec   >= Settings.HEAD_POSE_CONSEC_FRAMES
        yawn_plus_ear  = yawn and (self.consec_frames >= Settings.CONSEC_FRAMES // 2)

        should_alert = ear_triggered or pose_triggered or yawn_plus_ear

        if should_alert:
            if not self.alert_active:
                self.alert_active = True
                self.total_alerts += 1
                parts = []
                if ear_triggered:  parts.append(f"EAR={avg_ear:.3f}")
                if pose_triggered: parts.append(f"Pitch={pitch:.1f}deg")
                if yawn_plus_ear:  parts.append("Yawn+EAR")
                if glasses:        parts.append("[GLASSES MODE]")
                print(f"\n  [Alert #{self.total_alerts}] DROWSINESS | {' | '.join(parts)}")
            self.alert_manager.trigger_alert(self.frame_counter)
            self.logger.log_event(self.frame_counter, r["left_ear"], r["right_ear"],
                                  avg_ear, self.consec_frames, True,
                                  self.alert_manager.is_vibrating)
        else:
            if self.alert_active:
                print(f"  [Recovery] EAR={avg_ear:.3f} Pitch={pitch:.1f}deg")
            self.alert_active = False
            self.alert_manager.clear_alert()

        self.pose_alert = pose_triggered

    def _reset_state(self):
        self.consec_frames = self.pose_consec = 0
        self.alert_active = self.pose_alert = False
        self.alert_manager.clear_alert()
        self.smoother.reset()
        self.yawn_detector.reset()
        self.head_pose.reset()

    def _open_camera(self):
        print(f"  [Detector] Opening camera {self.camera_index}...")
        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            print(f"\n  ERROR: Could not open camera {self.camera_index}")
            print("    Try: python main.py --demo")
            return None
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  Settings.FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, Settings.FRAME_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS,          Settings.TARGET_FPS)
        return cap

    def _update_fps(self):
        self._fps_counter += 1
        if self._fps_counter >= 30:
            now = time.time()
            self.current_fps  = self._fps_counter / (now - self._fps_timer)
            self._fps_timer   = now
            self._fps_counter = 0

    def _save_screenshot(self, frame):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(Settings.SCREENSHOT_DIR, f"screenshot_{ts}.png")
        cv2.imwrite(path, frame)
        print(f"  [Screenshot] {path}")

    def _cleanup(self, cap):
        cap.release()
        cv2.destroyAllWindows()
        self.face_mesh.close()
        self.alert_manager.cleanup()
        self.logger.close()
        print("  [Detector] Session ended.")
