"""
src/display.py
──────────────
Handles all on-screen rendering for the drowsiness detection system.

Layout:
┌─────────────────────────────┬──────────────────────┐
│                             │   DASHBOARD PANEL    │
│   CAMERA FEED               │                      │
│   (with overlays)           │  EAR Graph           │
│                             │  Status indicators   │
│                             │  Seat Vibration Sim  │
│                             │  Session stats       │
└─────────────────────────────┴──────────────────────┘
"""

import cv2
import numpy as np
import time
from typing import Optional, Tuple, List
from config.settings import Settings


class DisplayRenderer:
    """Renders all UI elements onto the video frame + dashboard panel."""

    def __init__(self, frame_width: int, frame_height: int):
        self.fw = frame_width
        self.fh = frame_height
        self.dw = Settings.DASHBOARD_WIDTH  # Dashboard panel width
        self._ear_history: List[float] = []
        self._max_history = 120  # Keep last N EAR values for graph
        self._flash_state = 0   # For flashing alerts
        self._flash_counter = 0
        self._start_time = time.time()

        # Fonts
        self.FONT = cv2.FONT_HERSHEY_SIMPLEX
        self.FONT_BOLD = cv2.FONT_HERSHEY_DUPLEX

        # Pre-compute total window width
        self.total_width = self.fw + self.dw

    # ──────────────────────────────────────────────────────────────
    #  Main render method
    # ──────────────────────────────────────────────────────────────

    def render(
        self,
        frame: np.ndarray,
        left_ear: float,
        right_ear: float,
        avg_ear: float,
        frame_counter: int,
        consec_frames: int,
        alert_active: bool,
        vibration_active: bool,
        vibration_intensity: int,
        vibration_phase: int,
        fps: float,
        face_detected: bool,
        total_alerts: int,
        # New v2.0 fields
        mar: float = 0.0,
        yawn_active: bool = False,
        total_yawns: int = 0,
        pitch: float = 0.0,
        yaw: float = 0.0,
        roll: float = 0.0,
        pose_alert: bool = False,
        glasses_detected: bool = False,
    ) -> np.ndarray:
        """
        Compose the full display: camera feed (with overlays) + dashboard.
        Returns the final composite image to display.
        """
        self._flash_counter += 1

        if face_detected:
            self._ear_history.append(avg_ear)
            if len(self._ear_history) > self._max_history:
                self._ear_history.pop(0)

        annotated = frame.copy()
        annotated = self._draw_camera_overlays(
            annotated, avg_ear, consec_frames, alert_active, face_detected
        )

        if alert_active:
            annotated = self._draw_alert_border(annotated)

        dashboard = self._build_dashboard(
            left_ear, right_ear, avg_ear, frame_counter,
            consec_frames, alert_active, vibration_active,
            vibration_intensity, vibration_phase, fps,
            face_detected, total_alerts,
            mar, yawn_active, total_yawns,
            pitch, yaw, roll, pose_alert, glasses_detected,
        )

        combined = np.hstack([annotated, dashboard])
        return combined

    # ──────────────────────────────────────────────────────────────
    #  Camera frame overlays
    # ──────────────────────────────────────────────────────────────

    def _draw_camera_overlays(
        self, frame, avg_ear, consec_frames, alert_active, face_detected
    ):
        h, w = frame.shape[:2]

        # Semi-transparent top bar
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 52), Settings.COLOR_DARK_BG, -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

        # Title
        cv2.putText(frame, "DRIVER DROWSINESS DETECTION",
                    (10, 20), self.FONT_BOLD, 0.52,
                    Settings.COLOR_ACCENT, 1, cv2.LINE_AA)
        if not face_detected:
            # No face warning
            self._draw_centered_text(
                frame, "⚠  NO FACE DETECTED", h // 2,
                Settings.COLOR_WARNING, scale=0.9
            )
            return frame

        # Status bar at bottom
        overlay2 = frame.copy()
        cv2.rectangle(overlay2, (0, h - 55), (w, h), Settings.COLOR_DARK_BG, -1)
        cv2.addWeighted(overlay2, 0.8, frame, 0.2, 0, frame)

        # EAR value
        ear_color = Settings.COLOR_OK if avg_ear >= Settings.EAR_THRESHOLD else Settings.COLOR_ALERT
        cv2.putText(frame, f"EAR: {avg_ear:.3f}",
                    (10, h - 35), self.FONT_BOLD, 0.65,
                    ear_color, 2, cv2.LINE_AA)

        # Progress bar: consecutive frames / threshold
        bar_x, bar_y = 10, h - 18
        bar_w = w - 20
        bar_h = 10
        # Background
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h),
                      (60, 60, 60), -1)
        # Fill
        fill_ratio = min(consec_frames / Settings.CONSEC_FRAMES, 1.0)
        fill_w = int(bar_w * fill_ratio)
        fill_color = self._get_bar_color(fill_ratio)
        if fill_w > 0:
            cv2.rectangle(frame, (bar_x, bar_y),
                          (bar_x + fill_w, bar_y + bar_h), fill_color, -1)
        # Label
        cv2.putText(frame, f"Drowsy: {consec_frames}/{Settings.CONSEC_FRAMES}",
                    (bar_x + bar_w // 2 - 60, bar_y - 3),
                    self.FONT, 0.35, (200, 200, 200), 1, cv2.LINE_AA)

        # DROWSY ALERT banner
        if alert_active:
            flash = (self._flash_counter // 8) % 2 == 0
            if flash:
                banner_overlay = frame.copy()
                cv2.rectangle(banner_overlay, (0, h // 2 - 45), (w, h // 2 + 45),
                               (0, 0, 180), -1)
                cv2.addWeighted(banner_overlay, 0.6, frame, 0.4, 0, frame)
                cv2.putText(frame, "⚠  DROWSINESS DETECTED!",
                            (w // 2 - 160, h // 2 + 8),
                            self.FONT_BOLD, 0.85,
                            Settings.COLOR_WHITE, 2, cv2.LINE_AA)

        return frame

    def _draw_alert_border(self, frame: np.ndarray) -> np.ndarray:
        """Draw flashing red border around the frame when alert is active."""
        flash = (self._flash_counter // 6) % 2 == 0
        if flash:
            thickness = 8
            h, w = frame.shape[:2]
            cv2.rectangle(frame, (0, 0), (w - 1, h - 1),
                          Settings.COLOR_ALERT, thickness)
        return frame

    def _draw_centered_text(self, frame, text, y, color, scale=0.7):
        h, w = frame.shape[:2]
        (tw, th), _ = cv2.getTextSize(text, self.FONT_BOLD, scale, 2)
        x = (w - tw) // 2
        cv2.putText(frame, text, (x, y), self.FONT_BOLD, scale, color, 2, cv2.LINE_AA)

    def _get_head_pose_status(self, pitch: float, roll: float, pose_alert: bool) -> str:
        if pose_alert:
            if pitch > Settings.HEAD_PITCH_THRESHOLD:
                return "NODDING !"
            return "TILTING !"
        return "Normal"

    def _get_bar_color(self, ratio: float) -> tuple:
        if ratio < 0.5:
            return Settings.COLOR_OK
        elif ratio < 0.85:
            return Settings.COLOR_WARNING
        else:
            return Settings.COLOR_ALERT

    # ──────────────────────────────────────────────────────────────
    #  Dashboard panel
    # ──────────────────────────────────────────────────────────────

    def _build_dashboard(
        self, left_ear, right_ear, avg_ear, frame_counter,
        consec_frames, alert_active, vibration_active,
        vibration_intensity, vibration_phase, fps,
        face_detected, total_alerts,
        mar=0.0, yawn_active=False, total_yawns=0,
        pitch=0.0, yaw=0.0, roll=0.0,
        pose_alert=False, glasses_detected=False,
    ) -> np.ndarray:
        """Build the right-side dashboard panel."""
        panel = np.zeros((self.fh, self.dw, 3), dtype=np.uint8)
        panel[:] = Settings.COLOR_DARK_BG

        y = 0  # Current Y cursor

        # Header
        cv2.rectangle(panel, (0, 0), (self.dw, 48), (25, 25, 45), -1)
        cv2.putText(panel, "DASHBOARD", (10, 20),
                    self.FONT_BOLD, 0.55, Settings.COLOR_ACCENT, 1, cv2.LINE_AA)
        cv2.putText(panel, "Real-Time Monitor", (10, 40),
                    self.FONT, 0.35, (140, 140, 140), 1, cv2.LINE_AA)
        y = 58

        # System status
        y = self._draw_section(panel, y, "SYSTEM STATUS")
        # Glasses badge
        if glasses_detected:
            cv2.rectangle(panel, (self.dw - 90, y - 20), (self.dw - 4, y - 5),
                          (40, 100, 40), -1)
            cv2.putText(panel, "GLASSES MODE", (self.dw - 88, y - 8),
                        self.FONT, 0.28, (100, 255, 100), 1, cv2.LINE_AA)
        status_text  = "DROWSY  !" if alert_active else "AWAKE  OK"
        status_color = Settings.COLOR_ALERT if alert_active else Settings.COLOR_OK
        y = self._draw_kv(panel, y, "State", status_text, status_color)
        face_text  = "Detected" if face_detected else "Not Found"
        face_color = Settings.COLOR_OK if face_detected else Settings.COLOR_WARNING
        y = self._draw_kv(panel, y, "Face", face_text, face_color)
        y = self._draw_kv(panel, y, "FPS", f"{fps:.1f}", Settings.COLOR_WHITE)
        y += 4

        # EAR Readings
        y = self._draw_section(panel, y, "EAR READINGS")
        lc = Settings.COLOR_OK if left_ear  >= Settings.EAR_THRESHOLD else Settings.COLOR_ALERT
        rc = Settings.COLOR_OK if right_ear >= Settings.EAR_THRESHOLD else Settings.COLOR_ALERT
        ac = Settings.COLOR_OK if avg_ear   >= Settings.EAR_THRESHOLD else Settings.COLOR_ALERT
        y = self._draw_kv(panel, y, "Left Eye",  f"{left_ear:.4f}",  lc)
        y = self._draw_kv(panel, y, "Right Eye", f"{right_ear:.4f}", rc)
        y = self._draw_kv(panel, y, "Average",   f"{avg_ear:.4f}",   ac)
        y = self._draw_kv(panel, y, "Threshold", f"{Settings.EAR_THRESHOLD:.2f}", (160, 160, 160))
        y += 4

        # Drowsy frame counter
        y = self._draw_section(panel, y, "DROWSY COUNTER")
        y = self._draw_progress_bar(panel, y, consec_frames, Settings.CONSEC_FRAMES)
        y += 2

        # ── NEW: Yawn detection panel ───────────────────────────
        y = self._draw_section(panel, y, "YAWN DETECTION")
        yawn_color = Settings.COLOR_ALERT if yawn_active else Settings.COLOR_OK
        yawn_text  = "YAWNING !" if yawn_active else "Normal"
        y = self._draw_kv(panel, y, "Mouth", yawn_text, yawn_color)
        mar_color = Settings.COLOR_ALERT if mar >= Settings.MAR_THRESHOLD else (160, 160, 160)
        y = self._draw_kv(panel, y, "MAR", f"{mar:.4f}", mar_color)
        y = self._draw_kv(panel, y, "Threshold", f"{Settings.MAR_THRESHOLD:.2f}", (160, 160, 160))
        y = self._draw_kv(panel, y, "Total Yawns", str(total_yawns),
                          Settings.COLOR_WARNING if total_yawns > 0 else Settings.COLOR_WHITE)
        y += 4

        # ── NEW: Head pose panel ────────────────────────────────
        y = self._draw_section(panel, y, "HEAD POSE")
        pose_status = self._get_head_pose_status(pitch, roll, pose_alert)
        pose_scol   = Settings.COLOR_ALERT if pose_alert else Settings.COLOR_OK
        y = self._draw_kv(panel, y, "Status", pose_status, pose_scol)
        pitch_col = Settings.COLOR_ALERT if pitch > Settings.HEAD_PITCH_THRESHOLD \
            else Settings.COLOR_WHITE
        roll_col  = Settings.COLOR_ALERT if abs(roll) > Settings.HEAD_ROLL_THRESHOLD \
            else Settings.COLOR_WHITE
        y = self._draw_kv(panel, y, "Pitch", f"{pitch:+.1f} deg", pitch_col)
        y = self._draw_kv(panel, y, "Yaw",   f"{yaw:+.1f} deg",   Settings.COLOR_WHITE)
        y = self._draw_kv(panel, y, "Roll",  f"{roll:+.1f} deg",  roll_col)
        y += 4

        # EAR Trend graph
        y = self._draw_section(panel, y, "EAR TREND")
        y = self._draw_ear_graph(panel, y)
        y += 4

        # ──────────────────────────────────────────────────────────
        #  SEAT VIBRATION SIMULATION PANEL
        # ──────────────────────────────────────────────────────────
        y = self._draw_vibration_panel(
            panel, y, vibration_active, vibration_intensity, vibration_phase
        )
        y += 4

        # Session stats at bottom
        elapsed = time.time() - self._start_time
        elapsed_str = time.strftime("%H:%M:%S", time.gmtime(elapsed))
        y = self._draw_section(panel, y, "SESSION STATS")
        y = self._draw_kv(panel, y, "Duration", elapsed_str, Settings.COLOR_WHITE)
        y = self._draw_kv(panel, y, "Frames", str(frame_counter), Settings.COLOR_WHITE)
        y = self._draw_kv(panel, y, "Total Alerts", str(total_alerts),
                          Settings.COLOR_ALERT if total_alerts > 0 else Settings.COLOR_WHITE)

        # Controls hint at very bottom
        hint_y = self.fh - 28
        cv2.rectangle(panel, (0, hint_y - 4), (self.dw, self.fh), (15, 15, 30), -1)
        cv2.putText(panel, "Q=Quit  R=Reset  S=Screenshot",
                    (6, self.fh - 12), self.FONT, 0.30,
                    (100, 100, 100), 1, cv2.LINE_AA)

        return panel

    def _draw_section(self, panel, y, title) -> int:
        """Draw a section header."""
        cv2.rectangle(panel, (0, y), (self.dw, y + 20), Settings.COLOR_PANEL, -1)
        cv2.putText(panel, title, (8, y + 14), self.FONT_BOLD, 0.38,
                    Settings.COLOR_ACCENT, 1, cv2.LINE_AA)
        return y + 22

    def _draw_kv(self, panel, y, key, value, val_color) -> int:
        """Draw a key-value row."""
        cv2.putText(panel, key + ":", (10, y + 12), self.FONT, 0.36,
                    (160, 160, 160), 1, cv2.LINE_AA)
        cv2.putText(panel, value, (self.dw - len(value) * 8 - 10, y + 12),
                    self.FONT_BOLD, 0.38, val_color, 1, cv2.LINE_AA)
        return y + 17

    def _draw_progress_bar(self, panel, y, current, maximum) -> int:
        """Draw a horizontal progress bar."""
        bx, bh = 10, 14
        bw = self.dw - 20
        ratio = min(current / max(maximum, 1), 1.0)
        cv2.rectangle(panel, (bx, y), (bx + bw, y + bh), (50, 50, 50), -1)
        fill_w = int(bw * ratio)
        color = self._get_bar_color(ratio)
        if fill_w > 0:
            cv2.rectangle(panel, (bx, y), (bx + fill_w, y + bh), color, -1)
        label = f"{current} / {maximum}"
        cv2.putText(panel, label, (bx + bw // 2 - 20, y + bh - 2),
                    self.FONT, 0.32, (220, 220, 220), 1, cv2.LINE_AA)
        return y + bh + 4

    def _draw_ear_graph(self, panel, y) -> int:
        """Draw a mini EAR trend line graph."""
        gh = 50  # graph height
        gw = self.dw - 20
        gx, gy = 10, y

        # Background
        cv2.rectangle(panel, (gx, gy), (gx + gw, gy + gh), (30, 30, 50), -1)
        cv2.rectangle(panel, (gx, gy), (gx + gw, gy + gh), (70, 70, 90), 1)

        # Threshold line
        threshold_y = gy + gh - int((Settings.EAR_THRESHOLD / 0.4) * gh)
        cv2.line(panel, (gx, threshold_y), (gx + gw, threshold_y),
                 (100, 100, 180), 1)
        cv2.putText(panel, "thr", (gx + gw - 22, threshold_y - 2),
                    self.FONT, 0.28, (100, 100, 180), 1)

        # EAR line
        if len(self._ear_history) >= 2:
            n = min(len(self._ear_history), gw)
            history = self._ear_history[-n:]
            xs = np.linspace(gx, gx + gw, len(history), dtype=int)
            for i in range(1, len(history)):
                ey1 = gy + gh - int(min(history[i - 1] / 0.4, 1.0) * gh)
                ey2 = gy + gh - int(min(history[i] / 0.4, 1.0) * gh)
                color = Settings.COLOR_OK if history[i] >= Settings.EAR_THRESHOLD \
                    else Settings.COLOR_ALERT
                cv2.line(panel, (xs[i - 1], ey1), (xs[i], ey2), color, 1)

        return y + gh + 2

    # ──────────────────────────────────────────────────────────────
    #  Seat Vibration Simulation Panel (DEMO FEATURE)
    # ──────────────────────────────────────────────────────────────

    def _draw_vibration_panel(
        self, panel, y, active, intensity, phase
    ) -> int:
        """
        Renders the seat vibration simulation indicator.

        Shows:
        - "SEAT VIBRATION SIM" section header
        - Animated seat icon (shakes when active)
        - Intensity bars (pulsing when active)
        - Status text
        - DEMO disclaimer
        """
        # Section header
        y = self._draw_section(panel, y, "SEAT VIBRATION SIM")

        panel_h = 90
        px, pw = 8, self.dw - 16

        # Background box
        bg_color = (50, 10, 10) if active else (25, 25, 40)
        cv2.rectangle(panel, (px, y), (px + pw, y + panel_h), bg_color, -1)
        border_color = Settings.COLOR_ALERT if active else (60, 60, 80)
        cv2.rectangle(panel, (px, y), (px + pw, y + panel_h), border_color, 1)

        # Seat icon (simplified car seat shape)
        seat_cx = px + 30
        seat_cy = y + 42
        shake = 0
        if active:
            # Shake offset: oscillates based on vibration phase
            shake = [0, 2, 0, -2, 1, -1, 2, 0][phase % 8]

        # Seat back (rectangle)
        cv2.rectangle(panel,
                      (seat_cx - 14 + shake, seat_cy - 22),
                      (seat_cx + 14 + shake, seat_cy + 2),
                      (100, 100, 130), -1)
        # Seat cushion
        cv2.rectangle(panel,
                      (seat_cx - 16 + shake, seat_cy + 2),
                      (seat_cx + 16 + shake, seat_cy + 16),
                      (100, 100, 130), -1)
        # Seat legs
        cv2.line(panel,
                 (seat_cx - 10 + shake, seat_cy + 16),
                 (seat_cx - 10 + shake, seat_cy + 26),
                 (80, 80, 100), 2)
        cv2.line(panel,
                 (seat_cx + 10 + shake, seat_cy + 16),
                 (seat_cx + 10 + shake, seat_cy + 26),
                 (80, 80, 100), 2)

        # Vibration wave lines (animated)
        if active:
            wave_x = seat_cx + 18
            for i, offset_y in enumerate([-12, -4, 4, 12]):
                amp = 3 + (phase + i) % 4
                for dx in range(0, 20, 4):
                    wave_y = seat_cy + offset_y + int(amp * np.sin(dx * 0.5 + phase))
                    cv2.circle(panel, (wave_x + dx, wave_y),
                               1, Settings.COLOR_ALERT, -1)

        # Intensity bars
        bars_x = px + 65
        bars_y = y + 15
        bar_w = 10
        bar_spacing = 14
        max_bars = 10
        for i in range(max_bars):
            bar_h_val = 6 + i * 2
            bx = bars_x + i * bar_spacing
            by = bars_y + (20 - bar_h_val)
            # Show filled bars up to intensity level
            if active and i < intensity:
                # Pulse: alternate brightness with phase
                brightness = 255 if (i + phase) % 3 != 0 else 160
                bar_color = (0, 0, brightness)
            else:
                bar_color = (40, 40, 60)
            cv2.rectangle(panel, (bx, by), (bx + bar_w, by + bar_h_val),
                          bar_color, -1)

        # Status text
        if active:
            status = "VIBRATING  [ACTIVE]"
            scol = Settings.COLOR_ALERT
        else:
            status = "Standby"
            scol = (100, 100, 100)
        cv2.putText(panel, status, (px + 6, y + 52),
                    self.FONT_BOLD, 0.38, scol, 1, cv2.LINE_AA)

        # Pattern display when active
        if active:
            cv2.putText(panel,
                        f"Pattern: {Settings.VIBRATION_PATTERN[:3]}...",
                        (px + 6, y + 67),
                        self.FONT, 0.30, (150, 100, 100), 1, cv2.LINE_AA)

        # DEMO disclaimer
        cv2.putText(panel, "[DEMO - No hardware]",
                    (px + 6, y + panel_h - 5),
                    self.FONT, 0.28, (80, 80, 80), 1, cv2.LINE_AA)

        return y + panel_h + 4
