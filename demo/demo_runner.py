"""
demo/demo_runner.py
────────────────────
Demo mode — runs the FULL drowsiness detection pipeline WITHOUT a webcam.

What the animated cartoon face now shows:
  ① Eyes open / closing / closed          → EAR detection
  ② Mouth opening wide for yawning        → MAR / Yawn detection  ← NEW
  ③ Head nodding forward (pitch)          → Head pose detection   ← NEW
  ④ Head tilting sideways (roll)          → Head pose detection   ← NEW

The scenario cycles through a realistic drowsy-driving sequence:
  Phase 1  — Normal driving         (eyes open, head straight, mouth closed)
  Phase 2  — Getting tired          (eyes slightly drooping)
  Phase 3  — Yawning                (mouth opens wide)            ← NEW
  Phase 4  — Falling asleep         (eyes closing, head nodding)  ← NEW
  Phase 5  — Alert fires + recovery (eyes open, head lifts)
  Phase 6  — Head tilt event        (head rolling sideways)       ← NEW
  Phase 7  — Second yawn + drowsy   (combined signal)             ← NEW
  Phase 8  — Back to normal
  ... repeats

All signals (EAR, MAR, pitch, roll) are fed into the REAL detection
state machine — the same one used with a live webcam.
"""

import cv2
import numpy as np
import time
import os
import sys
import math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config.settings import Settings
from src.alert_manager import AlertManager
from src.display import DisplayRenderer
from src.logger import SessionLogger
from src.ear_calculator import EARSmoother
from src.yawn_detector import YawnDetector
from src.head_pose import HeadPoseEstimator
from src.glasses_detector import GlassesDetector


# ══════════════════════════════════════════════════════════════════
#  DEMO SCENARIO
#  Each entry: (duration_sec, ear, mar, pitch_deg, roll_deg, label)
#
#  ear   : 0.0 (fully closed) → 0.35 (fully open)
#  mar   : 0.0 (closed mouth) → 0.80 (wide yawn)
#  pitch : 0° (head straight) → +40° (head nodding forward)
#  roll  : 0° (head straight) → ±35° (head tilting sideways)
# ══════════════════════════════════════════════════════════════════

SCENARIO = [
    # (secs,  ear,   mar,   pitch,  roll,   label)
    (5.0,  0.29,  0.05,   2.0,    1.0,  "Normal driving — fully awake"),
    (3.0,  0.22,  0.10,   8.0,    2.0,  "Getting tired — eyes drooping"),
    (3.5,  0.19,  0.68,  -6.0,    1.0,  "YAWNING — head tilts back, eyes squint"),
    (3.0,  0.20,  0.12,  10.0,    3.0,  "Still tired after yawn"),
    (4.0,  0.09,  0.08,  28.0,    4.0,  "FALLING ASLEEP — eyes closed + head nodding"),
    (3.0,  0.28,  0.06,   5.0,    2.0,  "Alert fired — driver woke up"),
    (2.0,  0.29,  0.04,   3.0,    1.0,  "Brief normal period"),
    (3.0,  0.24,  0.08,   5.0,   32.0,  "HEAD TILT — dozing sideways"),
    (2.0,  0.27,  0.06,   4.0,    2.0,  "Recovery from tilt"),
    (2.5,  0.20,  0.70,  -4.0,    3.0,  "YAWNING AGAIN — second yawn"),
    (3.0,  0.14,  0.15,  22.0,    5.0,  "Yawn + drowsy EAR combined"),
    (2.0,  0.28,  0.05,   3.0,    1.0,  "Recovery"),
    (3.0,  0.30,  0.04,   2.0,    1.0,  "Normal — fully awake"),
]


def _interpolate(scenario, elapsed):
    """
    Return smoothly interpolated (ear, mar, pitch, roll, label)
    for the given elapsed time within the scenario.
    Uses cosine easing between phases for natural animation.
    """
    total = sum(d for d, *_ in scenario)
    t = elapsed % total

    acc = 0.0
    for i, (dur, ear, mar, pitch, roll, label) in enumerate(scenario):
        if t < acc + dur:
            # How far through this phase (0.0 → 1.0)
            phase_t = (t - acc) / dur

            # Get next phase values for smooth blending
            next_i = (i + 1) % len(scenario)
            n_ear   = scenario[next_i][1]
            n_mar   = scenario[next_i][2]
            n_pitch = scenario[next_i][3]
            n_roll  = scenario[next_i][4]

            # Only blend in the last 20% of each phase
            blend_start = 0.80
            if phase_t >= blend_start:
                alpha = (phase_t - blend_start) / (1.0 - blend_start)
                # Cosine ease
                alpha = (1 - math.cos(alpha * math.pi)) / 2
                ear   = ear   + alpha * (n_ear   - ear)
                mar   = mar   + alpha * (n_mar   - mar)
                pitch = pitch + alpha * (n_pitch - pitch)
                roll  = roll  + alpha * (n_roll  - roll)

            # Add small noise
            ear   = max(0.0, ear   + np.random.normal(0, 0.003))
            mar   = max(0.0, mar   + np.random.normal(0, 0.005))
            pitch = pitch + np.random.normal(0, 0.3)
            roll  = roll  + np.random.normal(0, 0.3)

            return ear, mar, pitch, roll, label
        acc += dur

    return 0.29, 0.05, 2.0, 1.0, "Normal"


# ══════════════════════════════════════════════════════════════════
#  CARTOON FACE RENDERER
#  Draws a 2D cartoon driver face that animates based on:
#    ear   → eye openness
#    mar   → mouth openness
#    pitch → head nod (vertical shift + perspective squeeze)
#    roll  → head tilt (rotation)
# ══════════════════════════════════════════════════════════════════

def generate_demo_frame(
    ear: float,
    mar: float,
    pitch: float,
    roll: float,
    frame_idx: int,
    fw: int = 640,
    fh: int = 480
) -> np.ndarray:
    """
    Generate an animated cartoon driver face.

    The face reacts to ALL four signals:
      - ear   → eyes open/close
      - mar   → mouth open/close (yawn)
      - pitch → head nods forward (face moves down + compresses)
      - roll  → head tilts sideways (whole face rotates)
    """
    # Dark car interior background
    frame = np.zeros((fh, fw, 3), dtype=np.uint8)
    # Background gradient (darker at top = headliner, lighter at bottom = dashboard glow)
    for y in range(fh):
        v = int(15 + (y / fh) * 25)
        frame[y, :] = (v, v - 5, v + 5)

    # ── Head position from pitch / roll ──────────────────────────
    # Pitch: head nods down → face center moves down + face squishes slightly
    pitch_offset_y = int(np.clip(pitch * 2.5, 0, 80))
    # Roll: head tilts → rotate whole face drawing
    roll_rad = math.radians(np.clip(roll, -40, 40))

    # Base face center
    cx = fw // 2
    cy = fh // 2 - 20 + pitch_offset_y

    # Vertical compression from pitch (perspective effect)
    v_scale = max(0.75, 1.0 - pitch * 0.005)

    # ── Create face canvas, then rotate for roll ─────────────────
    # Draw onto a temporary canvas, then warp-rotate for roll
    canvas = frame.copy()
    _draw_face(canvas, cx, cy, v_scale, ear, mar, frame_idx, fw, fh)

    # Apply roll rotation around face center
    if abs(roll) > 1.0:
        M = cv2.getRotationMatrix2D((cx, cy), -roll, 1.0)
        canvas = cv2.warpAffine(canvas, M, (fw, fh),
                                borderMode=cv2.BORDER_REFLECT_101)

    # Draw nodding indicator when pitch is high
    if pitch > 15:
        nod_alpha = min(1.0, (pitch - 15) / 25)
        _draw_nod_indicator(canvas, fw, fh, nod_alpha)

    # Draw tilt indicator when roll is high
    if abs(roll) > 15:
        _draw_tilt_indicator(canvas, fw, fh, roll)

    # Dashboard glow at bottom
    glow = canvas.copy()
    cv2.rectangle(glow, (0, fh - 70), (fw, fh), (0, 25, 50), -1)
    cv2.addWeighted(glow, 0.65, canvas, 0.35, 0, canvas)

    # Info bar
    cv2.putText(canvas,
                f"DEMO  |  EAR:{ear:.3f}  MAR:{mar:.3f}  Pitch:{pitch:+.1f}  Roll:{roll:+.1f}",
                (8, fh - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                (60, 200, 60), 1, cv2.LINE_AA)

    return canvas


def _draw_face(img, cx, cy, v_scale, ear, mar, frame_idx, fw, fh):
    """Draw the cartoon face components."""
    skin   = (185, 145, 110)
    dark   = (40,  30,  20)
    white  = (240, 235, 228)
    iris_c = (70,  90, 130)
    pupil  = (20,  25,  40)
    lip_c  = (140, 80,  80)
    lip_lo = (110, 55,  55)
    teeth  = (230, 225, 215)

    head_rx = 115
    head_ry = int(135 * v_scale)

    # ── Head / neck ───────────────────────────────────────────────
    # Neck
    cv2.rectangle(img, (cx - 32, cy + head_ry - 10),
                  (cx + 32, cy + head_ry + 55), skin, -1)
    # Head ellipse
    cv2.ellipse(img, (cx, cy), (head_rx, head_ry), 0, 0, 360, skin, -1)
    # Hair
    cv2.ellipse(img, (cx, cy - int(head_ry * 0.3)),
                (head_rx, int(head_ry * 0.75)), 0, 180, 360, dark, -1)
    # Ears
    cv2.ellipse(img, (cx - head_rx + 5, cy), (14, 20), 0, 90, 270, skin, -1)
    cv2.ellipse(img, (cx + head_rx - 5, cy), (14, 20), 0, 270, 90, skin, -1)

    # ── Eyes ─────────────────────────────────────────────────────
    eye_open_h = int(np.interp(ear, [0.0, 0.35], [1, 24]))
    eye_open_h = max(1, eye_open_h)
    eye_w      = 34
    eye_y      = cy - int(25 * v_scale)

    for ex in [cx - 48, cx + 48]:
        # Eye white
        cv2.ellipse(img, (ex, eye_y), (eye_w // 2, eye_open_h),
                    0, 0, 360, white, -1)
        # Iris + pupil (only visible when open enough)
        if eye_open_h > 5:
            iris_r = min(12, eye_open_h - 2)
            cv2.circle(img, (ex, eye_y), iris_r, iris_c, -1)
            cv2.circle(img, (ex, eye_y), max(1, iris_r - 5), pupil, -1)
            # Highlight
            cv2.circle(img, (ex - 3, eye_y - 3), 3, (220, 220, 220), -1)
        # Eyelid border
        cv2.ellipse(img, (ex, eye_y), (eye_w // 2, eye_open_h),
                    0, 0, 360, (80, 50, 30), 2)
        # Upper eyelid (always drawn, creates "drooping" look)
        lid_pts = np.array([
            [ex - eye_w // 2, eye_y],
            [ex, eye_y - eye_open_h - 2],
            [ex + eye_w // 2, eye_y]
        ], np.int32)
        cv2.polylines(img, [lid_pts], False, (60, 35, 20), 2, cv2.LINE_AA)

    # Eyebrows
    brow_y = eye_y - eye_open_h - 12
    for bx_off in [-48, 48]:
        pts = np.array([[cx + bx_off - 20, brow_y + 3],
                        [cx + bx_off,      brow_y],
                        [cx + bx_off + 20, brow_y + 3]], np.int32)
        cv2.polylines(img, [pts], False, dark, 3, cv2.LINE_AA)

    # ── Nose ─────────────────────────────────────────────────────
    nose_y = cy + int(10 * v_scale)
    cv2.line(img, (cx, nose_y - 10), (cx - 13, nose_y + 25),
             (150, 110, 80), 2)
    cv2.line(img, (cx, nose_y - 10), (cx + 13, nose_y + 25),
             (150, 110, 80), 2)
    cv2.ellipse(img, (cx - 10, nose_y + 25), (8, 5),
                0, 0, 360, (160, 120, 90), 1)
    cv2.ellipse(img, (cx + 10, nose_y + 25), (8, 5),
                0, 0, 360, (160, 120, 90), 1)

    # ── Mouth / Yawn ─────────────────────────────────────────────
    mouth_y = cy + int(58 * v_scale)

    # Mouth open height from MAR
    # MAR 0.0 = lips just touching, MAR 0.7+ = wide yawn
    mouth_open_h = int(np.interp(mar, [0.0, 0.15, 0.60, 0.80],
                                      [0,   3,    35,   52]))
    mouth_w = int(np.interp(mar, [0.0, 0.15, 0.60, 0.80],
                                  [22,  24,   36,   42]))

    if mouth_open_h <= 3:
        # Closed / slight smile
        cv2.ellipse(img, (cx, mouth_y), (22, 8),
                    0, 0, 180, lip_c, 2)
        cv2.ellipse(img, (cx, mouth_y - 2), (22, 5),
                    0, 0, 180, lip_lo, 3)
    else:
        # Open mouth — yawning
        # Dark interior
        cv2.ellipse(img, (cx, mouth_y), (mouth_w, mouth_open_h),
                    0, 0, 360, (30, 20, 20), -1)
        # Teeth (upper)
        if mouth_open_h > 8:
            teeth_h = min(14, mouth_open_h // 3)
            cv2.ellipse(img, (cx, mouth_y - mouth_open_h // 3),
                        (mouth_w - 6, teeth_h), 0, 0, 180, teeth, -1)
        # Tongue (when really wide open)
        if mouth_open_h > 22:
            tongue_y = mouth_y + mouth_open_h // 3
            cv2.ellipse(img, (cx, tongue_y), (14, 10),
                        0, 0, 360, (180, 80, 90), -1)
        # Upper lip
        cv2.ellipse(img, (cx, mouth_y), (mouth_w, mouth_open_h),
                    0, 180, 360, lip_c, 3)
        # Lower lip
        cv2.ellipse(img, (cx, mouth_y), (mouth_w, mouth_open_h),
                    0, 0, 180, lip_lo, 3)
        # Yawn label when wide open
        if mar > 0.45:
            cv2.putText(img, "YAWNING", (cx - 45, mouth_y + mouth_open_h + 20),
                        cv2.FONT_HERSHEY_DUPLEX, 0.55, (0, 120, 255), 2, cv2.LINE_AA)

    # ── Stress lines when very drowsy ────────────────────────────
    if ear < 0.15:
        for i, (lx, ly) in enumerate([(-70, -40), (70, -40), (-75, -10), (75, -10)]):
            cv2.line(img, (cx + lx, cy + ly),
                     (cx + lx + (8 if lx > 0 else -8), cy + ly + 4),
                     (100, 70, 50), 1, cv2.LINE_AA)


def _draw_nod_indicator(img, fw, fh, alpha):
    """Draw downward arrows when head is nodding forward."""
    color = tuple(int(c * alpha) for c in (0, 80, 255))
    for x in [fw // 2 - 40, fw // 2, fw // 2 + 40]:
        y_start = 30
        y_end   = 70
        cv2.arrowedLine(img, (x, y_start), (x, y_end),
                        color, 2, cv2.LINE_AA, tipLength=0.4)
    cv2.putText(img, "HEAD NODDING", (fw // 2 - 65, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)


def _draw_tilt_indicator(img, fw, fh, roll):
    """Draw a tilting arrow when head rolls sideways."""
    direction = "LEFT" if roll > 0 else "RIGHT"
    color = (0, 60, 255)
    cx_i, cy_i = fw // 2, 50
    cv2.ellipse(img, (cx_i, cy_i), (35, 35), 0,
                -90, -90 + roll * 2, color, 2)
    cv2.putText(img, f"HEAD TILT {direction}",
                (cx_i - 65, cy_i + 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)


# ══════════════════════════════════════════════════════════════════
#  MAIN DEMO LOOP
# ══════════════════════════════════════════════════════════════════

def run_demo():
    print("  [Demo] Starting full demo — EAR + Yawn + Head Pose\n")
    print("  [Demo] Scenario phases:")
    for i, (d, ear, mar, pitch, roll, label) in enumerate(SCENARIO):
        print(f"    {i+1:2d}. {label:<45}  ({d:.1f}s)")
    print()

    fw, fh = Settings.FRAME_WIDTH, Settings.FRAME_HEIGHT
    renderer     = DisplayRenderer(fw, fh)
    alert_manager = AlertManager()
    logger       = SessionLogger()
    ear_smoother = EARSmoother(window_size=5)
    yawn_det     = YawnDetector()

    # State
    consec_frames = 0
    pose_consec   = 0
    total_alerts  = 0
    alert_active  = False
    pose_alert    = False
    frame_counter = 0
    start_time    = time.time()

    os.makedirs(Settings.SCREENSHOT_DIR, exist_ok=True)

    # Print phase changes to console
    _last_label = ""

    # TURN OFF button on the dashboard (mouse click)
    win_title = Settings.WINDOW_TITLE + "  [DEMO MODE]"
    turn_off = [False]

    def _on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and renderer.is_off_button(x, y):
            turn_off[0] = True

    cv2.namedWindow(win_title)
    cv2.setMouseCallback(win_title, _on_mouse)

    try:
        while not turn_off[0]:
            elapsed = time.time() - start_time
            ear_raw, mar, pitch, roll, label = _interpolate(SCENARIO, elapsed)

            # Console phase announcements
            if label != _last_label:
                print(f"  [Demo Phase] {label}")
                _last_label = label

            frame_counter += 1

            # ── Smooth EAR ──────────────────────────────────────
            avg_ear   = ear_smoother.smooth(ear_raw)
            left_ear  = avg_ear + np.random.normal(0, 0.003)
            right_ear = avg_ear + np.random.normal(0, 0.003)

            # ── Yawn detection ──────────────────────────────────
            yawn_active = yawn_det.update(mar, pitch, avg_ear)

            # ── Head pose drowsy check ──────────────────────────
            pitch_drowsy = pitch > Settings.HEAD_PITCH_THRESHOLD
            roll_drowsy  = abs(roll) > Settings.HEAD_ROLL_THRESHOLD

            if pitch_drowsy or roll_drowsy:
                pose_consec += 1
            else:
                pose_consec = 0

            # ── EAR state machine ───────────────────────────────
            if avg_ear < Settings.EAR_THRESHOLD:
                consec_frames += 1
            else:
                consec_frames = 0

            # ── Alert decision ──────────────────────────────────
            ear_triggered  = consec_frames >= Settings.CONSEC_FRAMES
            pose_triggered = pose_consec   >= Settings.HEAD_POSE_CONSEC_FRAMES
            yawn_triggered = yawn_active

            should_alert = ear_triggered or pose_triggered or yawn_triggered

            if should_alert:
                if not alert_active:
                    alert_active = True
                    total_alerts += 1
                    parts = []
                    if ear_triggered:  parts.append(f"EAR={avg_ear:.3f}")
                    if pose_triggered: parts.append(
                        f"Pitch={pitch:.1f}°" if pitch_drowsy else f"Roll={roll:.1f}°")
                    if yawn_triggered: parts.append("Yawn")
                    print(f"\n  ⚠  [Alert #{total_alerts}] DROWSINESS | {' | '.join(parts)}")
                alert_manager.trigger_alert(frame_counter)
                logger.log_event(frame_counter, left_ear, right_ear, avg_ear,
                                 consec_frames, True, alert_manager.is_vibrating)
            else:
                if alert_active:
                    print(f"  ✔  [Recovery] Driver awake | EAR={avg_ear:.3f}")
                alert_active = False
                alert_manager.clear_alert()

            pose_alert = pose_triggered

            # ── Generate cartoon frame ──────────────────────────
            frame = generate_demo_frame(
                ear=avg_ear,
                mar=mar,
                pitch=pitch,
                roll=roll,
                frame_idx=frame_counter,
                fw=fw, fh=fh
            )

            # ── Render full dashboard ───────────────────────────
            display = renderer.render(
                frame=frame,
                left_ear=max(0, left_ear),
                right_ear=max(0, right_ear),
                avg_ear=max(0, avg_ear),
                frame_counter=frame_counter,
                consec_frames=consec_frames,
                alert_active=alert_active,
                vibration_active=alert_manager.is_vibrating,
                vibration_intensity=alert_manager.vibration_intensity,
                vibration_phase=alert_manager.vibration_phase,
                fps=30.0,
                face_detected=True,
                total_alerts=total_alerts,
                mar=mar,
                yawn_active=yawn_active,
                total_yawns=yawn_det.total_yawns,
                pitch=pitch,
                yaw=0.0,
                roll=roll,
                pose_alert=pose_alert,
                glasses_detected=False,
            )

            cv2.imshow(win_title, display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                print("\n  [Demo] Quit by user")
                break
            elif key == ord('r'):
                consec_frames = pose_consec = 0
                alert_active  = False
                alert_manager.clear_alert()
                ear_smoother.reset()
                yawn_det.reset()
                print("  [Demo] Reset")
            elif key == ord('s'):
                ts   = int(time.time())
                path = os.path.join(Settings.SCREENSHOT_DIR, f"demo_{ts}.png")
                cv2.imwrite(path, display)
                print(f"  [Demo] Screenshot: {path}")

            # Window closed with the X button
            if cv2.getWindowProperty(win_title, cv2.WND_PROP_VISIBLE) < 1:
                break

            time.sleep(1.0 / 30)   # ~30 FPS

        if turn_off[0]:
            print("
  [Demo] TURN OFF button pressed")

    except KeyboardInterrupt:
        print("\n  [Demo] Interrupted")
    finally:
        cv2.destroyAllWindows()
        alert_manager.cleanup()
        logger.close()
        print("  [Demo] Demo session ended.")