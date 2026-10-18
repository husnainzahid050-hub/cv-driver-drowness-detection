"""
config/settings.py
──────────────────
Central configuration for the Driver Drowsiness Detection System.
Adjust these values to tune detection sensitivity.
"""


class Settings:
    # ──────────────────────────────────────────────────────────────
    # EAR (Eye Aspect Ratio) Detection Parameters
    # ──────────────────────────────────────────────────────────────

    # EAR value below which an eye is considered "closed"
    # Typical range: 0.18 (very sensitive) to 0.25 (less sensitive)
    # Default 0.21 works well for most people in normal lighting
    EAR_THRESHOLD: float = 0.21

    # Number of CONSECUTIVE frames where EAR < threshold to trigger alert
    # At 30 FPS: 20 frames ≈ 0.67 seconds  |  30 frames ≈ 1 second
    # Normal blink = 2–4 frames — we ignore those
    CONSEC_FRAMES: int = 20

    # Minimum EAR for eyes to be considered "open" (resets counter)
    EAR_OPEN_THRESHOLD: float = 0.25

    # ──────────────────────────────────────────────────────────────
    # Camera / Video Settings
    # ──────────────────────────────────────────────────────────────
    TARGET_FPS: int = 30
    FRAME_WIDTH: int = 640
    FRAME_HEIGHT: int = 480

    # ──────────────────────────────────────────────────────────────
    # MediaPipe Settings
    # ──────────────────────────────────────────────────────────────

    # Minimum detection confidence for face detection
    MIN_DETECTION_CONFIDENCE: float = 0.5

    # Minimum tracking confidence for face tracking
    MIN_TRACKING_CONFIDENCE: float = 0.5

    # Refine landmark detection (gets more accurate eye/iris points)
    REFINE_LANDMARKS: bool = True

    # ──────────────────────────────────────────────────────────────
    # Eye Landmark Indices (MediaPipe Face Mesh — 468 points)
    # These are the 6 key points around each eye used for EAR
    # ──────────────────────────────────────────────────────────────

    # Left eye landmark indices
    LEFT_EYE_INDICES: list = [362, 385, 387, 263, 373, 380]

    # Right eye landmark indices
    RIGHT_EYE_INDICES: list = [33, 160, 158, 133, 153, 144]

    # Full eye outline for visualization
    LEFT_EYE_OUTLINE: list = [
        362, 382, 381, 380, 374, 373, 390, 249,
        263, 466, 388, 387, 386, 385, 384, 398
    ]
    RIGHT_EYE_OUTLINE: list = [
        33, 7, 163, 144, 145, 153, 154, 155,
        133, 173, 157, 158, 159, 160, 161, 246
    ]

    # ──────────────────────────────────────────────────────────────
    # Alert Settings
    # ──────────────────────────────────────────────────────────────
    SOUND_ENABLED: bool = True

    # Alert sound file (relative to project root)
    ALERT_SOUND_FILE: str = "assets/alert.wav"

    # How many times to repeat alert sound while drowsy
    ALERT_REPEAT_INTERVAL: int = 60  # frames between repeats

    # ──────────────────────────────────────────────────────────────
    # Seat Vibration Simulation Settings (DEMO FEATURE)
    # ──────────────────────────────────────────────────────────────
    VIBRATION_ENABLED: bool = True

    # Vibration pattern (milliseconds on, milliseconds off)
    # Simulates: strong buzz — pause — strong buzz — pause — long buzz
    VIBRATION_PATTERN: list = [300, 100, 300, 100, 800]

    # How long each vibration cycle lasts (seconds)
    VIBRATION_DURATION: float = 3.0

    # Vibration intensity levels: 1-10 (affects visual sim)
    VIBRATION_INTENSITY: int = 8

    # ──────────────────────────────────────────────────────────────
    # Display / UI Settings
    # ──────────────────────────────────────────────────────────────

    # Main window title
    WINDOW_TITLE: str = "Driver Drowsiness Detection System | FYP - Subhan"

    # Dashboard panel width (pixels) added beside video
    DASHBOARD_WIDTH: int = 320

    # Colors (BGR format for OpenCV)
    COLOR_OK: tuple = (0, 220, 0)          # Green — awake
    COLOR_WARNING: tuple = (0, 165, 255)   # Orange — borderline
    COLOR_ALERT: tuple = (0, 0, 255)       # Red — drowsy
    COLOR_ACCENT: tuple = (255, 200, 0)    # Cyan-ish accent
    COLOR_WHITE: tuple = (255, 255, 255)
    COLOR_BLACK: tuple = (0, 0, 0)
    COLOR_DARK_BG: tuple = (20, 20, 35)    # Dark panel background
    COLOR_PANEL: tuple = (35, 35, 55)

    # ──────────────────────────────────────────────────────────────
    # Screenshot Settings
    # ──────────────────────────────────────────────────────────────
    SCREENSHOT_DIR: str = "screenshots"

    # ──────────────────────────────────────────────────────────────
    # Yawn Detection (Mouth Aspect Ratio)
    # ──────────────────────────────────────────────────────────────

    # MAR value above which mouth is considered "wide open" (yawning)
    # Normal speech: 0.3–0.5  |  Yawn: > 0.5
    MAR_THRESHOLD: float = 0.50

    # Hysteresis: once a yawn has started, it only ends when MAR drops
    # below this lower value (prevents flicker / double counting)
    MAR_EXIT_THRESHOLD: float = 0.40

    # Frames of wide-open mouth before yawn is confirmed.
    # Yawns last 2–6 s, speech openings are short → duration filters talking.
    # At 30 FPS: 30 frames ≈ 1 second
    YAWN_CONSEC_FRAMES: int = 30

    # Brief dips below MAR_EXIT_THRESHOLD tolerated without ending the yawn
    # At 30 FPS: 6 frames ≈ 0.2 seconds
    YAWN_GAP_FRAMES: int = 6

    # ── Yawn vs. surprise / shock ("😮") ──
    # Yawn:     mouth opens SLOWLY, head tilts UP/back, eyes lazy / half-closed
    # Surprise: mouth opens SUDDENLY, head slightly FORWARD, eyes wide open

    # MAR at which the mouth is considered "starting to open"
    MAR_OPEN_START: float = 0.30

    # Opening from MAR_OPEN_START to MAR_THRESHOLD within this many
    # frames counts as a sudden (surprise) opening
    # At 30 FPS: 4 frames ≈ 0.13 seconds (scaled to the real camera FPS)
    YAWN_SUDDEN_ONSET_FRAMES: int = 4

    # Head tilt up/back (degrees, relative to the driver's normal pose)
    # that supports a yawn
    YAWN_HEAD_TILT_DEG: float = 5.0

    # Head moving forward (degrees, relative to normal pose) that
    # supports surprise / shock
    SURPRISE_HEAD_FORWARD_DEG: float = 4.0

    # EAR change relative to the driver's normal EAR:
    #   above (1 + ratio) → eyes widened (surprise)
    #   below (1 - ratio) → eyes narrowed (yawn)
    YAWN_EYE_CHANGE_RATIO: float = 0.10

    # ──────────────────────────────────────────────────────────────
    # Head Pose Estimation
    # ──────────────────────────────────────────────────────────────

    # Pitch angle (degrees) at which forward head nod triggers alert
    # Normal: -10 to +10  |  Drowsy nod: > 20  |  Deep nod: > 35
    HEAD_PITCH_THRESHOLD: float = 25.0

    # Roll angle (degrees) at which sideways head tilt triggers alert
    HEAD_ROLL_THRESHOLD: float = 30.0

    # Consecutive frames of bad head pose before alert fires
    HEAD_POSE_CONSEC_FRAMES: int = 25

    # Show 3D axis overlay on face
    SHOW_HEAD_AXES: bool = True

    # ──────────────────────────────────────────────────────────────
    # Glasses Detection
    # ──────────────────────────────────────────────────────────────

    # EAR range that suggests glasses are suppressing eye landmarks
    GLASSES_EAR_LOW: float = 0.13
    GLASSES_EAR_HIGH: float = 0.21

    # How stable (low variance) EAR must be to indicate glasses
    GLASSES_EAR_VARIANCE_THRESHOLD: float = 0.0008

    # Number of frames to observe before confirming glasses presence
    GLASSES_HISTORY_FRAMES: int = 90   # 3 seconds at 30 FPS
    GLASSES_CONFIRM_FRAMES: int = 3    # Evaluation passes needed

    # When glasses detected, relax EAR threshold by this amount
    GLASSES_EAR_THRESHOLD_REDUCTION: float = 0.04

    # Weights for EAR vs Head Pose when glasses detected
    GLASSES_EAR_WEIGHT: float = 0.3    # Reduced (glasses interfere)
    GLASSES_POSE_WEIGHT: float = 0.85  # Increased (reliable without eyes)

    # ──────────────────────────────────────────────────────────────
    # Logging
    # ──────────────────────────────────────────────────────────────
    LOG_FILE: str = "logs/drowsiness_log.csv"
    LOG_ENABLED: bool = True
