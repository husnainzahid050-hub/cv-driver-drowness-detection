"""
src/head_pose.py
────────────────
Head pose estimation using MediaPipe landmarks + OpenCV solvePnP.

Estimates 3 rotation angles:
  Pitch  (X-axis) — nodding forward/backward   ← main drowsiness indicator
  Yaw    (Y-axis) — turning left/right
  Roll   (Z-axis) — tilting sideways

Normal alert driving:  pitch ≈ -10° to +10°
Drowsy nod forward:    pitch > +20° (head dropping toward chest)
Falling asleep:        pitch > +35°

When wearing GLASSES, EAR becomes unreliable because the glasses
frame occludes the eye landmarks. In that case, head pose becomes
the PRIMARY drowsiness detection method:
  - If head is nodding (pitch > threshold) → alert
  - This works regardless of glasses

3D Model Reference Points (standard face geometry):
  These are known real-world 3D coordinates (in mm) for key facial
  landmarks. Combined with their 2D image positions from MediaPipe,
  solvePnP computes camera → face rotation.

Landmark indices used:
  Nose tip     : 4
  Chin         : 152
  Left eye L   : 33
  Right eye R  : 263
  Left mouth   : 61
  Right mouth  : 291
"""

import cv2
import numpy as np
from typing import Optional, Tuple
from config.settings import Settings


# Known 3D coordinates of facial landmarks (in millimetres)
# These approximate the average human face geometry.
# Axes follow the OpenCV camera convention (X right, Y DOWN, Z away from
# the camera) so a driver looking straight at the camera gives pitch ≈ 0°.
# A Y-up model would make every pitch reading come out ~180° off.
FACE_3D_MODEL = np.array([
    [0.0,    0.0,    0.0],    # Nose tip (4)
    [0.0,    63.6,   12.5],  # Chin (152)
    [-43.3, -32.7,   26.0],  # Left eye left corner (33)
    [43.3,  -32.7,   26.0],  # Right eye right corner (263)
    [-28.9,  28.9,   24.1],  # Left mouth corner (61)
    [28.9,   28.9,   24.1],  # Right mouth corner (291)
], dtype=np.float64)

# Corresponding MediaPipe landmark indices
POSE_LANDMARK_INDICES = [4, 152, 33, 263, 61, 291]

# Axis colours for drawing (BGR)
AXIS_COLORS = {
    "x": (0, 0, 255),   # Red   — pitch axis
    "y": (0, 255, 0),   # Green — yaw axis
    "z": (255, 0, 0),   # Blue  — roll axis
}


class HeadPoseEstimator:
    """
    Estimates head pose angles from MediaPipe face landmarks.
    Falls back gracefully when estimation fails.
    """

    def __init__(self):
        self._camera_matrix: Optional[np.ndarray] = None
        self._dist_coeffs = np.zeros((4, 1), dtype=np.float64)
        self._last_angles: Tuple[float, float, float] = (0.0, 0.0, 0.0)

        # Smoothing history
        self._pitch_history = []
        self._smooth_window = 8

        # State
        self.pitch = 0.0    # Forward nod (most important for drowsiness)
        self.yaw = 0.0      # Left/right turn
        self.roll = 0.0     # Side tilt
        self.pose_valid = False

    def _init_camera_matrix(self, frame_w: int, frame_h: int):
        """Estimate camera intrinsic matrix from frame size."""
        focal_length = frame_w
        cx, cy = frame_w / 2.0, frame_h / 2.0
        self._camera_matrix = np.array([
            [focal_length, 0,            cx],
            [0,            focal_length, cy],
            [0,            0,            1 ],
        ], dtype=np.float64)

    def estimate(
        self,
        landmarks,
        frame_w: int,
        frame_h: int
    ) -> Tuple[float, float, float]:
        """
        Estimate head pose from face landmarks.

        Returns:
            (pitch, yaw, roll) in degrees.
            Positive pitch = head nodding FORWARD (drowsy).
        """
        if self._camera_matrix is None:
            self._init_camera_matrix(frame_w, frame_h)

        # Extract 2D image points
        image_points = []
        for idx in POSE_LANDMARK_INDICES:
            lm = landmarks.landmark[idx]
            image_points.append([lm.x * frame_w, lm.y * frame_h])
        image_points = np.array(image_points, dtype=np.float64)

        try:
            success, rot_vec, trans_vec = cv2.solvePnP(
                FACE_3D_MODEL,
                image_points,
                self._camera_matrix,
                self._dist_coeffs,
                flags=cv2.SOLVEPNP_SQPNP
            )
        except cv2.error:
            self.pose_valid = False
            return self._last_angles

        if not success:
            self.pose_valid = False
            return self._last_angles

        # Convert rotation vector to rotation matrix
        rot_mat, _ = cv2.Rodrigues(rot_vec)

        # Decompose to Euler angles
        # Using Tait-Bryan angles (ZYX convention)
        sy = np.sqrt(rot_mat[0, 0] ** 2 + rot_mat[1, 0] ** 2)
        singular = sy < 1e-6

        if not singular:
            x = np.arctan2(rot_mat[2, 1], rot_mat[2, 2])
            y = np.arctan2(-rot_mat[2, 0], sy)
            z = np.arctan2(rot_mat[1, 0], rot_mat[0, 0])
        else:
            x = np.arctan2(-rot_mat[1, 2], rot_mat[1, 1])
            y = np.arctan2(-rot_mat[2, 0], sy)
            z = 0.0

        pitch_deg = float(np.degrees(x))
        yaw_deg   = float(np.degrees(y))
        roll_deg  = float(np.degrees(z))

        # Smooth pitch to reduce jitter
        self._pitch_history.append(pitch_deg)
        if len(self._pitch_history) > self._smooth_window:
            self._pitch_history.pop(0)
        smooth_pitch = float(np.mean(self._pitch_history))

        self.pitch = round(smooth_pitch, 1)
        self.yaw   = round(yaw_deg, 1)
        self.roll  = round(roll_deg, 1)
        self.pose_valid = True
        self._last_angles = (self.pitch, self.yaw, self.roll)

        return self.pitch, self.yaw, self.roll

    def is_drowsy_pose(self) -> bool:
        """
        Returns True if head pose indicates drowsiness.
        Triggered when:
          - Pitch > PITCH_THRESHOLD  (head nodding forward — falling asleep)
          - OR |Roll| > ROLL_THRESHOLD  (head tilting sideways — losing control)
        """
        if not self.pose_valid:
            return False
        pitch_drowsy = self.pitch > Settings.HEAD_PITCH_THRESHOLD
        roll_drowsy = abs(self.roll) > Settings.HEAD_ROLL_THRESHOLD
        return pitch_drowsy or roll_drowsy

    def get_status_text(self) -> str:
        if not self.pose_valid:
            return "No Pose"
        if self.pitch > Settings.HEAD_PITCH_THRESHOLD:
            return "NODDING !"
        if abs(self.roll) > Settings.HEAD_ROLL_THRESHOLD:
            return "TILTING !"
        return "Normal"

    def draw_axes(self, frame: np.ndarray, landmarks, frame_w: int, frame_h: int):
        """Draw 3D orientation axes on the face for visualization."""
        if not self.pose_valid or self._camera_matrix is None:
            return

        try:
            # Get nose tip as origin
            nose = landmarks.landmark[4]
            nose_2d = (int(nose.x * frame_w), int(nose.y * frame_h))

            # Build rotation vector again for drawing
            image_points = []
            for idx in POSE_LANDMARK_INDICES:
                lm = landmarks.landmark[idx]
                image_points.append([lm.x * frame_w, lm.y * frame_h])
            image_points = np.array(image_points, dtype=np.float64)

            _, rot_vec, trans_vec = cv2.solvePnP(
                FACE_3D_MODEL, image_points,
                self._camera_matrix, self._dist_coeffs,
                flags=cv2.SOLVEPNP_SQPNP
            )

            # Project axis endpoints onto image
            axis_length = 60.0
            axis_points_3d = np.float32([
                [axis_length, 0, 0],   # X axis → pitch
                [0, axis_length, 0],   # Y axis → yaw
                [0, 0, axis_length],   # Z axis → roll
            ])

            projected, _ = cv2.projectPoints(
                axis_points_3d, rot_vec, trans_vec,
                self._camera_matrix, self._dist_coeffs
            )

            p1 = (int(projected[0][0][0]), int(projected[0][0][1]))
            p2 = (int(projected[1][0][0]), int(projected[1][0][1]))
            p3 = (int(projected[2][0][0]), int(projected[2][0][1]))

            cv2.line(frame, nose_2d, p1, AXIS_COLORS["x"], 2, cv2.LINE_AA)
            cv2.line(frame, nose_2d, p2, AXIS_COLORS["y"], 2, cv2.LINE_AA)
            cv2.line(frame, nose_2d, p3, AXIS_COLORS["z"], 2, cv2.LINE_AA)

        except Exception:
            pass  # Never crash main loop

    def reset(self):
        self._pitch_history.clear()
        self.pitch = self.yaw = self.roll = 0.0
        self.pose_valid = False
