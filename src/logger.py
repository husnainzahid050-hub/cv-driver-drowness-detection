"""
src/logger.py
─────────────
CSV-based session logger.
Records every drowsiness event with timestamp, EAR values, and alert type.
"""

import csv
import os
import time
from datetime import datetime
from config.settings import Settings


class SessionLogger:
    """Logs drowsiness events to a CSV file for post-session analysis."""

    HEADERS = [
        "timestamp", "session_time_s", "frame",
        "left_ear", "right_ear", "avg_ear",
        "consec_frames", "alert_triggered", "vibration_triggered"
    ]

    def __init__(self):
        self._enabled = Settings.LOG_ENABLED
        self._filepath = Settings.LOG_FILE
        self._session_start = time.time()
        self._writer = None
        self._file = None
        self._event_count = 0

        if self._enabled:
            self._open_log()

    def _open_log(self):
        try:
            os.makedirs(os.path.dirname(self._filepath)
                        if os.path.dirname(self._filepath) else ".", exist_ok=True)
            self._file = open(self._filepath, "w", newline="", encoding="utf-8")
            self._writer = csv.DictWriter(self._file, fieldnames=self.HEADERS)
            self._writer.writeheader()
            print(f"  [Logger] Session log: {os.path.abspath(self._filepath)}")
        except Exception as e:
            print(f"  [Logger] Could not open log file: {e}")
            self._enabled = False

    def log_event(
        self,
        frame: int,
        left_ear: float,
        right_ear: float,
        avg_ear: float,
        consec_frames: int,
        alert: bool,
        vibration: bool
    ):
        if not self._enabled or self._writer is None:
            return
        try:
            self._writer.writerow({
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
                "session_time_s": round(time.time() - self._session_start, 2),
                "frame": frame,
                "left_ear": round(left_ear, 4),
                "right_ear": round(right_ear, 4),
                "avg_ear": round(avg_ear, 4),
                "consec_frames": consec_frames,
                "alert_triggered": int(alert),
                "vibration_triggered": int(vibration),
            })
            self._file.flush()
            self._event_count += 1
        except Exception:
            pass

    def close(self):
        if self._file:
            try:
                self._file.close()
                if self._event_count > 0:
                    print(f"  [Logger] Saved {self._event_count} events → {self._filepath}")
            except Exception:
                pass
