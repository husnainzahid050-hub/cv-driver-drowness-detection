"""
src/alert_manager.py
────────────────────
Manages ALL alert types for the drowsiness detection system:
  1. Visual alert  — red overlay + flashing warning on screen
  2. Audio alert   — beeping alarm sound
  3. Vibration sim — seat vibration simulation (DEMO — no hardware)

The three alerts work together:
  • Audio fires immediately when drowsiness threshold is crossed
  • Visual alert activates simultaneously (red screen flash)
  • Seat vibration simulation runs as a parallel thread

[SEAT VIBRATION SIMULATION NOTE]
─────────────────────────────────
In a real vehicle, this module would send PWM signals to a vibration
motor embedded in the driver's seat (e.g., via Arduino/Raspberry Pi).
Since this is a demo project with NO hardware, the vibration is
simulated entirely in software:
  • A visual "seat shake" animation is rendered on screen
  • A buzzing audio pattern (distinct from beep alert) plays
  • A dashboard panel shows "SEAT VIBRATING" with intensity bars
  • All of this mimics exactly what the hardware would do
"""

import threading
import time
import os
import sys
import wave
import struct
import math
from typing import Optional
from config.settings import Settings


# ──────────────────────────────────────────────────────────────────
#  Audio backend — try pygame, fall back to winsound / beep
# ──────────────────────────────────────────────────────────────────
_AUDIO_BACKEND = None

try:
    import pygame
    pygame.mixer.init(frequency=44100, size=-16, channels=1, buffer=512)
    _AUDIO_BACKEND = "pygame"
except ImportError:
    pass

if _AUDIO_BACKEND is None:
    try:
        import winsound
        _AUDIO_BACKEND = "winsound"
    except ImportError:
        pass

if _AUDIO_BACKEND is None:
    _AUDIO_BACKEND = "none"


def _generate_alert_wav(filepath: str, freq: int = 880,
                         duration: float = 0.5, volume: float = 0.8):
    """
    Generate a WAV alert beep programmatically.
    This runs once at startup so no audio file is needed.
    """
    sample_rate = 44100
    num_samples = int(sample_rate * duration)
    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else ".", exist_ok=True)
    with wave.open(filepath, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        for i in range(num_samples):
            # Apply simple envelope (fade in/out to avoid clipping)
            t = i / sample_rate
            envelope = min(1.0, min(t / 0.01, (duration - t) / 0.01))
            sample = int(32767 * volume * envelope * math.sin(2 * math.pi * freq * t))
            wf.writeframes(struct.pack("<h", sample))


def _generate_vibration_wav(filepath: str):
    """
    Generate a distinct buzzing pattern for the seat vibration simulation.
    This is a low-frequency rumble sound (80 Hz) with modulation —
    mimicking what a vibrating seat motor would sound like.
    """
    sample_rate = 44100
    duration = 1.0
    num_samples = int(sample_rate * duration)
    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else ".", exist_ok=True)
    with wave.open(filepath, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        for i in range(num_samples):
            t = i / sample_rate
            # Low rumble at 80 Hz with 5 Hz AM modulation = buzzing effect
            sample = int(32767 * 0.7 *
                         math.sin(2 * math.pi * 80 * t) *
                         (0.5 + 0.5 * math.sin(2 * math.pi * 5 * t)))
            wf.writeframes(struct.pack("<h", sample))


class AlertManager:
    """
    Central alert controller for the drowsiness detection system.
    Thread-safe — alert methods can be called from any thread.
    """

    def __init__(self):
        self._alert_active = False
        self._vibration_active = False
        self._vibration_thread: Optional[threading.Thread] = None
        self._audio_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        # Frame counter for alert repeat interval
        self._last_audio_frame = -999

        # Vibration state (for visual dashboard rendering)
        self.vibration_phase = 0       # 0–7, cycles for animation
        self.vibration_intensity = 0   # 0–10, current intensity

        # Generate audio files if they don't exist
        self._alert_wav = Settings.ALERT_SOUND_FILE
        self._vibration_wav = "assets/vibration_buzz.wav"
        self._ensure_audio_files()

        # Load sounds into pygame if available
        self._alert_sound = None
        self._vibration_sound = None
        if _AUDIO_BACKEND == "pygame":
            self._load_pygame_sounds()

        print(f"  [AlertManager] Audio backend: {_AUDIO_BACKEND}")

    def _ensure_audio_files(self):
        """Generate WAV files if they don't exist yet."""
        os.makedirs("assets", exist_ok=True)
        if not os.path.exists(self._alert_wav):
            _generate_alert_wav(self._alert_wav, freq=880, duration=0.6)
            print(f"  [AlertManager] Generated {self._alert_wav}")
        if not os.path.exists(self._vibration_wav):
            _generate_vibration_wav(self._vibration_wav)
            print(f"  [AlertManager] Generated {self._vibration_wav}")

    def _load_pygame_sounds(self):
        try:
            if os.path.exists(self._alert_wav):
                self._alert_sound = pygame.mixer.Sound(self._alert_wav)
                self._alert_sound.set_volume(0.9)
            if os.path.exists(self._vibration_wav):
                self._vibration_sound = pygame.mixer.Sound(self._vibration_wav)
                self._vibration_sound.set_volume(0.8)
        except Exception as e:
            print(f"  [AlertManager] Sound load error: {e}")

    # ──────────────────────────────────────────────────────────────
    #  Public API
    # ──────────────────────────────────────────────────────────────

    def trigger_alert(self, frame_count: int):
        """
        Call every frame when driver is detected as drowsy.
        Fires audio + activates vibration simulation.
        """
        self._alert_active = True

        # Play audio every N frames so it repeats while drowsy
        if frame_count - self._last_audio_frame >= Settings.ALERT_REPEAT_INTERVAL:
            self._last_audio_frame = frame_count
            if Settings.SOUND_ENABLED:
                self._play_alert_sound()

        # Start vibration simulation if not already running
        if Settings.VIBRATION_ENABLED and not self._vibration_active:
            self._start_vibration_simulation()

    def clear_alert(self):
        """Call when driver wakes up / eyes open."""
        self._alert_active = False
        self._stop_vibration_simulation()

    @property
    def is_alert_active(self) -> bool:
        return self._alert_active

    @property
    def is_vibrating(self) -> bool:
        return self._vibration_active

    def cleanup(self):
        """Release all resources."""
        self._stop_vibration_simulation()
        if _AUDIO_BACKEND == "pygame":
            try:
                pygame.mixer.quit()
            except Exception:
                pass

    # ──────────────────────────────────────────────────────────────
    #  Audio
    # ──────────────────────────────────────────────────────────────

    def _play_alert_sound(self):
        """Play alert beep in a background thread (non-blocking)."""
        def _play():
            try:
                if _AUDIO_BACKEND == "pygame" and self._alert_sound:
                    self._alert_sound.play()
                elif _AUDIO_BACKEND == "winsound":
                    import winsound
                    winsound.Beep(880, 600)
                else:
                    # Last resort: terminal bell
                    print("\a", end="", flush=True)
            except Exception as e:
                pass  # Silent fail — don't crash main loop

        t = threading.Thread(target=_play, daemon=True)
        t.start()

    # ──────────────────────────────────────────────────────────────
    #  Seat Vibration Simulation (DEMO FEATURE)
    # ──────────────────────────────────────────────────────────────

    def _start_vibration_simulation(self):
        """
        Start the seat vibration simulation thread.

        ─────────────────────────────────────────────────────────
        DEMO NOTE — What this simulates:
        ─────────────────────────────────────────────────────────
        In a real vehicle deployment, this function would:
          1. Send a serial/USB command to an Arduino microcontroller
          2. The Arduino would output a PWM signal to a vibration
             motor (ERM or LRA type) embedded in the driver's seat
          3. The pattern (300ms ON / 100ms OFF / 300ms ON / ...) would
             create a rhythmic "judka" (jerk/shake) feel in the seat
          4. This wakes up drivers who may not hear the audio alert
             (e.g., loud music, hearing impairment, deep sleep)

        Hardware that WOULD be used (for reference in thesis):
          - Arduino Uno / Raspberry Pi
          - Vibration Motor Module (ERM, 3V–5V DC)
          - MOSFET transistor (TIP120) for motor control
          - Flyback diode (1N4001) for protection
          - PWM pin output from microcontroller

        For this DEMO, we:
          ✔ Play a buzzing audio pattern (mimics motor sound)
          ✔ Animate the dashboard with a "SEAT VIBRATING" indicator
          ✔ Show intensity bars that pulse in the vibration pattern
          ✔ Log "VIBRATION TRIGGERED" to the session log
        ─────────────────────────────────────────────────────────
        """
        with self._lock:
            if self._vibration_active:
                return
            self._vibration_active = True

        self._vibration_thread = threading.Thread(
            target=self._vibration_loop, daemon=True
        )
        self._vibration_thread.start()

    def _vibration_loop(self):
        """
        Vibration simulation loop.
        Runs the Settings.VIBRATION_PATTERN repeatedly until stopped.
        """
        print("  [VibSim] 🔴 SEAT VIBRATION SIMULATION ACTIVE")
        phase = 0
        while self._vibration_active:
            on_time = Settings.VIBRATION_PATTERN[phase % len(Settings.VIBRATION_PATTERN)]
            is_on_phase = (phase % 2 == 0)  # Even = ON, Odd = OFF

            if is_on_phase:
                self.vibration_intensity = Settings.VIBRATION_INTENSITY
                # Play buzzing sound for audio simulation
                self._play_vibration_buzz()
            else:
                self.vibration_intensity = 0

            # Advance animation phase (for visual rendering)
            self.vibration_phase = (self.vibration_phase + 1) % 8

            time.sleep(on_time / 1000.0)
            phase += 1

        # Cleanup
        self.vibration_intensity = 0
        self.vibration_phase = 0
        print("  [VibSim] ⚫ Seat vibration simulation stopped")

    def _play_vibration_buzz(self):
        """Play the buzzing rumble sound for vibration simulation."""
        def _play():
            try:
                if _AUDIO_BACKEND == "pygame" and self._vibration_sound:
                    self._vibration_sound.play()
                elif _AUDIO_BACKEND == "winsound":
                    import winsound
                    winsound.Beep(80, 200)
            except Exception:
                pass
        t = threading.Thread(target=_play, daemon=True)
        t.start()

    def _stop_vibration_simulation(self):
        with self._lock:
            self._vibration_active = False
        if self._vibration_thread and self._vibration_thread.is_alive():
            self._vibration_thread.join(timeout=1.0)
