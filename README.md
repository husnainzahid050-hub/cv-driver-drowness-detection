# 🚗 Driver Drowsiness Detection System

**Final Year Project (FYP)**  
University of Azad Jammu and Kashmir  
By: **Subhan**

---

## Overview

A real-time driver drowsiness detection system using **MediaPipe Face Mesh** and the **Eye Aspect Ratio (EAR)** algorithm. The system continuously monitors the driver's eye state via a webcam and triggers multi-modal alerts — **audio beeping**, **visual red flash**, and a **simulated seat vibration** — when prolonged eye closure is detected.

```
╔══════════════════════════════════════════════════════════╗
║  Webcam  →  MediaPipe FaceMesh  →  EAR Calculation      ║
║     ↓              (468 landmarks)        ↓              ║
║  Display ←  Alert System  ←  Frame Counter               ║
║           (Audio + Visual + Seat Vibration Sim)          ║
╚══════════════════════════════════════════════════════════╝
```

---

## Project Structure

```
driver_drowsiness_detection/
│
├── main.py                     ← Entry point
│
├── config/
│   └── settings.py             ← All thresholds & parameters
│
├── src/
│   ├── detector.py             ← Main detection loop (core logic)
│   ├── ear_calculator.py       ← EAR formula implementation
│   ├── yawn_detector.py        ← Mouth Aspect Ratio (MAR) yawn detection
│   ├── head_pose.py            ← Head pose (pitch/yaw/roll) via solvePnP
│   ├── glasses_detector.py     ← Glasses detection + EAR adjustment
│   ├── alert_manager.py        ← Audio + visual + vibration alerts
│   ├── display.py              ← OpenCV UI renderer
│   └── logger.py               ← CSV session logging
│
├── demo/
│   └── demo_runner.py          ← Demo without webcam
│
├── tests/
│   ├── test_ear_calculator.py  ← EAR unit tests
│   └── test_alert_manager.py   ← Alert system tests
│
├── docs/
│   └── architecture.md         ← System architecture notes
│
├── assets/                     ← Alert sound files
├── logs/                       ← Session CSV logs
├── screenshots/                ← Saved screenshots
│
├── requirements.txt
└── README.md
```

---

## Installation

### Prerequisites
- Python 3.8 or higher
- Webcam (or use `--demo` mode)

### Steps

```bash
# 1. Clone / download the project
cd driver_drowsiness_detection

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate       # Linux/Mac
venv\Scripts\activate          # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Usage

```bash
# Normal mode — uses your webcam
python main.py

# Demo mode — NO webcam needed, simulates drowsiness automatically
python main.py --demo

# Use external camera (e.g., USB webcam)
python main.py --camera 1

# Custom EAR threshold
python main.py --threshold 0.22

# Custom consecutive frame limit (30 frames ≈ 1 second at 30fps)
python main.py --frames 30

# Disable audio
python main.py --no-sound

# Disable seat vibration simulation
python main.py --no-vibration
```

### Keyboard Shortcuts (while running)

| Key | Action |
|-----|--------|
| `Q` or `ESC` | Quit the system |
| `R` | Reset drowsy frame counter |
| `S` | Save screenshot |

---

## How It Works

### Eye Aspect Ratio (EAR)

The EAR formula was introduced by Soukupová & Čech (2016):

```
         ||p2 − p6|| + ||p3 − p5||
  EAR = ─────────────────────────────
               2 · ||p1 − p4||
```

Where **p1–p6** are the 6 MediaPipe Face Mesh landmark points around each eye:

```
       p2  p3
   p1           p4      ← horizontal span (stays constant)
       p6  p5
```

| State | EAR Range |
|-------|-----------|
| Open eye | 0.25 – 0.35 |
| Normal blink | 0.0 (lasts 2–4 frames) |
| **Drowsiness** | **< 0.21 for ≥ 20 frames** |

### Detection Algorithm

```
For each video frame:
  1. Detect 468 face landmarks via MediaPipe
  2. Extract 6 points from each eye
  3. Compute EAR (left), EAR (right), EAR (average)
  4. Apply 5-frame moving average to reduce noise
  5. If avg_EAR < 0.21:
       increment counter
       if counter ≥ 20:
         ALERT (audio + visual + seat vibration sim)
  6. Else:
       reset counter + clear alert
```

---

## Alert System

When drowsiness is detected, **three simultaneous alerts** fire:

| Alert | Description |
|-------|-------------|
| 🔴 **Visual** | Red border flashes around the frame + red banner overlay |
| 🔊 **Audio** | 880 Hz beep plays and repeats every ~2 seconds |
| 💺 **Seat Vibration** | Simulated seat shake (see below) |

---

## Seat Vibration Simulation (Demo Feature)

> **"Judka" (Jerk/Shake) Feature**

Some drivers fall into deep micro-sleep and **cannot hear the audio alert** due to loud music, hearing impairment, or deep fatigue. This feature simulates a vibrating driver's seat that would physically wake the driver.

### What the Demo Shows

Since this is a software demo with **no hardware**, the vibration is simulated entirely on-screen:

- 🎨 **Dashboard panel** shows an animated car-seat icon that "shakes"
- 📊 **Intensity bars** pulse in the vibration pattern
- 🔊 **Low-frequency buzzing sound** (80 Hz) plays — mimics motor noise
- 📋 **Status label** shows `VIBRATING [ACTIVE]`

### Vibration Pattern

```
Default pattern: [300ms ON] → [100ms OFF] → [300ms ON] → [100ms OFF] → [800ms ON]
                   buzz         pause          buzz         pause        long buzz
```

This "judka" rhythm is designed to be physically disruptive and hard to sleep through.

### Real Hardware (For Reference / Thesis)

In a real vehicle integration, this module would:

```
Software (Python)
    └─ Serial/USB → Arduino Uno
                        └─ PWM Pin → MOSFET (TIP120)
                                        └─ Vibration Motor (ERM 5V)
                                                └─ Embedded in driver seat cushion
```

**Components needed:**
- Arduino Uno (₹500–800)
- ERM Vibration Motor, 5V DC (₹150–300)
- TIP120 NPN Darlington transistor (₹20)
- 1N4001 flyback protection diode (₹5)
- 9V power supply

**Arduino sketch (for reference):**
```cpp
int motorPin = 9;  // PWM pin
void setup() { pinMode(motorPin, OUTPUT); }
void loop() {
  // Pattern: 300ms ON, 100ms OFF
  analogWrite(motorPin, 200);  delay(300);
  analogWrite(motorPin, 0);    delay(100);
  analogWrite(motorPin, 200);  delay(300);
  analogWrite(motorPin, 0);    delay(100);
  analogWrite(motorPin, 200);  delay(800);
  analogWrite(motorPin, 0);    delay(500);
}
```

---

## Configuration

All parameters are in `config/settings.py`:

```python
EAR_THRESHOLD = 0.21      # Lower = less sensitive
CONSEC_FRAMES = 20        # Frames before alert (20 ≈ 0.67s at 30fps)
VIBRATION_PATTERN = [300, 100, 300, 100, 800]  # ms ON/OFF pattern
SOUND_ENABLED = True
VIBRATION_ENABLED = True
```

---

## Running Tests

```bash
# All tests
python -m pytest tests/ -v

# With coverage
python -m pytest tests/ --cov=src --cov-report=term-missing
```

---

## Technology Stack

| Technology | Purpose |
|------------|---------|
| **Python 3.8+** | Core programming language |
| **OpenCV** | Video capture, frame processing, UI rendering |
| **MediaPipe Face Mesh** | Real-time 468-point facial landmark detection |
| **NumPy / SciPy** | EAR calculation, spatial distance computation |
| **Pygame** | Cross-platform audio playback |
| **Pytest** | Unit testing |

---

## Session Logging

The system logs all drowsiness events to `logs/drowsiness_log.csv`:

```csv
timestamp,session_time_s,frame,left_ear,right_ear,avg_ear,consec_frames,alert_triggered,vibration_triggered
2024-01-15 14:23:01.234,45.2,1356,0.1023,0.0987,0.1005,21,1,1
```

---

## Acknowledgements

- Soukupová, T. & Čech, J. (2016). *Real-Time Eye Blink Detection using Facial Landmarks* — EAR formula
- Google MediaPipe team — Face Mesh model

---

## License

Academic project — University of Azad Jammu and Kashmir. All rights reserved.
