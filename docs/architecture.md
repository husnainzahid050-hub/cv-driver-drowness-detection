# System Architecture

## Component Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                   DRIVER DROWSINESS DETECTION SYSTEM                │
│                         FYP — UAJK — Subhan                         │
└─────────────────────────────────────────────────────────────────────┘

INPUT                   PROCESSING                      OUTPUT
──────             ──────────────────────           ──────────────────

Webcam             ┌─────────────────────┐          OpenCV Window
(30 FPS)  ────────▶│  DrowsinessDetector │────────▶  (Camera Feed +
                   │  (detector.py)      │            Dashboard Panel)
                   └──────────┬──────────┘
                              │
                              ▼
                   ┌─────────────────────┐
                   │   MediaPipe         │          CSV Log
                   │   Face Mesh         │────────▶  (logs/session.csv)
                   │   468 landmarks     │
                   └──────────┬──────────┘
                              │
                              ▼
                   ┌─────────────────────┐
                   │  EAR Calculator     │
                   │  (ear_calculator.py)│
                   │                     │
                   │  • compute_ear()    │
                   │  • EARSmoother      │
                   └──────────┬──────────┘
                              │
                              ▼
                   ┌─────────────────────┐
                   │  Frame Counter      │
                   │  State Machine      │
                   │                     │
                   │  if EAR < 0.21:     │
                   │    counter++        │
                   │  if counter >= 20:  │
                   │    ALERT            │
                   └──────────┬──────────┘
                              │
                    ┌─────────┴──────────┐
                    ▼                    ▼
         ┌──────────────────┐   ┌────────────────────┐
         │  AlertManager    │   │  DisplayRenderer   │
         │ (alert_manager.py│   │  (display.py)      │
         │                  │   │                    │
         │ ① Audio Alert    │   │ • EAR overlays     │
         │   880Hz beep     │   │ • Status bar       │
         │                  │   │ • EAR graph        │
         │ ② Visual Alert   │   │ • Vibration panel  │
         │   Red flash      │   │ • Session stats    │
         │                  │   └────────────────────┘
         │ ③ Vib. Sim [DEMO]│
         │   Buzzing sound  │
         │   Shake animation│
         └──────────────────┘
```

## Data Flow

```
Frame (BGR) → RGB conversion → MediaPipe → 468 landmarks
                                                │
                              ┌─────────────────┘
                              │
                    Left eye (6 pts) + Right eye (6 pts)
                              │
                         EAR = (A+B) / 2C
                              │
                    5-frame moving average
                              │
                    avg_EAR < 0.21?
                         /       \
                        Yes       No
                        │         │
               consec++         reset counter
                        │
               counter >= 20?
                    /       \
                   Yes       No
                   │          │
              TRIGGER       continue
              ALERTS        monitoring
```

## Module Responsibilities

| Module | Lines | Responsibility |
|--------|-------|---------------|
| `main.py` | ~80 | CLI args, banner, entry point |
| `config/settings.py` | ~90 | All constants in one place |
| `src/detector.py` | ~180 | Main loop, state machine, keyboard |
| `src/ear_calculator.py` | ~100 | EAR formula, coordinate extraction, smoother |
| `src/alert_manager.py` | ~200 | Audio, visual, vibration simulation |
| `src/display.py` | ~280 | All OpenCV rendering |
| `src/logger.py` | ~70 | CSV event logging |
| `demo/demo_runner.py` | ~150 | Webcam-free demo mode |
| `tests/` | ~200 | Unit tests |

## Threading Model

```
Main Thread                     Worker Threads
───────────                     ──────────────
DrowsinessDetector.run()
  ├─ cap.read()                
  ├─ _process_frame()          
  ├─ _update_state()           
  │    └─ alert_manager        ──▶ _audio_thread (daemon)
  │         .trigger_alert()   ──▶ _vibration_thread (daemon)
  ├─ renderer.render()         
  └─ cv2.imshow()              
```

All worker threads are `daemon=True` — they auto-stop when main thread exits.

## EAR Landmark Indices (MediaPipe Face Mesh)

```
Left eye:  [362, 385, 387, 263, 373, 380]
Right eye: [33,  160, 158, 133, 153, 144]

Index meaning:
  [0] = outer corner (p1)
  [1] = upper outer (p2)
  [2] = upper inner (p3)
  [3] = inner corner (p4)
  [4] = lower inner (p5)
  [5] = lower outer (p6)
```
