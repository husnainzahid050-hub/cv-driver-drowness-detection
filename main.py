"""
╔══════════════════════════════════════════════════════════════╗
║        DRIVER DROWSINESS DETECTION SYSTEM                    ║
║        Using MediaPipe Face Mesh + EAR Algorithm             ║
║        Final Year Project — University of AJ&K               ║
║        By: Name : M Subhan Munir Abbasi & Hussnian Abbasi    |                                         ║
╚══════════════════════════════════════════════════════════════╝

Entry point. Run this file to start the system.

Usage:
    python main.py                  # Normal mode (webcam)
    python main.py --demo           # Demo mode (no webcam needed)
    python main.py --camera 1       # Use external camera index 1
    python main.py --threshold 0.22 # Custom EAR threshold
"""

import argparse
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

from src.detector import DrowsinessDetector
from config.settings import Settings


def parse_args():
    parser = argparse.ArgumentParser(
        description="Driver Drowsiness Detection System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "--demo", action="store_true",
        help="Run in demo mode (simulates drowsiness without webcam)"
    )
    parser.add_argument(
        "--camera", type=int, default=0,
        help="Camera index (default: 0 = built-in webcam)"
    )
    parser.add_argument(
        "--threshold", type=float, default=None,
        help=f"EAR threshold for drowsiness (default: {Settings.EAR_THRESHOLD})"
    )
    parser.add_argument(
        "--frames", type=int, default=None,
        help=f"Consecutive frames before alert (default: {Settings.CONSEC_FRAMES})"
    )
    parser.add_argument(
        "--no-sound", action="store_true",
        help="Disable audio alerts"
    )
    parser.add_argument(
        "--no-vibration", action="store_true",
        help="Disable seat vibration simulation"
    )
    return parser.parse_args()


def print_banner():
    print("\n" + "═" * 65)
    print("  🚗  DRIVER DROWSINESS DETECTION SYSTEM")
    print("       University of Azad Jammu & Kashmir | FYP Project")
    print("═" * 65)
    print(f"  EAR Threshold  : {Settings.EAR_THRESHOLD}")
    print(f"  Frame Limit    : {Settings.CONSEC_FRAMES} consecutive closed-eye frames")
    print(f"  Target FPS     : {Settings.TARGET_FPS}")
    print(f"  Alert Sound    : {'Enabled' if Settings.SOUND_ENABLED else 'Disabled'}")
    print(f"  Vibration Sim  : Enabled (Demo — no hardware required)")
    print("═" * 65)
    print("  Press  Q  to quit  |  R  to reset counter  |  S  to screenshot")
    print("═" * 65 + "\n")


def main():
    args = parse_args()

    # Apply CLI overrides to settings
    if args.threshold is not None:
        Settings.EAR_THRESHOLD = args.threshold
    if args.frames is not None:
        Settings.CONSEC_FRAMES = args.frames
    if args.no_sound:
        Settings.SOUND_ENABLED = False
    if args.no_vibration:
        Settings.VIBRATION_ENABLED = False

    print_banner()

    if args.demo:
        print("  [DEMO MODE] Running without webcam — simulating drowsiness events...\n")
        from demo.demo_runner import run_demo
        run_demo()
    else:
        detector = DrowsinessDetector(camera_index=args.camera)
        detector.run()


if __name__ == "__main__":
    main()
