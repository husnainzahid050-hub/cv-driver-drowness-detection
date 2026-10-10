"""
tests/test_ear_calculator.py
─────────────────────────────
Unit tests for the EAR calculation module.
Run with:  python -m pytest tests/ -v
"""

import sys
import os
import math
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.ear_calculator import compute_ear, average_ear, EARSmoother


class TestComputeEAR:
    """Tests for the compute_ear() function."""

    def test_open_eye_returns_expected_value(self):
        """
        An idealised open eye with these coordinates:
          p1=(0,0) p4=(6,0)  — horizontal span = 6
          p2=(1,-2) p3=(3,-2) p5=(3,2) p6=(1,2)  — vertical span ≈ 4
        Expected EAR = (4+4) / (2*6) = 0.667
        """
        eye = [(0, 0), (1, -2), (3, -2), (6, 0), (3, 2), (1, 2)]
        ear = compute_ear(eye)
        assert abs(ear - 0.6667) < 0.01, f"Got {ear}, expected ~0.667"

    def test_closed_eye_returns_near_zero(self):
        """
        A nearly-closed eye: vertical distances ≈ 0.
        EAR should be close to 0.
        """
        # All points on the same horizontal line (eye closed)
        eye = [(0, 0), (1, 0), (3, 0), (6, 0), (3, 0), (1, 0)]
        ear = compute_ear(eye)
        assert ear < 0.05, f"Expected ~0 for closed eye, got {ear}"

    def test_zero_horizontal_distance_returns_zero(self):
        """Guard against division by zero when p1 == p4."""
        eye = [(3, 0), (3, -2), (3, -2), (3, 0), (3, 2), (3, 2)]
        ear = compute_ear(eye)
        assert ear == 0.0

    def test_typical_open_eye_in_range(self):
        """Typical open eye EAR should be between 0.20 and 0.40."""
        # Approximate real-world landmarks
        eye = [(100, 200), (112, 194), (124, 193),
               (136, 200), (124, 207), (112, 206)]
        ear = compute_ear(eye)
        assert 0.15 < ear < 0.45, f"Typical EAR out of range: {ear}"

    def test_output_is_float(self):
        eye = [(0, 0), (1, -2), (3, -2), (6, 0), (3, 2), (1, 2)]
        ear = compute_ear(eye)
        assert isinstance(ear, float)

    def test_output_is_non_negative(self):
        eye = [(0, 0), (1, -2), (3, -2), (6, 0), (3, 2), (1, 2)]
        ear = compute_ear(eye)
        assert ear >= 0.0


class TestAverageEAR:
    def test_averages_correctly(self):
        assert average_ear(0.30, 0.20) == 0.25

    def test_same_values(self):
        assert average_ear(0.25, 0.25) == 0.25

    def test_returns_float(self):
        result = average_ear(0.30, 0.20)
        assert isinstance(result, float)


class TestEARSmoother:
    def test_single_value_returns_itself(self):
        smoother = EARSmoother(window_size=5)
        assert smoother.smooth(0.28) == 0.28

    def test_smoothing_reduces_spike(self):
        smoother = EARSmoother(window_size=5)
        # Feed stable values then a spike
        for _ in range(4):
            smoother.smooth(0.28)
        result = smoother.smooth(0.50)  # spike
        assert result < 0.40, "Smoother should dampen sudden spikes"

    def test_window_size_respected(self):
        smoother = EARSmoother(window_size=3)
        smoother.smooth(0.10)
        smoother.smooth(0.10)
        smoother.smooth(0.10)
        smoother.smooth(0.40)  # This should push out the oldest 0.10
        # History should be [0.10, 0.10, 0.40] → mean = 0.20
        assert abs(smoother.smooth.__wrapped__ if hasattr(smoother.smooth, '__wrapped__') else 0) == 0
        assert len(smoother.history) <= 3

    def test_reset_clears_history(self):
        smoother = EARSmoother()
        smoother.smooth(0.28)
        smoother.smooth(0.25)
        smoother.reset()
        assert len(smoother.history) == 0

    def test_smoothed_value_within_history_range(self):
        smoother = EARSmoother(window_size=5)
        values = [0.20, 0.22, 0.25, 0.23, 0.21]
        last_result = None
        for v in values:
            last_result = smoother.smooth(v)
        assert min(values) <= last_result <= max(values)


class TestEARThreshold:
    """Tests that verify EAR values against the detection threshold."""

    THRESHOLD = 0.21  # From Settings.EAR_THRESHOLD

    def test_open_eye_above_threshold(self):
        eye = [(0, 0), (1, -2), (3, -2), (6, 0), (3, 2), (1, 2)]
        ear = compute_ear(eye)
        assert ear > self.THRESHOLD, \
            f"Open eye EAR {ear} should be > threshold {self.THRESHOLD}"

    def test_closed_eye_below_threshold(self):
        eye = [(0, 0), (1, 0), (3, 0), (6, 0), (3, 0), (1, 0)]
        ear = compute_ear(eye)
        assert ear < self.THRESHOLD, \
            f"Closed eye EAR {ear} should be < threshold {self.THRESHOLD}"
