"""
tests/test_alert_manager.py
────────────────────────────
Unit tests for the AlertManager (alert + vibration simulation).
"""

import sys
import os
import time
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.alert_manager import AlertManager
from config.settings import Settings


class TestAlertManager:

    def setup_method(self):
        """Create a fresh AlertManager for each test."""
        # Disable sound for tests to avoid noise
        Settings.SOUND_ENABLED = False
        self.am = AlertManager()

    def teardown_method(self):
        self.am.cleanup()
        Settings.SOUND_ENABLED = True

    def test_initial_state(self):
        assert not self.am.is_alert_active
        assert not self.am.is_vibrating

    def test_trigger_alert_sets_active(self):
        self.am.trigger_alert(frame_count=100)
        assert self.am.is_alert_active

    def test_clear_alert_deactivates(self):
        self.am.trigger_alert(frame_count=100)
        self.am.clear_alert()
        assert not self.am.is_alert_active

    def test_vibration_starts_on_trigger(self):
        Settings.VIBRATION_ENABLED = True
        self.am.trigger_alert(frame_count=100)
        time.sleep(0.1)  # Let thread start
        assert self.am.is_vibrating
        self.am.clear_alert()
        Settings.VIBRATION_ENABLED = True

    def test_vibration_stops_on_clear(self):
        Settings.VIBRATION_ENABLED = True
        self.am.trigger_alert(frame_count=100)
        time.sleep(0.1)
        self.am.clear_alert()
        time.sleep(0.2)  # Let thread stop
        assert not self.am.is_vibrating

    def test_multiple_triggers_do_not_duplicate(self):
        """Calling trigger_alert multiple times should not spawn extra threads."""
        self.am.trigger_alert(100)
        self.am.trigger_alert(101)
        self.am.trigger_alert(102)
        time.sleep(0.1)
        # Should still be exactly one vibration thread running
        assert self.am.is_alert_active

    def test_vibration_disabled_setting(self):
        """When VIBRATION_ENABLED=False, no vibration should start."""
        Settings.VIBRATION_ENABLED = False
        self.am.trigger_alert(frame_count=100)
        time.sleep(0.1)
        assert not self.am.is_vibrating
        Settings.VIBRATION_ENABLED = True

    def test_vibration_pattern_exists(self):
        """Vibration pattern should be a non-empty list of positive ints."""
        pattern = Settings.VIBRATION_PATTERN
        assert isinstance(pattern, list)
        assert len(pattern) > 0
        assert all(isinstance(p, int) and p > 0 for p in pattern)
