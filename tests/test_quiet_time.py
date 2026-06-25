"""Tests for the quiet-time window logic."""

import datetime

from d20app.loop import in_quiet_window


def t(hh, mm=0):
    return datetime.time(hh, mm)


def test_same_day_window():
    assert in_quiet_window(t(12), "09:00", "17:00") is True
    assert in_quiet_window(t(8), "09:00", "17:00") is False
    assert in_quiet_window(t(9), "09:00", "17:00") is True      # inclusive start
    assert in_quiet_window(t(17), "09:00", "17:00") is False    # exclusive end


def test_window_wraps_midnight():
    assert in_quiet_window(t(23), "22:00", "07:00") is True
    assert in_quiet_window(t(3), "22:00", "07:00") is True
    assert in_quiet_window(t(12), "22:00", "07:00") is False
    assert in_quiet_window(t(22), "22:00", "07:00") is True
    assert in_quiet_window(t(7), "22:00", "07:00") is False


def test_disabled_when_blank_or_equal_or_invalid():
    assert in_quiet_window(t(3), "", "") is False
    assert in_quiet_window(t(3), "22:00", "") is False
    assert in_quiet_window(t(3), "08:00", "08:00") is False
    assert in_quiet_window(t(3), "nonsense", "07:00") is False
