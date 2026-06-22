"""Unit tests for the dice + rate-limiting logic (no hardware, no sleeping)."""

import random

import pytest

from d20app import dice


def test_roll_within_range_for_various_dice():
    rng = random.Random(1234)
    for sides in (2, 6, 20, 100):
        for _ in range(2000):
            value = dice.roll(sides, rng)
            assert 1 <= value <= sides


def test_roll_is_uniform_seeded():
    rng = random.Random(42)
    counts = {i: 0 for i in range(1, 21)}
    n = 20000
    for _ in range(n):
        counts[dice.roll(20, rng)] += 1
    expected = n / 20
    # Every face should appear; none wildly off (loose bound, just sanity).
    for face, c in counts.items():
        assert c > 0, f"face {face} never rolled"
        assert abs(c - expected) < expected * 0.25


def test_roll_rejects_bad_sides():
    with pytest.raises(ValueError):
        dice.roll(0)


@pytest.mark.parametrize(
    "value,dc,expected",
    [(20, 20, True), (19, 20, False), (18, 18, True), (17, 18, False), (100, 1, True)],
)
def test_is_treat(value, dc, expected):
    assert dice.is_treat(value, dc) is expected


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    def advance(self, dt):
        self.t += dt


def test_rollgate_allows_once_then_blocks_within_window():
    clock = FakeClock()
    gate = dice.RollGate(cooldown_s=600, clock=clock)

    assert gate.allow() is True       # first roll always allowed
    assert gate.allow() is False      # immediately again -> blocked
    clock.advance(599)
    assert gate.allow() is False      # still inside the window
    clock.advance(1)
    assert gate.allow() is True       # exactly at the boundary -> allowed
    assert gate.allow() is False


def test_rollgate_seconds_remaining():
    clock = FakeClock()
    gate = dice.RollGate(cooldown_s=600, clock=clock)
    assert gate.seconds_remaining() == 0.0
    gate.allow()
    assert gate.seconds_remaining() == pytest.approx(600)
    clock.advance(100)
    assert gate.seconds_remaining() == pytest.approx(500)
    clock.advance(600)
    assert gate.seconds_remaining() == 0.0


def test_attempt_roll_blocks_on_cooldown():
    clock = FakeClock()
    gate = dice.RollGate(cooldown_s=600, clock=clock)
    rng = random.Random(7)

    first = dice.attempt_roll(gate, sides=20, dc=20, rng=rng)
    assert first.rolled is True
    assert 1 <= first.value <= 20

    second = dice.attempt_roll(gate, sides=20, dc=20, rng=rng)
    assert second.rolled is False
    assert second.value is None
    assert second.treat is False


def test_attempt_roll_treat_decision_is_consistent():
    # Force a known roll by using a gate that always allows and a seeded rng
    # whose first D20 value we compute up front.
    clock = FakeClock()
    rng_probe = random.Random(99)
    expected_value = dice.roll(20, random.Random(99))

    gate = dice.RollGate(cooldown_s=0, clock=clock)
    result = dice.attempt_roll(gate, sides=20, dc=expected_value, rng=random.Random(99))
    assert result.value == expected_value
    assert result.treat is True      # roll >= dc when dc == roll
    assert "rolled" in result.describe()
