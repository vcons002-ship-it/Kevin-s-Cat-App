"""Dice rolling and rate-limiting — pure logic, no I/O, fully unit-testable.

The game rule: when a person is detected, we roll a die with ``sides`` faces
(D20, D100, ...). The cat gets a treat when the roll is greater than or equal to
the difficulty class ``dc`` (e.g. DC18 on a D20 -> treat on 18, 19 or 20).

A :class:`RollGate` enforces a cooldown so ordinary kitchen traffic doesn't
spam rolls. Both the random source and the clock are injectable so behaviour is
deterministic under test.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass


def roll(sides: int, rng: random.Random | None = None) -> int:
    """Roll a die and return a value in ``1..sides`` (inclusive).

    ``rng`` is injectable for deterministic tests; defaults to the module's
    shared, seeded-by-the-OS generator.
    """
    if sides < 1:
        raise ValueError(f"dice must have at least 1 side, got {sides}")
    r = rng if rng is not None else random
    return r.randint(1, sides)


def is_treat(roll_value: int, dc: int) -> bool:
    """True when ``roll_value`` meets or beats the difficulty class ``dc``."""
    return roll_value >= dc


@dataclass
class RollResult:
    """Outcome of a single roll attempt."""

    rolled: bool          # False when the cooldown gate blocked the attempt
    value: int | None     # the die result (None when not rolled)
    sides: int
    dc: int
    treat: bool           # True when value >= dc

    def describe(self) -> str:
        if not self.rolled:
            return "skipped (cooldown)"
        verdict = "TREAT!" if self.treat else "no treat"
        return f"rolled {self.value}/d{self.sides} vs DC{self.dc} -> {verdict}"


class RollGate:
    """Allow at most one roll per ``cooldown_s`` window.

    The clock is injectable (defaults to :func:`time.monotonic`) so tests can
    advance time without sleeping.
    """

    def __init__(self, cooldown_s: float, clock=time.monotonic) -> None:
        self.cooldown_s = float(cooldown_s)
        self._clock = clock
        self._last_allowed: float | None = None

    def allow(self) -> bool:
        """Return True if a roll is permitted now, recording the time if so."""
        now = self._clock()
        if self._last_allowed is None or (now - self._last_allowed) >= self.cooldown_s:
            self._last_allowed = now
            return True
        return False

    def seconds_remaining(self) -> float:
        """Seconds left before the next roll is permitted (0 when ready)."""
        if self._last_allowed is None:
            return 0.0
        elapsed = self._clock() - self._last_allowed
        return max(0.0, self.cooldown_s - elapsed)


def attempt_roll(
    gate: RollGate,
    sides: int,
    dc: int,
    rng: random.Random | None = None,
) -> RollResult:
    """Run the gate + roll + treat decision in one place.

    Returns a :class:`RollResult`; ``rolled`` is False when the cooldown blocked
    the attempt.
    """
    if not gate.allow():
        return RollResult(rolled=False, value=None, sides=sides, dc=dc, treat=False)
    value = roll(sides, rng)
    return RollResult(
        rolled=True,
        value=value,
        sides=sides,
        dc=dc,
        treat=is_treat(value, dc),
    )
