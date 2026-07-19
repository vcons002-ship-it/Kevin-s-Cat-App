"""Follow mode: which camera does each live feed show?

**Base model — recency, and it holds.** Primary shows the camera that saw a cat
most recently, secondary the next most recent. Those stick: further detections on
either changes nothing (they don't even swap places). Only a detection on a camera
*outside* the pair reshuffles it, and then it's simply "newest → primary, next
newest → secondary". A still-scan hit outside the pair is such a detection, so it
resets too.

An earlier design gated assignment on whether a camera had a cat *right now*, with
a hold timer and a debounce. That works for a cat walking between rooms and fails
for a sleeping one: "right now" is only `cat_scan_interval + 2` seconds wide (**2 s**
when the still-check is set to Always), so a sleeping cat dropped out between scans,
the slot freed, and the feed hid — then reappeared on the next scan. Recency has no
window: a room that saw a cat an hour ago still holds its slot.

**Two safeguards over the base model**, both off by default (0), for the case where
overlapping camera views make a cat near an edge trigger both:

- ``swap_confirm_count`` — how many detections a new camera needs before it may
  take a feed. 0/1 = the first detection swaps. Higher filters a transient pass
  through an overlap.
- ``camera_reuse_cooldown_seconds`` — how long a camera that just lost its slot is
  barred from taking one again. Breaks a two-camera oscillation, which the confirm
  count alone can't (both cameras keep genuinely detecting).

**Locks sit on top of everything.** A locked slot is pinned to its current camera:
never reassigned, never reset, and its camera is removed from the pool the unlocked
slots choose from — so the locked view can't be duplicated, and its ongoing
detections don't drag the other feed around. Locking pins the *view*, not the cat:
if she leaves, the locked feed keeps showing that (now empty) room until unlocked.
That's what makes "one cat asleep, one touring" workable — pin the sleeper, let the
other feed follow.
"""

from __future__ import annotations

import time

LIVE = "live"            # this camera has a cat on it right now
LAST_SEEN = "last-seen"  # holding the room where a cat was seen most recently


class FeedRouter:
    """Recency-based, sticky camera→slot assignment. See the module docstring."""

    def __init__(self, swap_confirm_count: int = 0,
                 camera_reuse_cooldown_seconds: float = 0.0):
        self.swap_confirm_count = int(swap_confirm_count)
        self.camera_reuse_cooldown_seconds = float(camera_reuse_cooldown_seconds)
        self._pair: list = []            # camera per slot (None = empty)
        self._locked: list = []          # lock flag per slot
        self._prev: dict = {}            # last-seen snapshot, to spot NEW detections
        self._pending: dict = {}         # camera -> detections counted toward a swap
        self._released_at: dict = {}     # camera -> when it last lost a slot

    def reset(self) -> None:
        """Forget everything (a new watch session starts clean)."""
        self._pair, self._locked = [], []
        self._prev, self._pending, self._released_at = {}, {}, {}

    def _fit(self, slots: int) -> None:
        while len(self._pair) < slots:
            self._pair.append(None)
            self._locked.append(False)
        del self._pair[slots:]
        del self._locked[slots:]

    def _reusable(self, cam: str, now: float) -> bool:
        """False while a just-displaced camera is serving its cooldown."""
        if self.camera_reuse_cooldown_seconds <= 0:
            return True
        released = self._released_at.get(cam)
        return released is None or (now - released) >= self.camera_reuse_cooldown_seconds

    def update(self, last_seen: dict, slots: int = 1, present=None,
               locks=None, now: float | None = None) -> list:
        """Assign cameras to ``slots`` feeds.

        ``last_seen`` is ``{camera: timestamp}`` of when each camera last saw a cat
        — **not** windowed, so a quiet room keeps its place. ``present`` is the
        cameras with a cat right now, used only to label a slot. ``locks`` is an
        iterable of slot indices the user has pinned. Returns one
        ``{"camera", "source", "locked"}`` per slot.
        """
        slots = max(0, int(slots))
        now = time.monotonic() if now is None else float(now)
        seen = {c: t for c, t in (last_seen or {}).items() if t}
        present = set(present or ())
        self._fit(slots)
        for i in range(slots):
            self._locked[i] = i in set(locks or ())

        # A camera whose timestamp advanced since the last poll saw something new.
        new_hits = {c for c, t in seen.items() if t > self._prev.get(c, 0.0)}
        self._prev = dict(seen)

        locked_cams = {c for c, lk in zip(self._pair, self._locked) if lk and c}
        unlocked = [i for i in range(slots) if not self._locked[i]]

        # An unlocked slot whose camera is no longer watched gives it up.
        for i in unlocked:
            if self._pair[i] and self._pair[i] not in seen:
                self._released_at[self._pair[i]] = now
                self._pair[i] = None

        # Don't leave a hole above a filled slot: if the primary's camera stopped
        # being watched, the secondary's should move up rather than the main feed
        # sitting blank underneath a working one. Only unlocked slots shuffle, and
        # this never reorders two occupied slots — it just closes gaps.
        occupied = [self._pair[i] for i in unlocked if self._pair[i]]
        for pos, i in enumerate(unlocked):
            self._pair[i] = occupied[pos] if pos < len(occupied) else None

        held = {c for c in self._pair if c}
        # Locked cameras are out of the running for the other feeds entirely.
        pool = {c: t for c, t in seen.items() if c not in locked_cams}
        by_recency = sorted(pool, key=lambda c: pool[c], reverse=True)

        # Count detections on cameras that aren't currently shown; anything already
        # on a feed isn't a swap candidate, so its tally is irrelevant.
        for cam in new_hits:
            if cam in held or cam in locked_cams:
                self._pending.pop(cam, None)
            else:
                self._pending[cam] = self._pending.get(cam, 0) + 1

        need = max(1, self.swap_confirm_count)
        trigger = next((c for c in by_recency
                        if c not in held
                        and self._pending.get(c, 0) >= need
                        and self._reusable(c, now)), None)

        if trigger is not None and unlocked:
            # Reshuffle the UNLOCKED slots only: the trigger takes the first of
            # them, the rest fall in by recency. Locked slots never move.
            order = [trigger] + [c for c in by_recency
                                 if c != trigger and self._reusable(c, now)]
            for pos, i in enumerate(unlocked):
                cam = order[pos] if pos < len(order) else None
                if self._pair[i] and self._pair[i] != cam:
                    self._released_at[self._pair[i]] = now
                self._pair[i] = cam
            self._pending.clear()

        # Fill any still-empty unlocked slot (first run, or the second feed was just
        # switched on). This isn't a swap, so it doesn't need confirmations — but it
        # does respect the cooldown, or a just-displaced camera would walk straight
        # back in through the side door.
        taken = {c for c in self._pair if c}
        for i in unlocked:
            if self._pair[i]:
                continue
            cam = next((c for c in by_recency
                        if c not in taken and self._reusable(c, now)), None)
            if cam:
                self._pair[i] = cam
                taken.add(cam)

        return [{"camera": self._pair[i],
                 "source": (None if not self._pair[i]
                            else LIVE if self._pair[i] in present else LAST_SEEN),
                 "locked": bool(self._locked[i])}
                for i in range(slots)]
