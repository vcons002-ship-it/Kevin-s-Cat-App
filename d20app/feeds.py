"""Follow mode: which camera does each live feed show? (#113)

The user has more rooms than feed slots, so when a cat walks out of frame they
switch cameras by hand. The app already knows where every cat is; this routes
that knowledge onto one or two feeds.

**The whole design is about NOT flickering.** The naive rule — "every feed shows
the most recent cat" — makes two cats in two rooms fight over both feeds, and lets
a cat crossing a doorway yank the feed off whatever you were watching. So:

- Each slot **adopts** a camera and **holds** it. Slots never compete for
  "most recent"; a slot's camera is its own until that camera goes quiet.
- A hold is released only after ``hold_s`` seconds with **no cat** on it. Any new
  detection on a held camera just refreshes the hold — it never causes a reassign.
- A freed slot adopts the most-recent cat-camera **not already held** by another
  slot, and only once that camera has been active for ``persist_s`` — so a brief
  doorway transient can't claim a feed.
- A secondary slot with nothing live to show falls back to the **previous room**
  (the camera most recently released), which is what makes room-to-room read as
  "new room primary, old room secondary". That fallback is never *held*: a real
  live cat always outranks a stale last-seen view.

Two cats in two rooms is therefore just the one-cat rule twice, independently —
which is exactly why the feeds stay put. All timing is injected (``now``) so the
behaviour is deterministic and testable.
"""

from __future__ import annotations

# Defaults; both are expected to want live tuning (#113), so they're config knobs.
HOLD_SECS = 3.0        # a held camera survives this long with no cat before it frees
PERSIST_SECS = 1.0     # a candidate must have been active this long to be adopted

LIVE = "live"          # slot is showing a camera that has a cat on it now
LAST_SEEN = "last-seen"  # secondary fallback: the room the cat just left


class FeedRouter:
    """Sticky, debounced camera→slot assignment. See the module docstring."""

    def __init__(self, hold_s: float = HOLD_SECS, persist_s: float = PERSIST_SECS,
                 max_recent: int = 8):
        self.hold_s = float(hold_s)
        self.persist_s = float(persist_s)
        self._max_recent = int(max_recent)
        self._slots: list[dict] = []      # {"camera", "last_active", "source"}
        self._since: dict[str, float] = {}   # camera -> when it most recently became active
        self._recent: list[str] = []      # released cameras, newest first ("previous room")

    def reset(self) -> None:
        """Forget all assignments (a new watch session starts clean)."""
        self._slots, self._since, self._recent = [], {}, []

    def _note_previous(self, camera: str) -> None:
        if not camera:
            return
        if camera in self._recent:
            self._recent.remove(camera)
        self._recent.insert(0, camera)
        del self._recent[self._max_recent:]

    def update(self, now: float, active: dict, slots: int = 1) -> list:
        """Assign cameras to ``slots`` feeds.

        ``active`` is ``{camera: last_seen}`` for cameras with a cat **right now**
        (any monotonic clock, same units as ``now``). Returns one dict per slot:
        ``{"camera": name or None, "source": "live" | "last-seen" | None}``.
        """
        slots = max(0, int(slots))
        active = active or {}

        # Track how long each camera has been continuously active, so a transient
        # can't be adopted. A camera that drops out loses its run.
        for cam in list(self._since):
            if cam not in active:
                del self._since[cam]
        for cam in active:
            self._since.setdefault(cam, now)

        # Grow/shrink to the requested slot count, keeping existing assignments.
        while len(self._slots) < slots:
            self._slots.append({"camera": None, "last_active": 0.0, "source": None})
        for dropped in self._slots[slots:]:
            self._note_previous(dropped["camera"])
        del self._slots[slots:]

        # Release pass. A LIVE hold survives `hold_s` of quiet; a LAST_SEEN filler
        # is re-decided every round so a real cat can always take the slot.
        for slot in self._slots:
            cam = slot["camera"]
            if not cam:
                continue
            if slot["source"] == LAST_SEEN:
                slot["camera"], slot["source"] = None, None
            elif cam in active:
                slot["last_active"] = now
            elif now - slot["last_active"] >= self.hold_s:
                self._note_previous(cam)
                slot["camera"], slot["source"] = None, None

        # Adopt pass, primary first.
        for index, slot in enumerate(self._slots):
            if slot["camera"]:
                continue
            taken = {s["camera"] for s in self._slots if s["camera"]}
            ready = [c for c in active
                     if c not in taken and now - self._since.get(c, now) >= self.persist_s]
            if ready:
                ready.sort(key=lambda c: active[c], reverse=True)   # most recent first
                slot.update(camera=ready[0], last_active=now, source=LIVE)
            elif index > 0:
                # Secondary with no live cat of its own: show the previous room.
                taken = {s["camera"] for s in self._slots if s["camera"]}
                prev = next((c for c in self._recent if c not in taken), None)
                if prev:
                    slot.update(camera=prev, last_active=now, source=LAST_SEEN)

        # Room-to-room must read as "new room primary, old room secondary". While
        # the primary is still *holding* a room the cat just left, a secondary can
        # legitimately adopt the room she walked into — so if the primary has no
        # live cat and a later slot does, promote it and demote the stale room.
        if self._slots and self._slots[0]["camera"] not in active:
            primary = self._slots[0]
            donor = next((s for s in self._slots[1:] if s["camera"] in active), None)
            if donor is not None:
                stale = primary["camera"]
                primary.update(camera=donor["camera"],
                               last_active=donor["last_active"], source=LIVE)
                if stale:
                    self._note_previous(stale)
                    donor.update(camera=stale, last_active=now, source=LAST_SEEN)
                else:
                    donor.update(camera=None, last_active=0.0, source=None)

        return [{"camera": s["camera"], "source": s["source"]} for s in self._slots]
