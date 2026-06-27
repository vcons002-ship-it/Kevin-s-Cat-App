"""Cat sighting tracker: when and where a cat was last seen.

The app ignores cats for *rolling* (only people earn a treat), but rather than
drop them silently it records each sighting here so the GUI can answer "show me
the cat" — when it was seen, on which camera, roughly where in the frame, and an
annotated snapshot.

Mirrors :class:`d20app.activitylog.ActivityLog`: a thread-safe, bounded,
file-backed list so the history survives a restart, with best-effort disk I/O
that never breaks the detection loop.

Each sighting carries a ``camera`` field even though the app watches one camera
today — that's the seam for the planned multi-camera "show cat" (switch the live
feed to whichever camera saw the cat).
"""

from __future__ import annotations

import json
import os
import threading
import time
from collections import deque

_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_PKG_DIR)

# cats.log lives at the repo root next to config.yaml (override for tests).
CATS_PATH = os.environ.get("D20_CATS_LOG", os.path.join(_REPO_ROOT, "cats.log"))
MAX_SIGHTINGS = 500


def describe_region(box, frame_size) -> str:
    """A human "where" for a box within a frame — e.g. ``"bottom-left"``.

    Splits the frame into thirds and names the cell the box's centre falls in:
    the row (top/middle/bottom) and column (left/center/right), joined as
    ``"middle-center"`` → ``"center"``. Returns ``""`` if the frame size is
    unknown.
    """
    if not frame_size:
        return ""
    w, h = frame_size
    if not w or not h:
        return ""
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    col = ("left", "center", "right")[min(2, max(0, int(cx / (w / 3.0))))]
    row = ("top", "middle", "bottom")[min(2, max(0, int(cy / (h / 3.0))))]
    if row == "middle" and col == "center":
        return "center"
    if row == "middle":
        return col
    if col == "center":
        return row
    return f"{row}-{col}"


class CatTracker:
    """A thread-safe, bounded, file-backed list of cat sightings."""

    def __init__(self, path: str = CATS_PATH, max_sightings: int = MAX_SIGHTINGS) -> None:
        self.path = path
        self.max_sightings = max_sightings
        self._lock = threading.Lock()
        self._sightings: deque = deque(maxlen=max_sightings)
        self._load()

    # -- persistence ---------------------------------------------------------
    def _load(self) -> None:
        if not self.path or not os.path.exists(self.path):
            return
        total = 0
        try:
            with open(self.path, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        s = json.loads(line)
                    except ValueError:
                        continue
                    if isinstance(s, dict) and "ts" in s:
                        total += 1
                        self._sightings.append(s)
        except OSError:
            return
        if total > self.max_sightings:
            self._rewrite()

    def _append_to_file(self, sighting: dict) -> None:
        if not self.path:
            return
        try:
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(sighting) + "\n")
        except OSError:
            pass

    def _rewrite(self) -> None:
        if not self.path:
            return
        try:
            with open(self.path, "w", encoding="utf-8") as fh:
                for s in self._sightings:
                    fh.write(json.dumps(s) + "\n")
        except OSError:
            pass

    # -- public API ----------------------------------------------------------
    def record(self, camera: str, box, frame_size, score: float,
               image: str | None = None, ts: float | None = None) -> dict:
        """Store one cat sighting and persist it. Returns the stored record."""
        x1, y1, x2, y2 = (int(v) for v in box)
        sighting = {
            "ts": time.time() if ts is None else float(ts),
            "camera": str(camera or ""),
            "region": describe_region((x1, y1, x2, y2), frame_size),
            "box": [x1, y1, x2, y2],
            "score": round(float(score), 3),
        }
        if image:
            sighting["image"] = str(image)
        with self._lock:
            self._sightings.append(sighting)
            self._append_to_file(sighting)
        return sighting

    def recent(self, limit: int | None = None) -> list:
        """Sightings, newest first."""
        with self._lock:
            items = list(self._sightings)
        items.reverse()
        if limit is not None:
            items = items[:limit]
        return items

    def last(self) -> dict | None:
        with self._lock:
            return dict(self._sightings[-1]) if self._sightings else None

    def count_since(self, ts: float) -> int:
        with self._lock:
            return sum(1 for s in self._sightings if s.get("ts", 0) >= ts)

    def today_count(self) -> int:
        """Sightings since local midnight."""
        now = time.localtime()
        midnight = time.mktime((now.tm_year, now.tm_mon, now.tm_mday,
                                0, 0, 0, 0, 0, -1))
        return self.count_since(midnight)

    def clear(self) -> None:
        with self._lock:
            self._sightings.clear()
            if self.path and os.path.exists(self.path):
                try:
                    open(self.path, "w").close()
                except OSError:
                    pass
