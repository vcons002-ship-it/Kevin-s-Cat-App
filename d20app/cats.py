"""Cat sighting tracker: when and where a cat was last seen.

The app ignores cats for *rolling* (only people earn a treat), but rather than
drop them silently it records each sighting here so the GUI can answer "show me
the cat" — when it was seen, on which camera, roughly where in the frame, and an
annotated snapshot.

Storage is **one file per day** (``cats/YYYY-MM-DD.jsonl``) and unbounded: the
whole point of the log is answering "did she stay in that room all afternoon?",
which a rolling window of the last N sightings cannot do. Daily files keep that
affordable — startup reads only the newest day or two to fill the in-memory
window the GUI shows, and the full-log viewer opens exactly the day you asked
for instead of scanning a year of history to find it.

Disk I/O is best-effort throughout and never breaks the detection loop.

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

# One file per day under cats/, at the repo root next to config.yaml.
CATS_DIR = os.environ.get("D20_CATS_DIR", os.path.join(_REPO_ROOT, "cats"))
# The single-file store this replaced; split into daily files once, then left alone.
LEGACY_CATS_PATH = os.environ.get("D20_CATS_LOG", os.path.join(_REPO_ROOT, "cats.log"))
# How many recent sightings stay in memory. NOT a retention limit — the files keep
# everything. This only bounds what the GUI card and the time-of-day prior read,
# so startup stays constant-time however much history has accumulated.
MEMORY_SIGHTINGS = 2000


def day_key(ts: float) -> str:
    """The local calendar day a timestamp belongs to — the filename stem."""
    return time.strftime("%Y-%m-%d", time.localtime(ts))


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


def zone_for(box, zones, roi=None) -> str:
    """The named semantic zone a detection box's centre falls in, or ``""``.

    ``zones`` is the camera's ``[{name, box: [x, y, w, h], exit: bool}]`` list,
    drawn on the **full preview frame** (like the ROI). Detection boxes live in
    ROI-crop coordinates, so pass the camera's ``roi`` to shift them back into
    full-frame space before testing (#68). First matching zone wins.
    """
    if not zones:
        return ""
    x1, y1, x2, y2 = box
    ox, oy = (roi[0], roi[1]) if roi and len(roi) == 4 else (0, 0)
    cx, cy = (x1 + x2) / 2.0 + ox, (y1 + y2) / 2.0 + oy
    for z in zones:
        zb = z.get("box") if isinstance(z, dict) else None
        if not (zb and len(zb) == 4):
            continue
        zx, zy, zw, zh = zb
        if zx <= cx <= zx + zw and zy <= cy <= zy + zh:
            return str(z.get("name") or "")
    return ""


def box_in_exit_zone(box, zones, roi=None) -> bool:
    """True when a box's centre falls inside a zone marked ``exit`` (a doorway) —
    the precise version of the trail's "may have left the view" check (#68)."""
    exits = [z for z in zones or [] if isinstance(z, dict) and z.get("exit")]
    return bool(exits) and zone_for(box, exits, roi) != ""


class CatTracker:
    """Thread-safe, file-backed cat sightings — unbounded history, one file per day."""

    def __init__(self, directory: str = CATS_DIR,
                 memory: int = MEMORY_SIGHTINGS,
                 legacy_path: str = LEGACY_CATS_PATH) -> None:
        self.directory = directory
        self.memory = memory
        self.legacy_path = legacy_path
        self._lock = threading.Lock()
        self._sightings: deque = deque(maxlen=memory)
        try:
            os.makedirs(directory, exist_ok=True)
        except OSError:
            pass
        self._migrate_legacy()
        self._load()

    # -- persistence ---------------------------------------------------------
    def _day_path(self, key: str) -> str:
        return os.path.join(self.directory, f"{key}.jsonl")

    def days(self) -> list:
        """Every day that has sightings, newest first (``["2026-07-25", …]``)."""
        try:
            names = os.listdir(self.directory)
        except OSError:
            return []
        return sorted((n[:-6] for n in names if n.endswith(".jsonl")), reverse=True)

    @staticmethod
    def _read(path: str, limit: int | None = None) -> list:
        """Sightings from one day file, oldest first. ``limit`` keeps only the tail.

        The deque trick bounds MEMORY on a pathological day (Always-mode scanning
        can write thousands of rows) without needing to know the length first.
        """
        try:
            with open(path, "r", encoding="utf-8") as fh:
                lines = deque(fh, maxlen=limit) if limit else fh.readlines()
        except OSError:
            return []
        out = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                s = json.loads(line)
            except ValueError:
                continue                      # a torn final line survives a crash
            if isinstance(s, dict) and "ts" in s:
                out.append(s)
        return out

    def _load(self) -> None:
        """Fill the in-memory window from the newest days backwards.

        Constant-time in the size of the history: it stops as soon as the window
        is full, so a year of files costs the same as a week.
        """
        collected: list = []
        for key in self.days():
            need = self.memory - len(collected)
            if need <= 0:
                break
            collected = self._read(self._day_path(key), limit=need) + collected
        with self._lock:
            self._sightings.clear()
            self._sightings.extend(collected[-self.memory:])

    def _migrate_legacy(self) -> None:
        """Split a pre-0.64 ``cats.log`` into daily files, once.

        Renamed rather than deleted afterwards — losing someone's sighting history
        to a migration would be unforgivable, and the file is small.
        """
        path = self.legacy_path
        if not path or not os.path.exists(path):
            return
        if self.days():
            return                            # already migrated (or a fresh install)
        by_day: dict = {}
        for s in self._read(path):
            by_day.setdefault(day_key(s.get("ts", 0)), []).append(s)
        try:
            for key, items in by_day.items():
                with open(self._day_path(key), "a", encoding="utf-8") as fh:
                    for s in items:
                        fh.write(json.dumps(s) + "\n")
            os.replace(path, path + ".migrated")
        except OSError:
            pass

    def _append_to_file(self, sighting: dict) -> None:
        try:
            with open(self._day_path(day_key(sighting["ts"])), "a",
                      encoding="utf-8") as fh:
                fh.write(json.dumps(sighting) + "\n")
        except OSError:
            pass

    # -- public API ----------------------------------------------------------
    def record(self, camera: str, box, frame_size, score: float,
               image: str | None = None, ts: float | None = None,
               label: str = "cat", source: str = "yolo", zone: str = "") -> dict:
        """Store one cat sighting and persist it. Returns the stored record.

        ``label`` is the detected locator class (usually ``cat``; may be ``dog`` when
        a no-dog household opts into treating dogs as the cat). ``source`` says HOW it
        was found — ``yolo`` (the live loop), or an escalation rung (``zoom+yolo`` /
        ``vlm+yolo`` / ``vlm``, #66) — so real usage shows which rung earns its keep.
        Old records without either field read back as ``cat``/``yolo``.
        """
        x1, y1, x2, y2 = (int(v) for v in box)
        sighting = {
            "ts": time.time() if ts is None else float(ts),
            "camera": str(camera or ""),
            "label": str(label or "cat"),
            "source": str(source or "yolo"),
            "region": describe_region((x1, y1, x2, y2), frame_size),
            "box": [x1, y1, x2, y2],
            "score": round(float(score), 3),
        }
        if zone:
            sighting["zone"] = str(zone)      # semantic spot ("the couch"), #68
        if image:
            sighting["image"] = str(image)
        with self._lock:
            self._sightings.append(sighting)
            self._append_to_file(sighting)
        return sighting

    def by_hour(self, camera: str | None = None) -> list:
        """Sighting counts per local hour of day (24 ints) — the raw material of
        the time-of-day presence prior (#68). Optionally one camera's."""
        counts = [0] * 24
        with self._lock:
            sightings = list(self._sightings)
        for s in sightings:
            if camera and s.get("camera") != camera:
                continue
            counts[time.localtime(s.get("ts", 0)).tm_hour] += 1
        return counts

    def likely_cameras(self, hour: int | None = None) -> list:
        """Cameras ranked by historical presence around this hour (±1, wrapping) —
        a **prior** for ordering a Find-My-Cat sweep, never a tracked state (#68).
        ``[(camera, weight)]``, strongest first; empty when there's no history."""
        hour = time.localtime().tm_hour if hour is None else int(hour) % 24
        window = {(hour - 1) % 24, hour, (hour + 1) % 24}
        weights: dict = {}
        with self._lock:
            sightings = list(self._sightings)
        for s in sightings:
            cam = s.get("camera") or ""
            if not cam:
                continue
            if time.localtime(s.get("ts", 0)).tm_hour in window:
                weights[cam] = weights.get(cam, 0) + 1
        return sorted(weights.items(), key=lambda kv: -kv[1])

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

    def last_for(self, camera: str) -> dict | None:
        """Newest sighting recorded for ``camera``, or None. Drives the live
        feed's "last known location" overlay (0.42.0)."""
        with self._lock:
            for s in reversed(self._sightings):
                if s.get("camera") == camera:
                    return dict(s)
        return None

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
        """Wipe every sighting, in memory and on disk. Deliberately total: the GUI
        offers this as "Clear log", and leaving old days behind for the full-log
        viewer to still show would make the button a lie."""
        with self._lock:
            self._sightings.clear()
            for key in self.days():
                try:
                    os.remove(self._day_path(key))
                except OSError:
                    pass

    def day(self, key: str) -> list:
        """Every sighting for one calendar day, newest first — the full-log view.

        Reads exactly that day's file, so it costs the same whether the app has a
        week of history or five years.
        """
        if not key or any(c in key for c in "/\\.") or len(key) != 10:
            return []                          # a filename, not a path
        return list(reversed(self._read(self._day_path(key))))
