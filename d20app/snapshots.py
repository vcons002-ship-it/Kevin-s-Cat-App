"""Saves annotated detection snapshots for the GUI to display.

Each saved file is a JPEG with detection boxes already drawn on it. The store
keeps only the most recent ``max_files`` so it can't fill the disk. Filenames
are returned to callers and served by the web app from :data:`SNAPSHOTS_DIR`.
"""

from __future__ import annotations

import os
import threading
import time

_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_PKG_DIR)
SNAPSHOTS_DIR = os.environ.get(
    "D20_SNAPSHOTS_DIR", os.path.join(_REPO_ROOT, "snapshots")
)
# 0 = keep everything. Sightings are unbounded (see cats.py) and each row shows
# its snapshot, so pruning these would leave older rows with missing pictures —
# exactly when you're scrolling back to see what actually happened.
MAX_FILES = 0


class SnapshotStore:
    """Write JPEG bytes to disk, newest-pruned, and hand back the filename."""

    def __init__(self, directory: str = SNAPSHOTS_DIR, max_files: int = MAX_FILES) -> None:
        self.directory = directory
        self.max_files = max_files
        self._lock = threading.Lock()
        self._counter = 0
        try:
            os.makedirs(directory, exist_ok=True)
        except OSError:
            pass

    def save(self, jpeg: bytes | None) -> str | None:
        """Persist ``jpeg`` and return its filename (or None on no data/error)."""
        if not jpeg:
            return None
        # Only the counter/filename allocation needs the lock; the blocking disk
        # write + prune run outside it so one camera worker's slow save doesn't
        # serialise the others (the filename is already unique).
        with self._lock:
            self._counter += 1
            name = f"snap_{int(time.time() * 1000)}_{self._counter}.jpg"
        try:
            with open(os.path.join(self.directory, name), "wb") as fh:
                fh.write(jpeg)
        except OSError:
            return None
        self._prune()
        return name

    def path(self, name: str) -> str:
        return os.path.join(self.directory, name)

    def _prune(self) -> None:
        if self.max_files <= 0:
            return                      # unbounded: a sighting keeps its picture
        try:
            files = [
                os.path.join(self.directory, f)
                for f in os.listdir(self.directory)
                if f.endswith(".jpg")
            ]
            files.sort(key=lambda p: os.path.getmtime(p))
            for old in files[: -self.max_files] if len(files) > self.max_files else []:
                try:
                    os.remove(old)
                except OSError:
                    pass
        except OSError:
            pass


SCREENSHOTS_DIR = os.environ.get(
    "D20_SCREENSHOTS_DIR", os.path.join(_REPO_ROOT, "screenshots")
)


class ScreenshotStore:
    """Screenshots the user asked for, by hand, from the live feed.

    Deliberately NOT a :class:`SnapshotStore`: those are automatic detection
    evidence and are pruned to a rolling window, which is right for something the
    app produced on its own. These were asked for on purpose, so nothing here is
    ever deleted — the app should not throw away what someone chose to keep.

    Filenames are ``YYYY-MM-DD_HH-MM-SS_Camera.jpg``: sortable, and readable
    against a clock, so a shot can be lined up with the same moment in the
    camera's own recordings without opening anything.
    """

    def __init__(self, directory: str = SCREENSHOTS_DIR) -> None:
        self.directory = directory
        self._lock = threading.Lock()
        try:
            os.makedirs(directory, exist_ok=True)
        except OSError:
            pass

    @staticmethod
    def _safe(camera: str) -> str:
        """A camera name reduced to something safe in a filename, never empty."""
        keep = [c if (c.isalnum() or c in "-_") else "-" for c in (camera or "camera")]
        return "".join(keep).strip("-") or "camera"

    @staticmethod
    def stamp() -> str:
        """A filename timestamp. Pass the SAME one to related saves so a set of
        images taken from one action sorts together instead of straddling a
        second boundary."""
        return time.strftime("%Y-%m-%d_%H-%M-%S")

    def save(self, jpeg: bytes | None, camera: str = "", suffix: str = "",
             stamp: str | None = None, ext: str = ".jpg") -> str | None:
        """Persist ``jpeg`` and return its filename (or None on no data/error).

        ``ext`` allows PNG for images that are evidence rather than illustration:
        a lossy copy of a frame is not the frame that was judged.
        """
        if not jpeg:
            return None
        base = f"{stamp or self.stamp()}_{self._safe(camera)}"
        if suffix:
            base = f"{base}_{self._safe(suffix)}"
        # Two shots in the same second must not overwrite each other; hold the lock
        # across the existence check AND the write so two requests can't pick the
        # same free name.
        with self._lock:
            name, n = f"{base}{ext}", 1
            while os.path.exists(os.path.join(self.directory, name)):
                n += 1
                name = f"{base}_{n}{ext}"
            try:
                with open(os.path.join(self.directory, name), "wb") as fh:
                    fh.write(jpeg)
            except OSError:
                return None
        return name

    def path(self, name: str) -> str:
        return os.path.join(self.directory, name)
