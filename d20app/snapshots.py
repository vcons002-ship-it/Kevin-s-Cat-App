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
MAX_FILES = 60


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
