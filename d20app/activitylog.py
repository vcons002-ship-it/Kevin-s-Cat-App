"""Persistent activity log: the timestamped events the GUI shows.

The detection loop appends events (rolls, treats, errors, start/stop); the web
GUI reads them back via ``/api/log``. Entries are kept in a bounded in-memory
buffer **and** appended to a JSONL file so the history survives a restart.

Writing to disk must never break the detection loop, so all file I/O here is
best-effort: any OS error is swallowed rather than propagated.
"""

from __future__ import annotations

import json
import os
import threading
import time
from collections import deque

_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_PKG_DIR)

# activity.log lives at the repo root next to config.yaml (override for tests).
LOG_PATH = os.environ.get("D20_ACTIVITY_LOG", os.path.join(_REPO_ROOT, "activity.log"))

# Entry kinds the GUI colour-codes by. Anything else is treated as "info".
KINDS = ("info", "roll", "treat", "error")
MAX_ENTRIES = 1000


class ActivityLog:
    """A thread-safe, bounded, file-backed list of timestamped events."""

    def __init__(self, path: str = LOG_PATH, max_entries: int = MAX_ENTRIES) -> None:
        self.path = path
        self.max_entries = max_entries
        self._lock = threading.Lock()
        self._entries: deque = deque(maxlen=max_entries)
        self._load()

    # -- persistence ---------------------------------------------------------
    def _load(self) -> None:
        """Load the tail of the on-disk log; compact the file if it has grown."""
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
                        entry = json.loads(line)
                    except ValueError:
                        continue
                    if isinstance(entry, dict) and "ts" in entry and "message" in entry:
                        total += 1
                        self._entries.append(entry)  # deque drops oldest past maxlen
        except OSError:
            return
        # If the file held more than we keep in memory, rewrite it trimmed down
        # so it can't grow without bound across many restarts.
        if total > self.max_entries:
            self._rewrite()

    def _append_to_file(self, entry: dict) -> None:
        if not self.path:
            return
        try:
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry) + "\n")
        except OSError:
            pass

    def _rewrite(self) -> None:
        """Overwrite the file with the current (bounded) in-memory entries."""
        if not self.path:
            return
        try:
            with open(self.path, "w", encoding="utf-8") as fh:
                for entry in self._entries:
                    fh.write(json.dumps(entry) + "\n")
        except OSError:
            pass

    # -- public API ----------------------------------------------------------
    def add(self, kind: str, message: str, ts: float | None = None) -> dict:
        """Append an event and persist it. Returns the stored entry."""
        entry = {
            "ts": time.time() if ts is None else float(ts),
            "kind": kind if kind in KINDS else "info",
            "message": str(message),
        }
        with self._lock:
            self._entries.append(entry)
            self._append_to_file(entry)
        return entry

    def entries(self, limit: int | None = None, newest_first: bool = True) -> list:
        """Return a copy of the entries, newest first by default."""
        with self._lock:
            items = list(self._entries)
        if newest_first:
            items.reverse()
        if limit is not None:
            items = items[:limit]
        return items

    def clear(self) -> None:
        """Drop all entries from memory and truncate the file."""
        with self._lock:
            self._entries.clear()
            if self.path and os.path.exists(self.path):
                try:
                    open(self.path, "w").close()
                except OSError:
                    pass
