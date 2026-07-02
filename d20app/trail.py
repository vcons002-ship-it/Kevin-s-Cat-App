"""The "cat trail": null-frame silhouettes, recency-coloured, per motion episode.

Frame-to-frame differencing (the motion pre-filter) only lights up the *edges* of
movement. Diffing against the last **null frame** — the scene when nothing was
moving — lights up the **whole cat silhouette** wherever it differs from the empty
room. Painting each silhouette's pixels with the time they were last covered, then
colouring by recency (blue = where the episode started → red = the latest position),
yields a cat-shaped trail across the frame showing path, direction, and timing in
one image (#67).

Design notes, honestly:
- The null frame **re-adopts the scene** after it has been still for a few seconds.
  That self-heals slow lighting/auto-exposure drift (the classic failure of a fixed
  background) — and it also means a cat that settles becomes part of "null", so when
  she later leaves, the diff briefly lights her **old spot** as well as her real
  motion (a "ghost"). Ghosts fade into the trail's oldest colours and are bounded by
  the episode window; they're a known background-subtraction quirk, not a bug to
  chase.
- A **motion episode** is a run of movement separated by long stillness. The trail
  keeps rendering after the cat settles (that's when you want to look at it) and
  resets when a *new* episode begins.
- The **trail endpoint** — the centroid box of the newest silhouette — is a
  coordinate the app knows numerically. If it's interior (not near a frame edge),
  the cat plausibly *didn't leave*: the escalation ladder uses it as its
  highest-priority "look here" hint, and the endpoint drives the honest
  **"probable location"** state when detection still fails.
- Work happens at a capped resolution (`MAX_DIM`) — a fraction of a megabyte per
  camera and a few small-image ops per frame; boxes are mapped back to the
  detector's (ROI-cropped) frame coordinates.

Thread-safety: `update()` runs on the camera worker; `render()`/`endpoint()` on web
threads — everything mutable sits behind one small lock.
"""

from __future__ import annotations

import threading
import time

NULL_REFRESH_SECS = 5.0     # still this long → re-adopt the scene as "empty"
EPISODE_GAP_SECS = 30.0     # first motion after this much stillness starts a new trail
MAX_DIM = 640               # trail working resolution cap (memory + CPU)
MIN_SILHOUETTE_FRAC = 0.0005  # ignore microscopic masks (noise that survived cleanup)
EDGE_MARGIN_FRAC = 0.04     # endpoint within this of an edge = "may have exited"
ENDPOINT_FRESH_SECS = 600.0  # older endpoints stop driving hints / "probable"


class TrailTracker:
    """Accumulate null-frame silhouettes into a recency-coloured trail."""

    def __init__(self, diff_threshold: int = 25) -> None:
        self.diff_threshold = int(diff_threshold)
        self._lock = threading.Lock()
        self._scale = 1.0            # small-res → frame-coords multiplier
        self._shape = None           # (h, w) of the small working frames
        self._null = None            # blurred small gray of the "empty" scene
        self._ts_buf = None          # float64 (h, w): last time each pixel was covered
        self._episode_start = 0.0
        self._last_box = None        # newest silhouette bbox, small coords
        self._last_ts = 0.0
        self._last_motion_ts = 0.0
        self._still_since = None

    # -- internals -----------------------------------------------------------
    def _small(self, gray):
        import cv2

        h, w = gray.shape[:2]
        scale = max(h, w) / float(MAX_DIM)
        if scale <= 1.0:
            self._scale = 1.0
            return cv2.medianBlur(gray, 5)
        self._scale = scale
        small = cv2.resize(gray, (int(round(w / scale)), int(round(h / scale))))
        return cv2.medianBlur(small, 5)

    def _reset_for(self, shape) -> None:
        import numpy as np

        self._shape = shape
        self._ts_buf = np.zeros(shape, dtype=np.float64)
        self._episode_start = 0.0
        self._last_box = None
        self._last_ts = 0.0

    # -- update (camera worker thread) ----------------------------------------
    def update(self, gray, moved: bool, now: float | None = None) -> None:
        """Feed one (ROI-cropped, full-res) grayscale frame + the motion verdict."""
        import cv2
        import numpy as np

        now = time.time() if now is None else now
        with self._lock:
            clean = self._small(gray)
            if self._shape != clean.shape:
                self._reset_for(clean.shape)
                self._null = clean
                self._still_since = now
                return
            if self._null is None:
                self._null = clean
                self._still_since = now
                return

            if not moved:
                if self._still_since is None:
                    self._still_since = now
                elif now - self._still_since >= NULL_REFRESH_SECS:
                    # Re-adopt the still scene: heals lighting drift; absorbs a
                    # settled cat (see the module docstring's ghost note).
                    self._null = clean
                    self._still_since = now
                return

            # Motion. A long-enough quiet gap first means a NEW episode.
            if self._last_motion_ts and now - self._last_motion_ts > EPISODE_GAP_SECS:
                self._ts_buf[:] = 0.0
                self._episode_start = 0.0
                self._last_box = None
                self._last_ts = 0.0
            self._last_motion_ts = now
            self._still_since = None

            delta = cv2.absdiff(self._null, clean)
            _, mask = cv2.threshold(delta, self.diff_threshold, 255, cv2.THRESH_BINARY)
            mask = cv2.morphologyEx(
                mask, cv2.MORPH_OPEN,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
            ys, xs = np.nonzero(mask)
            if len(xs) < MIN_SILHOUETTE_FRAC * mask.size:
                return
            self._ts_buf[ys, xs] = now
            self._last_box = (int(xs.min()), int(ys.min()),
                              int(xs.max()) + 1, int(ys.max()) + 1)
            self._last_ts = now
            if not self._episode_start:
                self._episode_start = now

    # -- reads (web threads) ---------------------------------------------------
    def endpoint(self, now: float | None = None):
        """The trail's newest silhouette, or None: ``{"box"`` (frame coords),
        ``"ts", "age_s", "interior"}``. ``interior`` False when the silhouette
        touches the frame-edge margin — the cat may have exited the view."""
        now = time.time() if now is None else now
        with self._lock:
            if self._last_box is None or not self._last_ts:
                return None
            if now - self._last_ts > ENDPOINT_FRESH_SECS:
                return None
            h, w = self._shape
            mx, my = w * EDGE_MARGIN_FRAC, h * EDGE_MARGIN_FRAC
            x1, y1, x2, y2 = self._last_box
            interior = x1 > mx and y1 > my and x2 < w - mx and y2 < h - my
            s = self._scale
            return {"box": (int(x1 * s), int(y1 * s), int(x2 * s), int(y2 * s)),
                    "ts": self._last_ts, "age_s": round(now - self._last_ts, 1),
                    "interior": interior}

    def has_trail(self) -> bool:
        with self._lock:
            return bool(self._episode_start and self._last_ts)

    def render(self, frame_bgr, now: float | None = None):
        """The trail composited over ``frame_bgr`` (the detector's ROI-cropped
        frame): silhouette pixels tinted by recency (blue = episode start → red =
        newest), endpoint boxed. Returns a new BGR image, or None if no trail."""
        import cv2
        import numpy as np

        now = time.time() if now is None else now
        with self._lock:
            if not (self._episode_start and self._last_ts) or self._ts_buf is None:
                return None
            ts = self._ts_buf.copy()
            t0, t1 = self._episode_start, self._last_ts
            last_box, scale = self._last_box, self._scale

        covered = ts > 0
        span = max(t1 - t0, 1e-6)
        frac = np.clip((ts - t0) / span, 0.0, 1.0)      # 0 = oldest → 1 = newest
        hue = ((1.0 - frac) * 120.0).astype(np.uint8)   # 120 = blue → 0 = red (OpenCV H)
        hsv = np.dstack([hue, np.full(ts.shape, 255, np.uint8),
                         np.where(covered, 255, 0).astype(np.uint8)])
        tint = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        h, w = frame_bgr.shape[:2]
        tint = cv2.resize(tint, (w, h), interpolation=cv2.INTER_NEAREST)
        mask = cv2.resize(covered.astype(np.uint8) * 255, (w, h),
                          interpolation=cv2.INTER_NEAREST) > 0
        out = frame_bgr.copy()
        out[mask] = (0.45 * out[mask] + 0.55 * tint[mask]).astype(np.uint8)
        if last_box:
            x1, y1, x2, y2 = (int(v * scale) for v in last_box)
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(out, "latest", (x1, max(14, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        return out
