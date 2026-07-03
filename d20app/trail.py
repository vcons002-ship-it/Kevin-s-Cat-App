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
# Real-hardware guards (0.39.0 — from the first NAS screenshot, which was a wall of
# green): an auto-exposure/white-balance/lighting shift makes the WHOLE frame differ
# from the null, flood-stamping the room as one giant "silhouette".
GLOBAL_CHANGE_FRAC = 0.35   # diff covering more than this = lighting event, not a cat → re-adopt null, stamp nothing
MAX_SILHOUETTE_FRAC = 0.25  # a single blob bigger than this isn't an animal either
MAX_BLOBS = 4               # keep only the largest few plausible blobs per frame
PATH_MAX_POINTS = 400       # centroid path memory per episode (draws the route line)
TINT_ALPHA = 0.35           # silhouette tint strength (0.55 was unreadable on real footage)


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
        self._path = []

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
        self._path = []              # [(ts, cx, cy)] silhouette centroids, small coords

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

            delta = cv2.absdiff(self._null, clean)
            _, mask = cv2.threshold(delta, self.diff_threshold, 255, cv2.THRESH_BINARY)
            mask = cv2.morphologyEx(
                mask, cv2.MORPH_OPEN,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))

            # Global-change guard (0.39.0): a diff covering most of the frame is a
            # lighting/exposure/white-balance event, not an animal — the first NAS
            # run flood-stamped the entire room green this way. Re-adopt the scene
            # as the new null and stamp NOTHING.
            if np.count_nonzero(mask) > GLOBAL_CHANGE_FRAC * mask.size:
                self._null = clean
                self._still_since = now
                return

            # Motion. A long-enough quiet gap first means a NEW episode.
            if self._last_motion_ts and now - self._last_motion_ts > EPISODE_GAP_SECS:
                self._ts_buf[:] = 0.0
                self._episode_start = 0.0
                self._last_box = None
                self._last_ts = 0.0
                self._path = []
            self._last_motion_ts = now
            self._still_since = None

            # Silhouette hygiene (0.39.0): keep only plausibly-animal-sized blobs —
            # a cat is neither 3 pixels nor a quarter of the room — and only the
            # largest few, so the trail is shapes, not confetti.
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
            lo, hi = MIN_SILHOUETTE_FRAC * mask.size, MAX_SILHOUETTE_FRAC * mask.size
            blobs = sorted((c for c in contours if lo <= cv2.contourArea(c) <= hi),
                           key=cv2.contourArea, reverse=True)[:MAX_BLOBS]
            if not blobs:
                return
            kept = np.zeros_like(mask)
            cv2.drawContours(kept, blobs, -1, 255, thickness=-1)
            ys, xs = np.nonzero(kept)
            if not len(xs):
                return
            self._ts_buf[ys, xs] = now
            self._last_box = (int(xs.min()), int(ys.min()),
                              int(xs.max()) + 1, int(ys.max()) + 1)
            self._last_ts = now
            if not self._episode_start:
                self._episode_start = now
            # The route line: one centroid per stamped frame, bounded per episode.
            self._path.append((now, float(xs.mean()), float(ys.mean())))
            if len(self._path) > PATH_MAX_POINTS:
                del self._path[:len(self._path) - PATH_MAX_POINTS]

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

    @staticmethod
    def _ramp_bgr(frac: float) -> tuple:
        """The recency colour for ``frac`` in [0, 1]: 0 = oldest (blue, OpenCV hue
        120) sweeping to 1 = newest (red, hue 0)."""
        import cv2
        import numpy as np

        hue = int(round((1.0 - max(0.0, min(1.0, frac))) * 120.0))
        px = np.uint8([[[hue, 255, 255]]])
        b, g, r = cv2.cvtColor(px, cv2.COLOR_HSV2BGR)[0, 0]
        return (int(b), int(g), int(r))

    def render(self, frame_bgr, now: float | None = None):
        """The trail composited over ``frame_bgr`` (the detector's ROI-cropped
        frame): silhouette pixels tinted by recency (blue = episode start → red =
        newest, at :data:`TINT_ALPHA`), the centroid **route line** drawn in the
        same ramp, the endpoint boxed, and a colour legend. Returns a new BGR
        image, or None if no trail."""
        import cv2
        import numpy as np

        now = time.time() if now is None else now
        with self._lock:
            if not (self._episode_start and self._last_ts) or self._ts_buf is None:
                return None
            ts = self._ts_buf.copy()
            t0, t1 = self._episode_start, self._last_ts
            last_box, scale = self._last_box, self._scale
            path = list(self._path)

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
        out[mask] = ((1.0 - TINT_ALPHA) * out[mask]
                     + TINT_ALPHA * tint[mask]).astype(np.uint8)

        # The route line: silhouette centroids joined in time order, each segment
        # coloured by its (newer) endpoint's age — the "where did she go" answer
        # at a glance, readable even where the tint is subtle.
        for (ta, xa, ya), (tb, xb, yb) in zip(path, path[1:]):
            col = self._ramp_bgr((tb - t0) / span)
            cv2.line(out, (int(xa * scale), int(ya * scale)),
                     (int(xb * scale), int(yb * scale)), col, 2, cv2.LINE_AA)
        for (tp, xp, yp) in path[:1] + path[-1:]:
            cv2.circle(out, (int(xp * scale), int(yp * scale)), 4,
                       self._ramp_bgr((tp - t0) / span), -1, cv2.LINE_AA)

        if last_box:
            x1, y1, x2, y2 = (int(v * scale) for v in last_box)
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(out, "latest", (x1, max(14, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

        # Legend (bottom-left): the colour ramp with its meaning, so the image
        # explains itself. Age span shown so "blue" has a number.
        bar_w, bar_h, pad = min(140, w - 20), 10, 8
        by = h - pad - bar_h
        cv2.rectangle(out, (pad - 3, by - 22), (pad + bar_w + 60, h - pad + 3),
                      (0, 0, 0), -1)
        for i in range(bar_w):
            cv2.line(out, (pad + i, by), (pad + i, by + bar_h),
                     self._ramp_bgr(i / max(1, bar_w - 1)), 1)
        cv2.putText(out, f"path: old -> new ({int(round(span))}s)",
                    (pad, by - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(out, "now", (pad + bar_w + 6, by + bar_h),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
        return out
