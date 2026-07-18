"""Temporal score fusion ("track-before-detect"): confirm a cat from a string of
weak detections that no single frame could confirm.

The gap this closes: YOLO judges every frame independently, so a cat scoring 0.35
in eight consecutive frames — with the box gliding smoothly along a plausible path —
is discarded eight times. Persistence in *time* is evidence the single-frame
threshold throws away. This is the recall-raising mirror image of
``confirm_frames`` (which ANDs per-frame *successes* to raise precision): here,
per-frame *near-misses* are fused into one confirmation.

The rules (all pure math, all testable):

- A **hit** is a locator-class box scoring in the weak band (``weak_floor`` ≤ score
  < the normal ``cat_confidence``). Strong boxes also feed the fuser (they extend
  tracks) but they confirm on their own through the normal path — so a track that
  contains one is **not** confirmed here (``strong_at``, #110). Without that guard
  fusion took credit for obviously-strong cats and logged them as "N weak hits",
  which also made a genuine weak recovery indistinguishable from a misfire.
- Hits chain into a **track** when consecutive boxes overlap (IoU ≥ ``min_iou``)
  or their centroids sit within a box-diagonal of each other — a cat moves
  smoothly; teleporting hits are noise.
- A track **confirms** when, within the last ``window`` seconds, it has at least
  ``min_hits`` hits AND its net centroid travel is at least ``min_travel_frac`` of
  the frame diagonal. The travel requirement is the decoy guard: a cat-shaped
  cushion produces correlated weak hits that never go anywhere, and per #69/#70
  the one thing that separates a decoy from a cat is that the cat *moves*.
  (The deliberate flip side: a stationary weak cat never fuses — still cats are
  the averaging/tiling scan's job, 0.34.0/0.16.0.)
- After confirming, a track holds off (``reconfirm`` seconds) so one walking cat
  doesn't spam a sighting per frame.

Only YOLO evidence enters (0.33.0's rule holds: no VLM anywhere in this path);
the fusion is *accumulating* YOLO across time, not lowering the bar — a confirmed
track's score is the mean of its member hits, reported honestly as such.
"""

from __future__ import annotations

import math
import time

# Tuning (class-level so tests can read them; per-instance overrides via __init__).
WINDOW_SECS = 5.0        # hits older than this fall off the track
MIN_HITS = 4             # weak hits needed inside the window
MIN_IOU = 0.2            # chain overlap: consecutive hits must intersect this much…
MIN_TRAVEL_FRAC = 0.03   # …and the track must net-travel ≥ 3% of the frame diagonal
RECONFIRM_SECS = 30.0    # cooldown after a confirmation (per track)
MAX_TRACKS = 6           # a bounded handful — hits that fit no track start a new one
_MAX_EVENTS = 200        # debug records held in memory between drains (#110)


def _iou(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    return inter / float(area_a + area_b - inter or 1)


def _center(box):
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


def _dist(p, q) -> float:
    return math.hypot(p[0] - q[0], p[1] - q[1])


class TrackFuser:
    """Accumulate weak locator hits across frames; confirm consistent movers.

    ``update(ts, hits)`` takes this frame's locator-class boxes as
    ``[(score, (x1, y1, x2, y2))]`` (weak AND strong — strong hits extend tracks)
    and returns a confirmation dict or ``None``:

    ``{"box": last box, "score": mean score, "n": hits in window,
       "span_s": oldest→newest, "travel": net centroid px, "path": [centroids]}``
    """

    def __init__(self, frame_size=None, window: float = WINDOW_SECS,
                 min_hits: int = MIN_HITS, min_iou: float = MIN_IOU,
                 min_travel_frac: float = MIN_TRAVEL_FRAC,
                 reconfirm: float = RECONFIRM_SECS, strong_at: float = 0.0,
                 debug: bool = False):
        self.frame_size = frame_size          # (w, h); settable late (first frame)
        self.window = float(window)
        self.min_hits = int(min_hits)
        self.min_iou = float(min_iou)
        self.min_travel_frac = float(min_travel_frac)
        self.reconfirm = float(reconfirm)
        # A track containing a box this strong was already confirmed by the normal
        # path, so fusion must NOT take credit for it (#110). Settable late, like
        # frame_size, because cat_confidence is hot-reloadable. 0 = no guard.
        self.strong_at = float(strong_at)
        self.debug = bool(debug)              # opt-in diagnostics (#110)
        self.events: list = []                # drained by take_events()
        self._next_id = 1
        self._tracks: list[dict] = []         # {"id", "hits": [(ts, score, box, label)], "cooldown_until"}

    def _min_travel_px(self) -> float:
        if not self.frame_size:
            return 0.0
        w, h = self.frame_size
        return self.min_travel_frac * math.hypot(w, h)

    def _joins(self, track, box) -> bool:
        last = track["hits"][-1][2]
        if _iou(last, box) >= self.min_iou:
            return True
        # A fast mover can clear its own IoU between frames: allow a centroid jump
        # of up to one box diagonal.
        diag = math.hypot(last[2] - last[0], last[3] - last[1])
        return _dist(_center(last), _center(box)) <= diag

    def update(self, ts: float | None, hits) -> dict | None:
        """Feed one frame's locator hits; return a confirmation or ``None``.

        ``hits`` are ``(score, box)`` or ``(score, box, label)`` — the label is
        optional (kept for diagnostics, #110) so callers that don't track it still
        work.
        """
        now = time.time() if ts is None else float(ts)
        # Prune: drop hits outside the window, then empty tracks.
        for tr in self._tracks:
            tr["hits"] = [h for h in tr["hits"] if now - h[0] <= self.window]
        self._tracks = [tr for tr in self._tracks if tr["hits"]]

        for hit in sorted(hits or [], key=lambda h: -h[0]):
            score, box = hit[0], hit[1]
            label = hit[2] if len(hit) > 2 else ""
            box = tuple(int(v) for v in box)
            tr = next((t for t in self._tracks if self._joins(t, box)), None)
            if tr is None:
                if len(self._tracks) >= MAX_TRACKS:
                    continue                   # bounded: junk can't grow state
                tr = {"id": self._next_id, "hits": [], "cooldown_until": 0.0}
                self._next_id += 1
                self._tracks.append(tr)
            tr["hits"].append((now, float(score), box, label))

        best = None
        for tr in self._tracks:
            if now < tr["cooldown_until"] or len(tr["hits"]) < self.min_hits:
                continue
            path = [_center(h[2]) for h in tr["hits"]]
            travel = _dist(path[0], path[-1])
            min_travel = self._min_travel_px()
            # Did a box in this track already clear the normal single-frame bar?
            # If so the normal path confirmed it and fusion must stand down (#110):
            # otherwise an obviously-strong cat is logged as "N weak hits", and the
            # genuine weak-recovery case can't be told from a redundant misfire.
            strong = bool(self.strong_at) and any(h[1] >= self.strong_at
                                                  for h in tr["hits"])
            if strong or travel < min_travel:
                # strong → the normal path already confirmed it (#110);
                # short travel → decoy guard: no movement, no cat.
                if self.debug:
                    self._log_event(
                        "deferred_strong_box" if strong else "below_min_travel",
                        tr, travel, min_travel, strong, now)
                continue
            cand = {
                "box": list(tr["hits"][-1][2]),
                "score": round(sum(h[1] for h in tr["hits"]) / len(tr["hits"]), 3),
                "n": len(tr["hits"]),
                "span_s": round(tr["hits"][-1][0] - tr["hits"][0][0], 2),
                "travel": round(travel, 1),
                "path": [(round(x, 1), round(y, 1)) for x, y in path],
            }
            if best is None or cand["score"] > best[0]["score"]:
                best = (cand, tr, travel, min_travel)
        if best is None:
            return None
        cand, tr, travel, min_travel = best
        tr["cooldown_until"] = now + self.reconfirm
        if self.debug:
            self._log_event("confirmed", tr, travel, min_travel, False, now)
        return cand

    def _log_event(self, kind: str, tr: dict, travel: float, min_travel: float,
                   strong: bool, now: float) -> None:
        """Record one structured diagnostic record (#110). Opt-in via ``debug``;
        bounded so a long run can't grow memory without limit."""
        hits = tr["hits"]
        self.events.append({
            "ts": now,
            "event": kind,
            "track": tr.get("id"),
            "n": len(hits),
            "span_s": round(hits[-1][0] - hits[0][0], 2),
            "score_mean": round(sum(h[1] for h in hits) / len(hits), 3),
            "score_max": round(max(h[1] for h in hits), 3),
            "travel_px": round(travel, 1),
            "min_travel_px": round(min_travel, 1),
            "strong_at": self.strong_at,
            "strong_box_present": strong,
            "box": list(hits[-1][2]),
            "hits": [{"score": round(h[1], 3), "label": h[3], "box": list(h[2])}
                     for h in hits],
        })
        del self.events[:-_MAX_EVENTS]

    def take_events(self) -> list:
        """Drain the diagnostic records collected since the last call."""
        out, self.events = self.events, []
        return out
