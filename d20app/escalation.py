"""VLM-assisted detection escalation: motion-guided zoom/crop + a cheap-first ladder.

The problem this solves: a small/distant cat shrinks below the net's resolving power
when the full frame is letterboxed to the model input (#17's 0.00-confidence case).
The fix is what a human does — **zoom in where it makes sense and look again**:

1. **zoom+yolo** (free, no VLM): full-resolution square crops around the "look here"
   hints — where motion just happened (`MotionPrefilter.last_blobs`), where the cat
   was last seen (`CatTracker.recent()`), and where a CPU predictor says a *moving*
   cat is headed (:func:`predict_hint_box`) — re-checked by the ordinary YOLO net.
   In a crop the cat is big again; most misses should die here.
2. **vlm detect**: moondream's `detect("cat")` proposes regions on the full frame;
   each is cropped at full resolution and confirmed by **YOLO** — a bare detect
   region is never trusted alone (open-vocabulary detectors false-fire on
   cat-shaped decoys).
3. **vlm query**: last resort — the voted yes/no question on the crops.

**Only YOLO confirms.** The maintainer's decoy benchmarks (issue #69) measured
VLMs at 37–42% false positives; majority voting reduces run-to-run *variance*,
not that systematic *bias* — three votes just agree wrongly. So a votes-only
"yes" (rung 2's query fallback, all of rung 3) can never produce ``found=True``:
it is returned as ``vlm_probable`` — a lead for the caller to surface as
"probable" and to hand YOLO for confirmation — never a recorded fact.

Cheapest rung first; stop at the first YOLO-confirmed hit. The ladder takes its detectors
as **injected callables**, so every decision here is pure, deterministic, and tested
without a net or a GPU. All boxes are ``(x1, y1, x2, y2)`` ints in the same
(adjusted, ROI-cropped) frame coordinates the rest of the app uses; only
:func:`map_normalized_box` speaks moondream's normalized-[0,1] dialect.

This module runs **on demand** (the `/api/vlm/escalate` endpoint) — never inside the
fast treat path.
"""

from __future__ import annotations

import time

# Crop sizing: pad hints generously (a motion blob is often just part of the cat),
# never hand the VLM a postage stamp (< MIN_CROP) or a whole wall (> MAX_CROP), and
# bound the work one request can do.
PAD_FRAC = 0.35
MIN_CROP = 320
MAX_CROP = 1024
MAX_CROPS = 4

# predict_hint_box: a track older than this is stale — a cat can cross a room in
# far less, so extrapolating from it is guessing, not predicting.
PREDICT_MAX_AGE = 10.0


def clamp_box(box, frame_size) -> tuple:
    """Clip ``(x1, y1, x2, y2)`` to a ``(w, h)`` frame, returning ints."""
    w, h = frame_size
    x1, y1, x2, y2 = (int(round(v)) for v in box)
    return (max(0, min(x1, w)), max(0, min(y1, h)),
            max(0, min(x2, w)), max(0, min(y2, h)))


def _box_area(box) -> int:
    x1, y1, x2, y2 = box
    return max(0, x2 - x1) * max(0, y2 - y1)


def _iou(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    return inter / float(_box_area(a) + _box_area(b) - inter)


def merge_hint_boxes(boxes, iou_thresh: float = 0.4, max_boxes: int = MAX_CROPS) -> list:
    """Dedupe hint boxes: drop any box that overlaps (IoU) or is contained by an
    already-kept larger one; keep at most ``max_boxes``, largest area first.

    Hints come from several sources (motion blobs, last sighting, prediction) that
    often point at the same spot — one crop there is enough.
    """
    out = []
    for box in sorted((b for b in boxes if b and _box_area(b) > 0),
                      key=_box_area, reverse=True):
        contained = any(
            box[0] >= k[0] and box[1] >= k[1] and box[2] <= k[2] and box[3] <= k[3]
            for k in out)
        if contained or any(_iou(box, k) >= iou_thresh for k in out):
            continue
        out.append(tuple(int(v) for v in box))
        if len(out) >= max_boxes:
            break
    return out


def square_crop_box(box, frame_size, pad_frac: float = PAD_FRAC,
                    min_size: int = MIN_CROP, max_size: int = MAX_CROP) -> tuple:
    """The square window to cut around a hint box, in frame coords.

    Pads the hint by ``pad_frac`` on each side (a motion blob is often just the
    moving *part* of the cat), expands the shorter side to a square, enforces
    ``min_size``/``max_size``, and clamps to the frame by **shifting** the window
    rather than shrinking it — a hint at the frame edge still gets a full-size crop.
    """
    w, h = frame_size
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    side = max(bw, bh) * (1 + 2 * pad_frac)
    side = int(round(min(max(side, min_size), max_size)))
    side = min(side, w, h)                      # a small frame caps the window
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    nx1 = int(round(cx - side / 2.0))
    ny1 = int(round(cy - side / 2.0))
    nx1 = max(0, min(nx1, w - side))            # shift, don't shrink, at the edges
    ny1 = max(0, min(ny1, h - side))
    return (nx1, ny1, nx1 + side, ny1 + side)


def square_crops(frame, hint_boxes, **kw) -> list:
    """Cut padded square crops around hints: ``[{"image": view, "box": (x1,y1,x2,y2)}]``.

    ``box`` is the crop window in frame coords (what :func:`map_box_to_frame` maps
    detections back through); ``image`` is a numpy view, not a copy.
    """
    h, w = frame.shape[:2]
    crops = []
    for hint in hint_boxes:
        cb = square_crop_box(hint, (w, h), **kw)
        x1, y1, x2, y2 = cb
        if x2 - x1 < 2 or y2 - y1 < 2:
            continue
        crops.append({"image": frame[y1:y2, x1:x2], "box": cb})
    return crops


def map_box_to_frame(box_in_crop, crop_box) -> tuple:
    """Translate a detection box from crop coords back into frame coords."""
    ox, oy = crop_box[0], crop_box[1]
    x1, y1, x2, y2 = box_in_crop
    return (int(x1 + ox), int(y1 + oy), int(x2 + ox), int(y2 + oy))


def map_normalized_box(nbox: dict, frame_size):
    """moondream Region ``{x_min..y_max}`` in normalized [0,1] → pixel box, or None.

    Clamps to the frame and rejects degenerate (<2 px a side) regions — a sliver
    from the VLM is noise, not a cat.
    """
    w, h = frame_size
    try:
        box = (float(nbox["x_min"]) * w, float(nbox["y_min"]) * h,
               float(nbox["x_max"]) * w, float(nbox["y_max"]) * h)
    except (KeyError, TypeError, ValueError):
        return None
    box = clamp_box(box, frame_size)
    if box[2] - box[0] < 2 or box[3] - box[1] < 2:
        return None
    return box


def predict_hint_box(tracks, now: float | None = None, frame_size=None,
                     max_age: float = PREDICT_MAX_AGE):
    """CPU "cat targeting": extrapolate where a *moving* cat is now, or None.

    ``tracks`` is ``[(ts, (x1, y1, x2, y2)), ...]`` — recent timestamped fixes
    (motion blobs and/or sightings), any order. Uses the two most recent fixes for
    a centroid velocity, extrapolates to ``now``, sizes the box like the last fix
    **inflated by staleness** (the longer since the last fix, the wider the
    uncertainty), and clamps to the frame. Returns None when there aren't 2 fixes,
    the track is stale (> ``max_age`` s), or the fixes are simultaneous — the
    "not enough data → the VLM is the failsafe" behaviour. A sleeping cat has no
    velocity: prediction helps the moving case only.
    """
    fixes = sorted((t for t in tracks or [] if t and t[1]), key=lambda t: t[0])
    if len(fixes) < 2:
        return None
    now = time.time() if now is None else now
    (t0, b0), (t1, b1) = fixes[-2], fixes[-1]
    if now - t1 > max_age or t1 <= t0:
        return None
    c0 = ((b0[0] + b0[2]) / 2.0, (b0[1] + b0[3]) / 2.0)
    c1 = ((b1[0] + b1[2]) / 2.0, (b1[1] + b1[3]) / 2.0)
    dt = t1 - t0
    vx, vy = (c1[0] - c0[0]) / dt, (c1[1] - c0[1]) / dt
    lead = now - t1
    px, py = c1[0] + vx * lead, c1[1] + vy * lead
    bw, bh = b1[2] - b1[0], b1[3] - b1[1]
    # Uncertainty pad grows with staleness: +25% of the box per second unseen.
    grow = 1.0 + 0.25 * lead
    hw, hh = bw * grow / 2.0, bh * grow / 2.0
    box = (px - hw, py - hh, px + hw, py + hh)
    if frame_size:
        box = clamp_box(box, frame_size)
        if box[2] - box[0] < 2 or box[3] - box[1] < 2:
            return None
        return box
    return tuple(int(round(v)) for v in box)


MOSAIC_MAX_TILES = 9


def frame_mosaic(frames, tile: int = 320):
    """Tile timestamped frames into one numbered grid image, oldest first (#68).

    moondream reads single images only — no video input — so temporal questions
    ("did a cat pass through?") are asked over a **mosaic** of the last few frames,
    each tile numbered and stamped with its age relative to the newest. Returns a
    BGR grid (≤ :data:`MOSAIC_MAX_TILES` tiles, square-ish layout), or None when
    there are no frames. ``frames`` is ``[(ts, bgr)]`` in any order.
    """
    import cv2
    import numpy as np

    frames = sorted((f for f in frames or [] if f and f[1] is not None),
                    key=lambda t: t[0])[-MOSAIC_MAX_TILES:]
    n = len(frames)
    if not n:
        return None
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    newest = frames[-1][0]
    tiles = []
    for i, (ts, f) in enumerate(frames):
        h, w = f.shape[:2]
        s = tile / float(max(h, w))
        r = cv2.resize(f, (max(1, int(w * s)), max(1, int(h * s))))
        canvas = np.zeros((tile, tile, 3), np.uint8)
        y, x = (tile - r.shape[0]) // 2, (tile - r.shape[1]) // 2
        canvas[y:y + r.shape[0], x:x + r.shape[1]] = r
        age = newest - ts
        label = f"{i + 1} (now)" if age < 0.5 else f"{i + 1} (-{age:.0f}s)"
        cv2.putText(canvas, label, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 2, cv2.LINE_AA)
        tiles.append(canvas)
    while len(tiles) < rows * cols:
        tiles.append(np.zeros((tile, tile, 3), np.uint8))
    return np.vstack([np.hstack(tiles[r * cols:(r + 1) * cols])
                      for r in range(rows)])


def _best_locator_box(dets, locator_classes, floor):
    """Strongest locator-class detection ``(label, score, box)`` from a YOLO result
    list, or None. ``dets`` is ``[(label, score, (x1,y1,x2,y2))]``."""
    best = None
    for lab, score, box in dets or []:
        if lab in locator_classes and score >= floor:
            if best is None or score > best[1]:
                best = (lab, score, box)
    return best


def escalate(frame, hints=(), *, run_yolo, vlm_detect=None, vlm_query=None,
             locator_classes=("cat",), cat_confidence: float = 0.5,
             max_crops: int = MAX_CROPS, query_passes: int = 3) -> dict:
    """Run the cheap-first escalation ladder on one frame. See the module docstring.

    ``run_yolo(img) -> [(label, score, box-in-img)]`` is required;
    ``vlm_detect(img) -> [normalized Region dicts]`` and
    ``vlm_query(img) -> {"answer", "ratio", ...}`` are optional — None marks that
    rung unavailable (moondream absent / disabled) and it is skipped with a note.
    Bounded: ≤ ``max_crops`` crops per rung, exactly one ``vlm_detect`` call, and
    voted queries stop after the first "yes" (it can only ever yield one lead).

    ``found=True`` requires a YOLO confirmation (source ``zoom+yolo``/``vlm+yolo``).
    A votes-only VLM "yes" comes back in ``vlm_probable`` (box/score/ratio) with
    ``found=False`` — an unconfirmed lead, per #69's decoy false-positive data.
    """
    h, w = frame.shape[:2]
    frame_size = (w, h)
    locator_classes = tuple(locator_classes) or ("cat",)
    rungs, all_crops = [], []
    vlm_probable = None

    def _finish(found, label=None, source=None, box=None, score=0.0, ratio=None):
        return {"found": found, "label": label, "source": source,
                "box": list(box) if box else None, "score": round(float(score), 3),
                "ratio": ratio, "vlm_probable": vlm_probable,
                "rungs": rungs, "crops": all_crops}

    # -- rung 1: zoom + yolo on the hint crops (no VLM) -----------------------
    hint_boxes = merge_hint_boxes(hints, max_boxes=max_crops)
    crops = square_crops(frame, hint_boxes)
    t0 = time.perf_counter()
    hit = None
    for c in crops:
        all_crops.append({"box": list(c["box"]), "source_rung": "zoom+yolo"})
        best = _best_locator_box(run_yolo(c["image"]), locator_classes, cat_confidence)
        if best:
            hit = (best[0], best[1], map_box_to_frame(best[2], c["box"]))
            break
    rungs.append({"name": "zoom+yolo", "ran": bool(crops),
                  "ms": round((time.perf_counter() - t0) * 1000.0, 1),
                  "crops": len(crops), "found": bool(hit),
                  "note": "" if crops else "no hints (no recent motion/sighting)"})
    if hit:
        return _finish(True, hit[0], "zoom+yolo", hit[2], hit[1])

    # -- rung 2: vlm detect proposes regions; YOLO/query confirm --------------
    t0 = time.perf_counter()
    regions = []
    if vlm_detect is None:
        rungs.append({"name": "vlm detect", "ran": False, "ms": 0.0, "crops": 0,
                      "found": False, "note": "VLM unavailable"})
    else:
        try:
            raw_regions = vlm_detect(frame) or []
        except Exception as exc:        # noqa: BLE001 — a dead VLM shouldn't kill rung 3
            raw_regions, note = [], f"detect failed: {exc}"
        else:
            note = "" if raw_regions else "no regions proposed"
        regions = [b for b in (map_normalized_box(r, frame_size) for r in raw_regions)
                   if b][:max_crops]
        found_here = None
        for rb in regions:
            cb = square_crop_box(rb, frame_size)
            x1, y1, x2, y2 = cb
            crop_img = frame[y1:y2, x1:x2]
            all_crops.append({"box": list(cb), "source_rung": "vlm detect"})
            best = _best_locator_box(run_yolo(crop_img), locator_classes, cat_confidence)
            if best:
                found_here = ("vlm+yolo", best[0], best[1],
                              map_box_to_frame(best[2], cb), None)
                break
            # A votes-only "yes" is a LEAD, not a find (#69): remember the first
            # one and keep going — a YOLO confirm on a later region beats it.
            if vlm_query is not None and vlm_probable is None:
                q = vlm_query(crop_img)
                if q and q.get("answer") == "yes":
                    votes = q.get("votes") or {}
                    n = q.get("passes") or 1
                    score = (votes.get("yes", 0) / n) if n else 0.0
                    vlm_probable = {"box": list(rb), "score": round(score, 3),
                                    "ratio": q.get("ratio"), "rung": "vlm detect"}
        if vlm_probable and not found_here:
            note = (note + "; " if note else "") + "VLM voted yes — unconfirmed"
        rungs.append({"name": "vlm detect", "ran": True,
                      "ms": round((time.perf_counter() - t0) * 1000.0, 1),
                      "crops": len(regions), "found": bool(found_here), "note": note})
        if found_here:
            src, lab, score, box, ratio = found_here
            return _finish(True, lab, src, box, score, ratio)

    # -- rung 3: vlm query on the hint crops (only when detect had no regions) --
    # Votes-only, so its best outcome is a LEAD (vlm_probable), never a find (#69).
    t0 = time.perf_counter()
    if vlm_query is None or regions or not crops:
        why = ("VLM unavailable" if vlm_query is None else
               "detect already proposed regions" if regions else "no hint crops")
        rungs.append({"name": "vlm query", "ran": False, "ms": 0.0, "crops": 0,
                      "found": False, "note": why})
    else:
        for c in crops:
            q = vlm_query(c["image"])
            if q and q.get("answer") == "yes":
                votes = q.get("votes") or {}
                n = q.get("passes") or 1
                score = (votes.get("yes", 0) / n) if n else 0.0
                vlm_probable = {"box": list(c["box"]), "score": round(score, 3),
                                "ratio": q.get("ratio"), "rung": "vlm query"}
                break
        rungs.append({"name": "vlm query", "ran": True,
                      "ms": round((time.perf_counter() - t0) * 1000.0, 1),
                      "crops": len(crops), "found": False,
                      "note": "VLM voted yes — unconfirmed" if vlm_probable else ""})

    return _finish(False)
