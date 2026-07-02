"""Sighting heat maps: where the cats actually are, accumulated over history (#68).

Every sighting in ``cats.log`` carries a box in the camera's (ROI-cropped) frame
coordinates — 500 of them per the tracker's cap. Stacking those boxes into a
density field and colour-mapping it shows the camera's *hot spots* (the basket,
the couch arm, the sunny patch) at a glance, and is the visual counterpart of the
time-of-day prior (`CatTracker.by_hour` / `likely_cameras`).

Pure rendering: numpy/cv2 in, BGR image out; no state, no locks.
"""

from __future__ import annotations


def render_heatmap(frame_bgr, boxes, alpha: float = 0.45):
    """The sighting-density heat map composited over ``frame_bgr``.

    ``boxes`` is ``[(x1, y1, x2, y2)]`` in the same frame coords (clipped here, so
    boxes recorded under an older ROI that overhang the current frame degrade to
    their visible part instead of erroring). Returns a new BGR image, or None when
    there are no usable boxes.
    """
    import cv2
    import numpy as np

    h, w = frame_bgr.shape[:2]
    acc = np.zeros((h, w), np.float32)
    used = 0
    for box in boxes or []:
        if not box or len(box) != 4:
            continue
        x1, y1, x2, y2 = (int(v) for v in box)
        x1, x2 = max(0, min(x1, w)), max(0, min(x2, w))
        y1, y2 = max(0, min(y1, h)), max(0, min(y2, h))
        if x2 - x1 < 2 or y2 - y1 < 2:
            continue
        acc[y1:y2, x1:x2] += 1.0
        used += 1
    if not used:
        return None

    # Soften box edges into a density field, then normalise for the colour map.
    k = max(9, (min(h, w) // 40) | 1)              # odd kernel scaled to the frame
    acc = cv2.GaussianBlur(acc, (k, k), 0)
    acc = (255 * acc / acc.max()).astype(np.uint8)
    colour = cv2.applyColorMap(acc, cv2.COLORMAP_JET)
    mask = acc > 16                                 # leave cold areas untinted
    out = frame_bgr.copy()
    out[mask] = ((1 - alpha) * out[mask] + alpha * colour[mask]).astype(np.uint8)
    return out
