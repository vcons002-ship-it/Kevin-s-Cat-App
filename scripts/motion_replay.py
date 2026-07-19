#!/usr/bin/env python3
"""Replay a clip through the app's REAL motion prefilter, both ways.

Written after a cat crossed a room and the app didn't switch feeds, while an
offline test of the same clip said motion should have fired robustly. Before
trusting that test, the offline path has to match the live one.

It imports :class:`d20app.detector.MotionPrefilter` and
``apply_image_adjustments`` directly — not a copy — and pulls the camera's own
motion settings out of ``config.yaml``, so the only thing left to differ is the
one thing being measured: **how far apart the two diffed frames are.**

The live loop diffs *consecutive reads*, and how much time those span depends on
whether the camera's frames queue or are dropped (see
``scripts/rtsp_latency_probe.py``). So this reports both:

  every-frame  — consecutive decoded frames (~1/camera_fps apart). What the app
                 sees if frames QUEUE.
  sampled      — frames ~1/scan_fps apart. What the app sees if frames are
                 DROPPED, and what a naive offline test measures.

If "sampled" fires and "every-frame" doesn't, the offline test was measuring an
easier problem — and no amount of threshold tuning would have caught the cat.

Usage:
    python scripts/motion_replay.py clip.mp4 --camera Entry
    python scripts/motion_replay.py clip.mp4 --area-frac 0.002 --diff 20
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from d20app import config as config_mod                     # noqa: E402
from d20app.detector import MotionPrefilter, apply_image_adjustments   # noqa: E402


def _camera_spec(name):
    """The watched camera's settings, so the replay uses what the app uses."""
    cfg = config_mod.load()
    targets = config_mod.camera_targets(cfg)
    if not targets:
        return None, float(getattr(cfg, "scan_fps", 5.0))
    if name:
        for spec in targets:
            if spec.get("name") == name:
                return spec, float(spec["scan_fps"])
        names = ", ".join(repr(s.get("name")) for s in targets)
        sys.exit(f"No watched camera named {name!r}. Known: {names}")
    return targets[0], float(targets[0]["scan_fps"])


def _run(frames, prefilter_args, roi, adjust, gap_ms=None):
    """Feed frames through a fresh prefilter; return (fires, total, margins).

    ``gap_ms`` is how far apart the frames are in the video, so the prefilter's
    reference-age lookback sees the same timeline the live app does.
    """
    import cv2

    mp = MotionPrefilter(**prefilter_args)
    fires, margins = 0, []
    for idx, frame in enumerate(frames):
        # Exactly the live order: adjustments, then ROI crop, then grayscale.
        img = apply_image_adjustments(frame, **adjust)
        if roi:
            x, y, w, h = roi
            img = img[y:y + h, x:x + w]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        moved = mp.update(gray, ts_ms=None if gap_ms is None else idx * gap_ms)
        fires += bool(moved)
        h, w = gray.shape[:2]
        min_area = mp.min_area_frac * h * w
        biggest = 0.0
        for (x1, y1, x2, y2) in (mp.last_blobs or []):
            biggest = max(biggest, float((x2 - x1) * (y2 - y1)))
        margins.append((biggest, min_area))
    return fires, len(frames), margins


def _load(path, step):
    """Every `step`-th frame of the clip, decoded once."""
    import cv2

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        sys.exit(f"could not open {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    out, i = [], 0
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        if i % step == 0:
            out.append(frame)
        i += 1
    cap.release()
    return out, fps, i


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("clip")
    ap.add_argument("--camera", help="use this watched camera's settings (recommended)")
    ap.add_argument("--scan-fps", type=float, help="override the sampling rate")
    ap.add_argument("--area-frac", type=float, help="override motion_min_area_frac")
    ap.add_argument("--diff", type=int, help="override motion_diff_threshold")
    ap.add_argument("--blob-px", type=int, help="override motion_min_blob_px")
    ap.add_argument("--reference-ms", type=int, default=None,
                    help="compare against a frame this old IN THE VIDEO "
                         "(default: the camera's motion_reference_ms)")
    args = ap.parse_args()

    spec, scan_fps = _camera_spec(args.camera)
    if args.scan_fps:
        scan_fps = args.scan_fps

    prefilter_args = {
        "min_area_frac": args.area_frac if args.area_frac is not None
        else float(spec["motion_min_area_frac"]) if spec else 0.003,
        "diff_threshold": args.diff if args.diff is not None
        else int(spec["motion_diff_threshold"]) if spec else 25,
        "min_blob_px": args.blob_px if args.blob_px is not None
        else int(spec["motion_min_blob_px"]) if spec else 14,
    }
    cfg_all = config_mod.load()
    reference_ms = (args.reference_ms if args.reference_ms is not None
                    else int(getattr(cfg_all, "motion_reference_ms", 0)))
    prefilter_args["reference_ms"] = reference_ms
    roi = (spec or {}).get("roi") or None
    adjust = {k: (spec or {}).get(k, d) for k, d in
              (("gamma", 1.0), ("brightness", 0), ("contrast", 1.0), ("saturation", 1.0))}

    src_frames, clip_fps, total = _load(args.clip, 1)
    if not src_frames:
        sys.exit("no frames decoded")
    h, w = src_frames[0].shape[:2]

    print(f"clip     : {args.clip}")
    print(f"           {total} frames, {w}x{h}, {clip_fps:.1f} fps")
    print(f"settings : {'camera ' + repr(spec['name']) if spec else 'defaults'}  "
          f"scan_fps={scan_fps:g}")
    print(f"           min_area_frac={prefilter_args['min_area_frac']} "
          f"diff={prefilter_args['diff_threshold']} "
          f"min_blob_px={prefilter_args['min_blob_px']}")
    print(f"           roi={roi or 'none'}  adjustments={adjust}")
    print(f"           motion_reference_ms={reference_ms}"
          + ("  (0 = compare with the previous frame, whatever age that is)"
             if not reference_ms else
             "  (comparison spans this much video regardless of frame spacing)"))
    if roi:
        print("           NOTE: an ROI is set — it crops before motion, which also "
              "shrinks the\n                 area threshold (min_area = frac x H x W).")

    step = max(1, int(round((clip_fps or scan_fps) / max(0.1, scan_fps))))
    sampled = src_frames[::step]
    gap_every = 1.0 / clip_fps if clip_fps else 0.0
    gap_sampled = step / clip_fps if clip_fps else 0.0

    print(f"\n{'mode':<14}{'gap':>9}{'frames':>9}{'motion':>9}   biggest blob vs threshold")
    print("-" * 74)
    for name, frames, gap in (("every-frame", src_frames, gap_every),
                              ("sampled", sampled, gap_sampled)):
        fires, n, margins = _run(frames, prefilter_args, roi, adjust,
                                 gap_ms=gap * 1000.0 if gap else None)
        peak = max((b for b, _ in margins), default=0.0)
        thr = margins[0][1] if margins else 0.0
        ratio = (peak / thr) if thr else 0.0
        print(f"{name:<14}{gap * 1000:>7.0f}ms{n:>9}{fires:>6}/{n}   "
              f"{peak:>9.0f} vs {thr:<9.0f} ({ratio:.2f}x)")

    print("\nRead it like this:")
    print("  both fire            -> spacing isn't the problem; look elsewhere.")
    print("  only 'sampled' fires -> a naive offline test over-reports. Whether the")
    print("                          app sees this depends on the queue-vs-drop")
    print("                          answer from scripts/rtsp_latency_probe.py.")
    print("  neither fires        -> the clip genuinely shouldn't have triggered with")
    print("                          these settings; the harness was misconfigured.")
    print("  (ratio < 1.00x means the biggest changed region never reached the area")
    print("   threshold — that's the knob that decides it.)")


if __name__ == "__main__":
    main()
