#!/usr/bin/env python3
"""Why did "find my cat" miss a cat the Test tool finds in the same image?

Live case: find reported no cat; the frame IT saved as scanned, uploaded to the
Test tool at the same model/tiling/overlap, found a cat at 0.47. Same image, same
nominal settings, different answer — so the difference is in the settings dict
find builds, not in the frame.

This replays that exact image through the SAME function the web app uses
(``webapp._run_test_detection``) with the settings find builds, then flips one
knob at a time until the cat appears. Whichever flip changes the answer is the
bug.

Run it on the machine with the models, against a screenshots/find/*_scanned.jpg:

    python scripts/find_replay.py screenshots/find/2026-07-25_02-58-14_Office_scanned.jpg --camera Office
"""

from __future__ import annotations

import argparse
import copy
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from d20app import config as config_mod                    # noqa: E402
from d20app.detector import PersonDetector                 # noqa: E402
from d20app.webapp import _run_test_detection              # noqa: E402


def _find_settings(cfg, spec):
    """EXACTLY what d20app/webapp.py builds for a find. Keep this in step with it."""
    det = PersonDetector(source="__replay__", model=spec["model"],
                         accelerator=spec["accelerator"],
                         confidence=spec["person_confidence"],
                         cat_confidence=spec["cat_confidence"],
                         label_floor=spec["label_floor"],
                         locator_classes=spec["locator_classes"])
    return {
        "model": cfg.find_model or det.model,
        "accelerator": det.accelerator,
        "person_confidence": det.confidence,
        "cat_confidence": cfg.find_confidence or det.cat_confidence,
        "label_floor": det.label_floor,
        "locator_classes": list(det.locator_classes),
        "tiling": cfg.find_tiling or "3x3",
        "tile_overlap": cfg.find_tile_overlap,
    }


def _cats(dets, classes):
    return [d for d in dets if d["label"] in classes]


def _run(frame, settings, label):
    _annotated, dets, ms = _run_test_detection(frame, settings)
    cats = _cats(dets, settings["locator_classes"])
    best = max((d["score"] for d in cats), default=0.0)
    top = ", ".join(f"{d['label']} {d['score']:.2f}" for d in dets[:4]) or "nothing"
    print(f"  {label:<34} cat={best:.2f}  ({ms:>6.0f} ms)  saw: {top}")
    return best


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image", help="a screenshots/find/*_scanned.jpg")
    ap.add_argument("--camera", required=True, help="which camera's settings find used")
    args = ap.parse_args()

    import cv2

    frame = cv2.imread(args.image)
    if frame is None:
        sys.exit(f"could not read {args.image}")

    cfg = config_mod.load()
    spec = next((s for s in config_mod.camera_targets(cfg)
                 if s.get("name") == args.camera), None)
    if spec is None:
        names = ", ".join(repr(s.get("name")) for s in config_mod.camera_targets(cfg))
        sys.exit(f"No watched camera named {args.camera!r}. Known: {names}")

    base = _find_settings(cfg, spec)
    h, w = frame.shape[:2]
    print(f"image    : {args.image}  ({w}x{h})")
    print(f"camera   : {args.camera}")
    print("find ran with:")
    for k, v in base.items():
        print(f"  {k:<20} {v!r}")

    print("\nbaseline — exactly what find does:")
    base_score = _run(frame, base, "find settings")

    # One knob at a time. Whichever flips the answer is the difference between
    # find and the Test tool.
    variants = [
        ("label_floor -> 0.01", {"label_floor": 0.01}),
        ("person_confidence -> 0.01", {"person_confidence": 0.01}),
        ("cat_confidence -> 0.01", {"cat_confidence": 0.01}),
        ("locator + dog", {"locator_classes": ["cat", "dog"]}),
        ("tiling -> off", {"tiling": "off"}),
        ("tiling -> 2x2", {"tiling": "2x2"}),
        ("tiling -> 4x4", {"tiling": "4x4"}),
        ("overlap -> 0.35", {"tile_overlap": 0.35}),
        ("imgsz -> 960", {"imgsz": 960}),
        # find omits these entirely, so the test detector uses no-op defaults —
        # if the camera actually has adjustments set, that IS a real difference.
        ("camera image adjustments", {
            "gamma": spec["gamma"], "brightness": spec["brightness"],
            "contrast": spec["contrast"], "saturation": spec["saturation"]}),
        ("camera model (not find's)", {"model": spec["model"]}),
    ]
    print("\none knob at a time:")
    flips = []
    for label, patch in variants:
        s = copy.deepcopy(base)
        s.update(patch)
        try:
            score = _run(frame, s, label)
        except Exception as exc:                     # noqa: BLE001 — keep going
            print(f"  {label:<34} FAILED: {exc}")
            continue
        if score > base_score + 0.05:
            flips.append((label, score))

    print("\n" + "=" * 70)
    if not flips:
        print("No single knob explains it. The difference is not in these settings —\n"
              "next suspect is the frame itself (what find passes vs what the tester\n"
              "decoded from the same file).")
    else:
        print("These changed the answer — the top one is the difference:")
        for label, score in sorted(flips, key=lambda f: -f[1]):
            print(f"  {label:<34} cat={score:.2f}  (find got {base_score:.2f})")


if __name__ == "__main__":
    main()
