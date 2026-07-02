#!/usr/bin/env python3
"""Export a YOLO11 model to a fixed-size ONNX for the high-resolution locator scan.

The app runs ONNX via ``cv2.dnn`` only — ``ultralytics`` + ``torch`` are needed
**just for this one-off, offline export** (not at runtime). OpenCV's importer needs
a *static* input shape, so each size is its own file (e.g. ``yolo11m_960.onnx``).

Usage:
    pip install ultralytics            # pulls torch; a throwaway venv is fine
    python scripts/export_yolo.py --model yolo11m --imgsz 960
    python scripts/export_yolo.py --model yolo11m --imgsz 1280

Writes ``d20app/models/<model>_<imgsz>.onnx``. Those variants are already
registered in ``d20app/yolo.py`` (``MODELS``); once the file is present the GUI's
"locator input size" / model picker can use it. Without it, the locator scan falls
back to the native size + tiling (no crash).
"""

from __future__ import annotations

import argparse
import os
import sys

_MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "d20app", "models")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="yolo11m",
                    help="base model name / .pt (default: yolo11m)")
    ap.add_argument("--imgsz", type=int, default=960,
                    help="square input size to export at (e.g. 960, 1280)")
    ap.add_argument("--opset", type=int, default=12)
    ap.add_argument("--half", action="store_true",
                    help="export FP16 weights (benchmark #70: ~0 accuracy cost, up to "
                         "2.2x faster on CUDA; verified with onnxruntime-CUDA — cv2.dnn "
                         "FP16 handling is unverified). Output stem gains _fp16.")
    ap.add_argument("--out", default=None,
                    help="output filename stem (default: <model> for a native-size "
                         "export, else <model>_<imgsz>). e.g. --out yolo26x")
    args = ap.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        print("ultralytics is not installed. Run:  pip install ultralytics\n"
              "(it pulls torch; use a throwaway venv — these are export-only deps.)",
              file=sys.stderr)
        return 2

    weights = args.model if args.model.endswith(".pt") else f"{args.model}.pt"
    print(f"Exporting {weights} at imgsz={args.imgsz} (downloads weights if needed)…")
    model = YOLO(weights)
    # Force the **raw** detection head: YOLO26 is NMS-free end-to-end by default and
    # its export emits a (1,300,6) tensor cv2.dnn mis-decodes. end2end=False gives the
    # familiar (1,84,N) head the app's decoder expects. No-op for heads without it.
    head = model.model.model[-1]
    if hasattr(head, "end2end"):
        head.end2end = False
    # The golden recipe (#70 §2 / #71): raw head forced above (end2end=False), plus
    # explicit nms=False / dynamic=False / batch=1 so nothing re-bakes an NMS head.
    # A good export's output is (1, 84, N) with no NMS/TopK ops; (1, 300, 6) + TopK
    # is a bad one (the app now refuses to decode those).
    out = model.export(format="onnx", imgsz=args.imgsz, opset=args.opset,
                       simplify=True, dynamic=False, batch=1, nms=False,
                       half=bool(args.half))

    base = args.model[:-3] if args.model.endswith(".pt") else args.model
    stem = args.out if args.out else f"{base}_{args.imgsz}"
    if args.half and not stem.endswith("_fp16"):
        stem += "_fp16"
    dest = os.path.join(_MODELS_DIR, f"{stem}.onnx")
    os.makedirs(_MODELS_DIR, exist_ok=True)
    os.replace(out, dest)
    print(f"Wrote {os.path.relpath(dest)}")
    print("It's registered in d20app/yolo.py MODELS — pick it in the GUI's "
          "locator settings / Test-detection model list.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
