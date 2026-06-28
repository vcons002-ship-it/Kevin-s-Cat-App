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
    out = YOLO(weights).export(format="onnx", imgsz=args.imgsz,
                               opset=args.opset, simplify=True)

    base = args.model[:-3] if args.model.endswith(".pt") else args.model
    dest = os.path.join(_MODELS_DIR, f"{base}_{args.imgsz}.onnx")
    os.makedirs(_MODELS_DIR, exist_ok=True)
    os.replace(out, dest)
    print(f"Wrote {os.path.relpath(dest)}")
    print("It's registered in d20app/yolo.py MODELS — pick it in the GUI's "
          "locator settings / Test-detection model list.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
