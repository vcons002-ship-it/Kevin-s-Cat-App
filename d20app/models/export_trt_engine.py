#!/usr/bin/env python3
"""Build TensorRT engines for the app's models — a cached, per-GPU setup step (#82).

Engines are **GPU-specific**: build them on the machine that will run them (an
engine built on the 5090 won't load on the 3070 and vice-versa). A build takes a
few minutes per model; the result is cached next to the ONNX files and reused —
this script is a one-time setup step, never something the app runs per launch.

Hard prerequisite: a **CUDA-13-capable NVIDIA driver**. TensorRT's pip build
pulls CUDA-13 dependencies; on an older driver (Debian 12's repo driver 535.x =
CUDA 12.2) installing tensorrt breaks the torch stack (#82). This script checks
the driver and refuses rather than letting that happen. On a good driver:

    pip install ultralytics tensorrt cuda-python     # build-time deps
    python d20app/models/export_trt_engine.py yolo26x_fp16 yolo26m_fp16

Measured payoff on the NAS 3070 (#82, real tiled pipeline, identical accuracy):
26x 1.6× (175 ms → 111 ms), 26m 1.37×, 11n 1.21×.
"""
import argparse
import os
import shutil
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))     # repo root
from d20app import yolo                                          # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build cached, GPU-specific TensorRT engines (#82).")
    ap.add_argument("variants", nargs="+",
                    help=f"model variant(s), e.g. yolo26x_fp16 (known: "
                         f"{', '.join(sorted(yolo.MODELS))})")
    ap.add_argument("--force", action="store_true",
                    help="rebuild even if a cached engine exists")
    ap.add_argument("--fp32", action="store_true",
                    help="build FP32 (default is FP16 — identical accuracy, "
                         "materially faster on the 3070; #70 §4)")
    args = ap.parse_args()

    ver = yolo._driver_cuda_version()
    if ver is None or ver < yolo._TRT_MIN_CUDA:
        sys.exit(f"This driver reports CUDA {ver} — TensorRT needs "
                 f">= {yolo._TRT_MIN_CUDA:g}. Upgrade the NVIDIA driver first "
                 "(NVIDIA's CUDA repo; Debian 12's own repo caps at 535/12.2). "
                 "Do NOT pip-install tensorrt on this driver — it breaks torch (#82).")
    try:
        from ultralytics import YOLO
    except ImportError:
        sys.exit("Needs 'ultralytics' (build-time only): "
                 "pip install ultralytics tensorrt cuda-python")

    for variant in args.variants:
        if variant not in yolo.MODELS:
            sys.exit(f"unknown variant {variant!r}; known: {sorted(yolo.MODELS)}")
        out = yolo.engine_path(variant)
        if os.path.exists(out) and not args.force:
            print(f"{os.path.basename(out)}: already built (cached; --force to redo)")
            continue
        base = variant.replace("_fp16", "")
        model = YOLO(f"{base}.pt")                   # ultralytics fetches the .pt
        model.model.model[-1].end2end = False        # the golden head (#70 §2)
        built = model.export(format="engine", imgsz=yolo.input_size(variant),
                             half=not args.fp32, simplify=True, dynamic=False,
                             batch=1)
        shutil.move(str(built), out)
        print(f"built {os.path.basename(out)} — GPU-specific; rebuild after any "
              "GPU change")


if __name__ == "__main__":
    main()
