#!/usr/bin/env python3
"""Diagnose YOLO GPU / Intel-iGPU acceleration end-to-end.

Run it from the project folder:

    ./venv/bin/python check_accelerator.py

It reports which compute devices this machine actually exposes (OpenCL via cv2,
and Intel's OpenVINO), what your configured ``accelerator`` *resolves* to (a real
GPU vs a silent CPU fallback), and a quick ms/frame timing for CPU vs your chosen
backend — so you can confirm the offload is real instead of secretly running on
the CPU.

This is the answer to "I picked OpenVINO GPU — is it actually using the iGPU?".
Nothing here changes your config; it only measures.
"""

from __future__ import annotations

import sys
import time

import numpy as np

from d20app import config as config_mod
from d20app import yolo


def _bench(runner, size: int, n: int = 20) -> float:
    """Average ms/frame for ``n`` inferences on a blank blob (timing only)."""
    blob = np.zeros((1, 3, size, size), dtype=np.float32)
    runner.infer(blob)                       # warm-up (model load / first-run JIT)
    t0 = time.time()
    for _ in range(n):
        runner.infer(blob)
    return (time.time() - t0) / n * 1000.0


def _report_opencl() -> None:
    import cv2

    print("OpenCL  (the 'opencl' accelerator):")
    if not cv2.ocl.haveOpenCL():
        print("  ⚠ not available here — 'opencl' would silently run on the CPU.")
        return
    cv2.ocl.setUseOpenCL(True)
    name = None
    for getter in ("Device_getDefault", "Device"):
        try:                                  # the exact API name varies by cv2 build
            obj = getattr(cv2.ocl, getter)
            dev = obj.getDefault() if getter == "Device" else obj()
            name = f"{dev.name()} (vendor: {dev.vendorName()})"
            break
        except Exception:                     # noqa: BLE001 — best-effort device name
            continue
    print(f"  ✅ available — device: {name or 'unknown'}")


def _report_openvino() -> bool:
    """Print OpenVINO status; return True if an Intel GPU device is present."""
    print("\nOpenVINO  (the 'openvino-gpu' / 'openvino-auto' accelerators):")
    try:
        import openvino as ov
    except ImportError:
        print("  ⚠ 'openvino' not installed — re-run setup, or: pip install openvino")
        return False
    try:
        core = ov.Core()
        devices = core.available_devices
    except Exception as exc:                  # noqa: BLE001
        print(f"  ⚠ openvino present but Core() failed: {exc}")
        return False

    print(f"  ✅ openvino {ov.__version__} installed — devices: {devices}")
    has_gpu = any(d == "GPU" or d.startswith("GPU.") for d in devices)
    if has_gpu:
        try:
            print(f"  ✅ Intel GPU present: {core.get_property('GPU', 'FULL_DEVICE_NAME')}")
        except Exception:                     # noqa: BLE001
            print("  ✅ Intel GPU device present.")
    else:
        print("  ⚠ no GPU device exposed (only CPU). 'openvino-gpu' will fall back to")
        print("    CPU; 'openvino-auto' will pick CPU. Install the Intel GPU compute")
        print("    drivers (e.g. intel-opencl-icd / the NEO runtime) to enable the iGPU.")
    return has_gpu


def main() -> int:
    cfg = config_mod.load()
    print(f"Configured: detector_model={cfg.detector_model}  "
          f"accelerator={cfg.accelerator}\n")

    import cv2
    have_opencl = cv2.ocl.haveOpenCL()
    _report_opencl()
    has_gpu = _report_openvino()

    # The accelerator only affects the YOLO models; if MobileNet is selected we
    # still demo with yolo11n so the device test is meaningful.
    model = cfg.detector_model if cfg.detector_model.startswith("yolo") else "yolo11n"
    if model != cfg.detector_model:
        print(f"\n(Your detector_model is {cfg.detector_model}; the accelerator only "
              f"affects YOLO, so testing with {model}.)")
    size = yolo.input_size(model)

    print(f"\nResolving accelerator={cfg.accelerator!r} for {model} (input {size})…")
    used, runner, ms = cfg.accelerator, None, None
    try:
        runner = yolo.load_net(model, cfg.accelerator)
        ms = _bench(runner, size)
        print(f"  ✅ loaded on '{cfg.accelerator}': {ms:.0f} ms/frame")
    except Exception as exc:                  # noqa: BLE001
        print(f"  ⚠ '{cfg.accelerator}' could not start ({exc})")
        print("    The app would fall back to CPU automatically.")
        used = "cpu"
        runner = yolo.load_net(model, "cpu")
        ms = _bench(runner, size)
        print(f"  → CPU: {ms:.0f} ms/frame")

    if used == "cpu":
        print("\n  Running on CPU. Pick an 'openvino-*' or 'opencl' accelerator in the")
        print("  GUI (Detection card) to offload onto an Intel iGPU.")
        return 0

    # Is the chosen backend *actually* on a GPU, or just a faster CPU runtime?
    # openvino-gpu only loads if a GPU compiled; AUTO/opencl depend on a device.
    if used == "openvino-gpu":
        on_gpu = True                         # GPU compile would have raised otherwise
    elif used == "openvino-auto":
        on_gpu = has_gpu
    elif used == "opencl":
        on_gpu = have_opencl
    else:
        on_gpu = False

    cpu_ms = _bench(yolo.load_net(model, "cpu"), size)
    speed = cpu_ms / ms if ms else 0.0
    verdict = f"{speed:.1f}× faster" if ms < cpu_ms else "no faster"
    print(f"\n  cv2.dnn CPU: {cpu_ms:.0f} ms/frame  →  '{used}': {ms:.0f} ms/frame "
          f"({verdict})")

    if on_gpu:
        print("  ✅ Running on the GPU — real hardware offload, and the CPU is freed up.")
    else:
        # No GPU engaged. OpenVINO's CPU runtime can still beat cv2.dnn's — say so
        # honestly rather than claiming an offload that didn't happen.
        print("  ⚠ No GPU device was engaged — this is running on the CPU.")
        if used.startswith("openvino") and ms < cpu_ms * 0.9:
            print(f"    (OpenVINO's CPU runtime is still ~{speed:.1f}× faster than cv2.dnn "
                  "here — a free win even without an iGPU.)")
        print("    To use the iGPU, install the Intel GPU compute drivers and re-run.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
