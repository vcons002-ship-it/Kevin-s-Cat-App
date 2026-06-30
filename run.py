#!/usr/bin/env python3
"""Entry point: start the web GUI (and with it, the detection loop manager).

Usage::

    python run.py

Then open the printed URL in a browser on the same WiFi to configure the
camera, speaker, sound, and game rules, and to Start/Stop detection.
"""

from __future__ import annotations

import logging
import os
import sys

if sys.version_info < (3, 11):
    sys.exit(
        "Kevin's Cat App requires Python 3.11+ (its dependencies do). "
        f"You're running {sys.version.split()[0]}. "
        "Run it via the venv created by setup.sh: ./venv/bin/python run.py"
    )

from d20app import __version__
from d20app import config as config_mod
from d20app.caster import detect_lan_ip
from d20app.loop import DetectionLoop
from d20app.webapp import create_app


def _maybe_reexec_for_cuda(cfg) -> None:
    """If the NVIDIA CUDA accelerator is selected, make the CUDA runtime libs
    discoverable *before* anything dlopens them — otherwise onnxruntime-gpu silently
    degrades to CPU (~37× slower). glibc caches ``LD_LIBRARY_PATH`` at process start,
    so the only reliable in-process fix is to set it and re-exec once. Guarded by an
    env flag so we re-exec at most once, and a no-op unless ``accelerator`` is
    ``onnx-cuda`` (so the default CPU launch is completely unaffected)."""
    if getattr(cfg, "accelerator", "cpu") != "onnx-cuda":
        return
    if os.environ.get("_D20_CUDA_REEXEC"):       # already re-exec'd once
        return
    from d20app import yolo

    lib = yolo._torch_cuda_lib_dir()
    if not lib:
        return                                    # no torch → runner will error clearly
    cur = os.environ.get("LD_LIBRARY_PATH", "")
    if lib in cur.split(os.pathsep):
        return                                    # already discoverable, nothing to do
    os.environ["LD_LIBRARY_PATH"] = lib + (os.pathsep + cur if cur else "")
    os.environ["_D20_CUDA_REEXEC"] = "1"
    print(f"  (onnx-cuda) added torch's CUDA libs to LD_LIBRARY_PATH; reloading…")
    os.execv(sys.executable, [sys.executable] + sys.argv)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    cfg = config_mod.load()
    _maybe_reexec_for_cuda(cfg)
    loop = DetectionLoop()
    app = create_app(loop)

    lan_ip = detect_lan_ip()
    port = cfg.web_port
    print(f"\n  Kevin's Cat App v{__version__} is running!")
    print(f"  Open the GUI:  http://{lan_ip}:{port}   (or http://localhost:{port})\n")

    # threaded=True so discovery endpoints (which block) don't freeze the page.
    app.run(host="0.0.0.0", port=port, threaded=True)


if __name__ == "__main__":
    main()
