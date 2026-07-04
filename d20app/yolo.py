"""YOLO11 object detection via OpenCV's ``cv2.dnn`` (ONNX) — no PyTorch at runtime.

The detection backend for :class:`d20app.detector.PersonDetector`: it produces
``[(label, score, (x1, y1, x2, y2))]`` boxes that the rest of the pipeline
(person trigger, ``cat`` labelling, annotated snapshots) consumes.

Why YOLO: the previously-bundled MobileNet-SSD was fast but weak in low light /
odd poses (it scored 0.00 on a real dim night frame). YOLO11n scored ~0.87 on the
same frame — a much better night/occlusion detector — so SSD was dropped entirely
in 0.25.0 (#57) and YOLO is now the only backend.

The selectable lineup is benchmark-settled (#70/#71; see :data:`MODELS`):

- ``yolo26x`` — the workhorse: 91% recall / 0% FP at 3×3/0.20 tiling. Export-only
  (~213 MB); FP16 makes it the everyday pick on CUDA (167 ms warm on a 3070).
- ``yolo26m`` — the lightweight: 82%/0% at 2×2/0.20, 64 ms FP16.
- ``yolo11n`` — the floor: 75%/0% at 3×3, small and CPU-friendly. The default.
  Bundled at **640** (#80: the earlier 320 export was never benchmarked and ran
  far below the 640 numbers it was labelled with).
- Dropped models (``yolo11m`` and friends) now raise a loud, actionable error
  (#79) — the no-silent-worse-detector stance of 0.25.0, applied consistently.

Each model is exported from its Ultralytics ``*.pt`` to a fixed-size ONNX (see
d20app/models/README.md). Raw output is ``(1, 84, N)``: 4 box coords (cx, cy, w,
h in letterboxed pixels) + 80 COCO class scores per anchor.

Acceleration (``accelerator`` argument to :func:`load_net`):

- ``cpu`` (default) — OpenCV ``cv2.dnn`` on the CPU. Always available.
- ``opencl`` — same ``cv2.dnn`` net but with the ``OPENCL_FP16`` target, so the
  conv layers run on an OpenCL device (e.g. an Intel iGPU). No extra Python deps;
  needs the host's OpenCL runtime. OpenCV silently falls back to CPU if no OpenCL
  device is present, so it's safe but not guaranteed to actually offload.
- ``openvino-gpu`` / ``openvino-auto`` — run the ONNX through Intel's **OpenVINO**
  runtime (optional ``openvino`` package) on the ``GPU`` device, or ``AUTO`` which
  picks GPU and falls back to CPU itself. The dependable iGPU path — typically
  2–4× CPU on Intel hardware — useful headroom for the heavier models.
  **Intel-only**; needs the host Intel GPU compute drivers.

Whatever the backend, inference returns the same ``(1, 84, N)`` tensor, so the
letterbox + NMS decode below is shared. ``load_net`` returns a small *runner*
(``.infer(blob) -> ndarray``) wrapping whichever engine was selected.
"""

from __future__ import annotations

import logging
import os

_log = logging.getLogger(__name__)

# COCO-80 class names in model order. Index 0 is "person", 15 is "cat" — the two
# the app cares about; the rest are named in the activity log like any mover.
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush",
]

_MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

# Known YOLO variants → their exported ONNX file and the fixed input size that
# file was exported at (must match, or the net errors / decodes garbage).
# The benchmark-settled lineup (#70/#71, full 199-cat + 43-null set, golden exports):
# 26x = workhorse (91% recall / 0% FP @ 3×3/0.20), 26m = lightweight (82%/0% @
# 2×2/0.20), 11n = floor (75%/0%). 11m and 26n were dropped — with the golden export
# 26m beats 11m and 11n beats 26n (11x was tested and rejected: FP-prone). A config
# naming a dropped model gets a loud error (DROPPED_MODELS below, #79) — never a
# silent worse detector. FP16 exports cost ~0
# accuracy and run up to 2.2× faster on CUDA (#70 §4) — benchmark-verified on
# onnxruntime-CUDA only; cv2.dnn's handling of FP16 weights is NOT verified, so use
# them with the auto / onnx-cuda accelerators. Every model must be GOLDEN-exported:
# raw (1, 84, N) head, no NMS/TopK ops (end2end=False — scripts/export_yolo.py);
# an end2end (1, 300, 6) export mis-decodes and detect_boxes() now rejects it loudly.
# Files not committed (26x, fp16 variants) appear in the picker once exported.
MODELS = {
    # The floor ships at 640 (#80): the earlier bundled 320 export was never
    # benchmarked and performed far below the 640 numbers it was labelled with —
    # the 75%/0% figures belong to the 640 golden 11n, which is what's bundled now.
    "yolo11n": {"file": "yolo11n.onnx", "size": 640,
                "label": "YOLO11n (640) — floor, low CPU", "selectable": True},
    # FP16 for lineup consistency (#86): every model gets a CUDA-precision onnx.
    # 11n barely speeds up in FP16 (not compute-bound, #70 §4) — this exists so
    # the CUDA path never has to fall back to FP32 just for the floor model.
    # (#90: precision is NOT a user-facing choice — the _fp16 entries are
    # registered for old configs + provisioning, hidden from pickers, and
    # resolve_variant() picks the right file from the accelerator.)
    "yolo11n_fp16": {"file": "yolo11n_fp16.onnx", "size": 640,
                     "label": "YOLO11n FP16 (640) — floor, CUDA",
                     "selectable": False},
    "yolo26m": {"file": "yolo26m.onnx", "size": 640,
                "label": "YOLO26m (640) — lightweight", "selectable": True},
    "yolo26m_fp16": {"file": "yolo26m_fp16.onnx", "size": 640,
                     "label": "YOLO26m FP16 (640) — lightweight, CUDA",
                     "selectable": False},
    "yolo26x": {"file": "yolo26x.onnx", "size": 640,
                "label": "YOLO26x (640) — workhorse", "selectable": True},
    "yolo26x_fp16": {"file": "yolo26x_fp16.onnx", "size": 640,
                     "label": "YOLO26x FP16 (640) — workhorse, CUDA",
                     "selectable": False},
}
DEFAULT_VARIANT = "yolo11n"

# Models DROPPED after the golden benchmark (#70/#71, revisited in #79): naming one
# raises a loud, actionable error instead of quietly running a worse detector —
# the same no-silent-fallback stance as the MobileNet-SSD removal (0.25.0). The
# 960/1280 "locator" exports also embodied a rejected hypothesis: >640 input
# measured WORSE than 640 on the full set; tiling is the resolution lever.
DROPPED_MODELS = {
    "yolo11m": "dropped in #71 — golden-exported yolo26m beats it",
    "yolo11m_960": "dropped in #79 — >640 input measured worse than 640 (#70); use 3x3 tiling",
    "yolo11m_1280": "dropped in #79 — >640 input measured worse than 640 (#70); use 3x3 tiling",
    "yolo11x": "rejected in #70 — false-positive-prone",
    "yolo26n": "dropped in #71 — golden-exported yolo11n beats it",
    "mobilenet_ssd": "removed in 0.25.0 (#57) — lost every benchmark to YOLO",
}

# Where each accelerator runs the conv layers.
#   cpu / opencl                 — OpenCV cv2.dnn (CPU, or an OpenCL iGPU target)
#   openvino-gpu / openvino-auto — Intel OpenVINO runtime (iGPU)
#   onnx-cuda                    — onnxruntime-gpu on an NVIDIA GPU (CUDAExecutionProvider)
#   tensorrt                     — a prebuilt, GPU-specific .engine (#82): ~1.6× the
#                                  onnx-cuda workhorse on the 3070, identical accuracy.
#                                  Opt-in; needs a CUDA-13-capable driver (see below).
#   auto                         — onnx-cuda when it genuinely binds, else CPU (loudly)
ACCELERATORS = ("auto", "cpu", "opencl", "openvino-gpu", "openvino-auto", "onnx-cuda",
                "tensorrt")

# TensorRT's pip build pulls CUDA-13 dependencies. On an older driver (e.g. 535.x =
# CUDA 12.2) those packages can't run AND installing them breaks the torch stack
# (#82 learned this the hard way: `undefined symbol: ncclCommResume`). So the
# accelerator refuses to even try below this driver capability — and nothing in the
# app ever pip-installs tensorrt on the user's behalf.
_TRT_MIN_CUDA = 13.0

# Back-compat aliases for the single-model era (some tests/callers import these).
ONNX_PATH = os.path.join(_MODELS_DIR, MODELS[DEFAULT_VARIANT]["file"])
INPUT_SIZE = MODELS[DEFAULT_VARIANT]["size"]
_NMS_IOU = 0.45


def model_path(variant: str = DEFAULT_VARIANT) -> str:
    """Absolute path to a variant's ONNX file (no existence check)."""
    return os.path.join(_MODELS_DIR, MODELS[variant]["file"])


def resolve_variant(model: str, accelerator: str = "cpu",
                    exists=os.path.exists) -> str:
    """The concrete file variant to load for a (logical model, accelerator) pair.

    Precision is an implementation detail of the runtime, not a user choice
    (#90): cv2.dnn (cpu/opencl) needs FP32; onnxruntime (auto/onnx-cuda) wants
    the FP16 onnx; tensorrt's engine is FP16 regardless, and its fallback runs
    onnxruntime, so it prefers FP16 too; OpenVINO converts precision itself but
    is faster fed FP16. Legacy configs naming a ``*_fp16`` model normalize to
    the base — they can never force FP16 onto cv2.dnn."""
    base = (model or "").replace("_fp16", "")
    if base not in MODELS:
        return model                        # unknown: let load_net raise clearly
    accel = (accelerator or "cpu").lower()
    if accel in ("auto", "onnx-cuda", "tensorrt", "openvino-gpu", "openvino-auto"):
        fp16 = f"{base}_fp16"
        if fp16 in MODELS and exists(model_path(fp16)):
            return fp16
    return base


def _annotate(runner, effective: str, reason: str = ""):
    """Stamp what actually ran onto the runner (#90) — callers surface it, so a
    fallback is never silently labelled as the requested accelerator."""
    runner.effective_accelerator = effective
    runner.fallback_reason = reason
    return runner


# Targeted-boost model preference (0.42.0), strongest first per #70: 26x (91%/0%)
# > 26m (82%/0%) > the 11n floor. FP16 variants only pay on CUDA (2.2× there,
# nothing elsewhere); "auto" may resolve to CPU, so it gets the FP32 list — always
# correct, merely slower if CUDA happens to be active.
_BOOST_ORDER_CUDA = ("yolo26x_fp16", "yolo26x", "yolo26m_fp16", "yolo26m")
_BOOST_ORDER = ("yolo26x", "yolo26m")


def boost_variant(current: str, accelerator: str = "cpu",
                  exists=os.path.exists) -> str | None:
    """The strongest *available* model to run a targeted boost with, or ``None``
    when the camera's own ``current`` model is already the best on disk.

    Walks the #70 strength order and stops at ``current`` — a boost must never
    *downgrade* (a 26x camera doesn't "boost" with 26m). ``exists`` is injectable
    for tests.
    """
    order = _BOOST_ORDER_CUDA if accelerator == "onnx-cuda" else _BOOST_ORDER
    for variant in order:
        if variant == current:
            return None                      # everything after this is weaker
        if exists(model_path(variant)):
            return variant
    return None


def input_size(variant: str = DEFAULT_VARIANT) -> int:
    """The fixed square input size the given variant was exported at."""
    return MODELS[variant]["size"]


def engine_path(variant: str = DEFAULT_VARIANT) -> str:
    """Absolute path to a variant's TensorRT engine (no existence check).

    Engines are **GPU-specific** — built once per machine by
    ``models/export_trt_engine.py`` and cached (#82); they are never committed.
    One engine per base model (#86): our engines are always built FP16 (the
    settled precision), so ``yolo26x`` and ``yolo26x_fp16`` share
    ``yolo26x.engine``."""
    return os.path.join(_MODELS_DIR, f"{variant.replace('_fp16', '')}.engine")


class _CvDnnRunner:
    """Wraps a ``cv2.dnn`` net (CPU or an OpenCL target) behind ``infer``."""

    def __init__(self, net, target=None):
        import cv2

        if target is not None:
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            net.setPreferableTarget(target)
        self._net = net

    def infer(self, blob):
        self._net.setInput(blob)
        return self._net.forward()          # (1, 84, N)


class _OpenVinoRunner:
    """Wraps an OpenVINO compiled model (``GPU`` / ``AUTO`` device) behind ``infer``."""

    def __init__(self, onnx_path: str, device: str):
        import openvino as ov               # optional dep; import only when asked

        core = ov.Core()
        self._compiled = core.compile_model(core.read_model(onnx_path), device)
        self._out = self._compiled.output(0)

    def infer(self, blob):
        return self._compiled([blob])[self._out]     # (1, 84, N)


def _torch_cuda_lib_dir():
    """torch's bundled ``lib/`` dir, or ``None``. torch ships a complete, compatible
    CUDA 12 / cuDNN 9 runtime there; pointing the loader at it is the clean single fix
    for onnxruntime-gpu's CUDA provider (which otherwise can't find libcublasLt/cudnn
    and *silently* degrades to CPU — 37× slower). torch is already a moondream dep."""
    try:
        import torch
    except Exception:       # noqa: BLE001 — torch is optional
        return None
    lib = os.path.join(os.path.dirname(torch.__file__), "lib")
    return lib if os.path.isdir(lib) else None


def _ensure_cuda_libs_on_path():
    """Best-effort: prepend torch's CUDA lib dir to ``LD_LIBRARY_PATH`` so onnxruntime's
    CUDA provider can load its runtime deps. Note: glibc may have cached the path at
    process start, so the launch scripts also export it — this is belt-and-braces, and
    importing torch (below, in the runner) pre-loads the same libs into the process."""
    lib = _torch_cuda_lib_dir()
    if not lib:
        return
    cur = os.environ.get("LD_LIBRARY_PATH", "")
    if lib not in cur.split(os.pathsep):
        os.environ["LD_LIBRARY_PATH"] = lib + (os.pathsep + cur if cur else "")


class _OnnxRuntimeRunner:
    """Wraps an onnxruntime ``InferenceSession`` on the NVIDIA CUDA provider.

    onnxruntime-gpu's CUDA provider **silently falls back to CPU** if its CUDA runtime
    libs aren't discoverable — which looks like it works but runs ~37× slower. So we:
      1. put torch's bundled CUDA libs on the loader path and import torch to pre-load
         them into the process, then
      2. after the session is built, check which provider it actually selected and
         raise loudly if CUDA was requested but CPU was chosen — never run slow silently.
    The decode in :func:`detect_boxes` is unchanged: ``infer`` returns the same
    ``(1, 84, N)`` tensor the cv2.dnn / OpenVINO runners do.
    """

    def __init__(self, onnx_path: str):
        _ensure_cuda_libs_on_path()
        try:
            import torch  # noqa: F401 — pre-load torch's CUDA libs into the process
        except Exception:           # noqa: BLE001 — not fatal; libs may be elsewhere
            pass
        import onnxruntime as ort    # optional dep; import only when asked

        avail = ort.get_available_providers()
        if "CUDAExecutionProvider" not in avail:
            raise RuntimeError(
                "onnx-cuda needs onnxruntime-gpu with a working CUDAExecutionProvider, "
                f"but only {avail} are available. Install the CUDA-12 build of "
                "onnxruntime-gpu (the default pip build targets CUDA 13 and fails with "
                "'libcudart.so.13 not found' on a CUDA-12 system)."
            )
        self._sess = ort.InferenceSession(
            onnx_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        active = self._sess.get_providers()[0]
        if active != "CUDAExecutionProvider":
            raise RuntimeError(
                f"onnx-cuda requested but onnxruntime is running on {active}, not CUDA "
                "— that's the silent ~37× slowdown trap. The CUDA runtime libs likely "
                "aren't on LD_LIBRARY_PATH; export torch's lib dir before launch:\n"
                "  export LD_LIBRARY_PATH=$(python -c \"import os,torch;"
                "print(os.path.dirname(torch.__file__)+'/lib')\"):$LD_LIBRARY_PATH"
            )
        self._input = self._sess.get_inputs()[0].name

    def infer(self, blob):
        # blob is the cv2.dnn (1,3,S,S) float32 NCHW tensor — exactly the ONNX input.
        return self._sess.run(None, {self._input: blob})[0]     # (1, 84, N)


def _parse_cuda_version(text: str):
    """The driver's max CUDA version out of ``nvidia-smi`` header text, or None.

    Newer drivers (610.x) relabelled the field ``CUDA UMD Version:`` (#85) —
    the optional ``UMD`` matches both spellings, old and new."""
    import re

    m = re.search(r"CUDA(?: UMD)? Version:\s*([0-9]+(?:\.[0-9]+)?)", text or "")
    return float(m.group(1)) if m else None


def _driver_cuda_version():
    """The installed NVIDIA driver's CUDA capability (e.g. 13.3), or None when
    there's no working driver/nvidia-smi. This is the **driver's** ceiling, not
    the toolkit version — exactly what gates TensorRT (#82).

    The ``nvidia-smi`` header is a fragile signal (it was relabelled once
    already, #85), so a working CUDA torch is accepted as a secondary source:
    torch only *runs* CUDA if the driver supports its toolkit version, so
    ``torch.version.cuda`` is a valid lower bound on the driver's ceiling."""
    import subprocess

    try:
        out = subprocess.run(["nvidia-smi"], capture_output=True, text=True,
                             timeout=10).stdout
    except Exception:       # noqa: BLE001 — no driver, no binary, no GPU: all "no"
        out = ""
    ver = _parse_cuda_version(out)
    if ver is not None:
        return ver
    try:
        import torch

        if torch.cuda.is_available() and torch.version.cuda:
            return _parse_cuda_version(f"CUDA Version: {torch.version.cuda}")
    except Exception:       # noqa: BLE001 — torch optional/broken: no signal
        pass
    return None


def _strip_engine_metadata(data: bytes) -> bytes:
    """Drop the Ultralytics metadata header from an exported ``.engine`` file.

    Ultralytics prepends ``<4-byte LE length><JSON dict>`` to the serialized
    engine; raw TensorRT can't deserialize that. A bare engine passes through
    unchanged."""
    import json
    import struct

    if len(data) > 8:
        (n,) = struct.unpack("<I", data[:4])
        if 0 < n < len(data) - 4:
            try:
                meta = json.loads(data[4:4 + n].decode("utf-8"))
                if isinstance(meta, dict):
                    return data[4 + n:]
            except (UnicodeDecodeError, ValueError):
                pass
    return data


class _TensorRtRunner:
    """A prebuilt TensorRT engine behind ``infer`` (#82).

    Measured on the NAS 3070 (real 3×3 tiled pipeline): 26x FP16 175 ms → 111 ms
    (1.6×), identical 91%/0% accuracy; 26m 1.37×, 11n 1.21×. The engine file is
    GPU-specific — deserialization fails loudly if it was built elsewhere.

    Honest status: written against the TensorRT 10 API (``cuda-python`` for the
    device buffers) but **not yet run on real hardware from this app** — the #82
    benchmark used its own harness. First NAS run should compare a few frames'
    boxes against the onnx-cuda runner before trusting it.
    """

    def __init__(self, engine_file: str):
        import numpy as np
        import tensorrt as trt
        from cuda import cudart

        self._cudart = cudart
        self._np = np
        with open(engine_file, "rb") as fh:
            data = _strip_engine_metadata(fh.read())
        runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        engine = runtime.deserialize_cuda_engine(data)
        if engine is None:
            raise RuntimeError(
                f"couldn't deserialize {os.path.basename(engine_file)} — TensorRT "
                "engines are GPU-specific: rebuild it on THIS machine "
                "(python d20app/models/export_trt_engine.py <variant>, #82)")
        self._engine = engine
        self._ctx = engine.create_execution_context()
        self._in_name = self._out_name = None
        for i in range(engine.num_io_tensors):
            name = engine.get_tensor_name(i)
            if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self._in_name = name
            else:
                self._out_name = name

        def np_dtype(name):
            return np.float16 if engine.get_tensor_dtype(name) == trt.DataType.HALF \
                else np.float32

        self._in_dtype = np_dtype(self._in_name)
        self._out_dtype = np_dtype(self._out_name)
        self._in_shape = tuple(engine.get_tensor_shape(self._in_name))
        self._out_shape = tuple(engine.get_tensor_shape(self._out_name))
        self._in_bytes = int(np.prod(self._in_shape)) * np.dtype(self._in_dtype).itemsize
        self._out_bytes = int(np.prod(self._out_shape)) * np.dtype(self._out_dtype).itemsize
        self._d_in = self._malloc(self._in_bytes)
        self._d_out = self._malloc(self._out_bytes)
        self._ctx.set_tensor_address(self._in_name, self._d_in)
        self._ctx.set_tensor_address(self._out_name, self._d_out)
        err, self._stream = cudart.cudaStreamCreate()
        self._check(err)

    def _check(self, err) -> None:
        if err != self._cudart.cudaError_t.cudaSuccess:
            raise RuntimeError(f"CUDA error in the TensorRT runner: {err}")

    def _malloc(self, nbytes: int):
        err, ptr = self._cudart.cudaMalloc(nbytes)
        self._check(err)
        return ptr

    def infer(self, blob):
        np, cudart = self._np, self._cudart
        host_in = np.ascontiguousarray(blob.astype(self._in_dtype, copy=False))
        self._check(cudart.cudaMemcpyAsync(
            self._d_in, host_in.ctypes.data, self._in_bytes,
            cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, self._stream)[0])
        if not self._ctx.execute_async_v3(self._stream):
            raise RuntimeError("TensorRT inference failed (execute_async_v3)")
        host_out = np.empty(self._out_shape, self._out_dtype)
        self._check(cudart.cudaMemcpyAsync(
            host_out.ctypes.data, self._d_out, self._out_bytes,
            cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, self._stream)[0])
        self._check(cudart.cudaStreamSynchronize(self._stream)[0])
        return host_out.astype(np.float32, copy=False)   # (1, 84, N), like the others


def _load_tensorrt(variant: str):
    """Bring up the TensorRT runner, or raise an actionable ``RuntimeError``.

    Ordered so the cheapest/most-fundamental failure speaks first: driver
    capability → package → engine file. The driver guard exists because
    installing tensorrt on a pre-CUDA-13 driver **breaks torch** (#82) — the
    error says so instead of letting anyone pip their way into that hole.
    """
    ver = _driver_cuda_version()
    if ver is None:
        raise RuntimeError(
            "tensorrt accelerator: no working NVIDIA driver detected (nvidia-smi "
            "not found or failed). Use 'auto', 'onnx-cuda', or 'cpu'.")
    if ver < _TRT_MIN_CUDA:
        raise RuntimeError(
            f"tensorrt needs a driver supporting CUDA >= {_TRT_MIN_CUDA:g}, but this "
            f"driver reports CUDA {ver:g}. Do NOT pip-install tensorrt on this driver "
            "— its CUDA-13 dependencies break torch (#82). Upgrade the NVIDIA driver "
            "first (NVIDIA's CUDA repo; Debian 12's own repo caps at 535/12.2), or "
            "use 'onnx-cuda'.")
    try:
        import tensorrt  # noqa: F401 — optional, opt-in dep
        from cuda import cudart  # noqa: F401 — cuda-python, ships alongside
    except ImportError as exc:
        raise RuntimeError(
            "tensorrt accelerator: the 'tensorrt' + 'cuda-python' packages aren't "
            "installed. On this CUDA-13-capable driver: pip install tensorrt "
            "cuda-python (never on an older driver — it breaks torch, #82).") from exc
    epath = engine_path(variant)
    if not os.path.exists(epath):
        raise RuntimeError(
            f"no TensorRT engine for {variant} ({os.path.relpath(epath)}). Engines "
            "are GPU-specific and take minutes to build, so they're a cached, "
            "one-time setup step: python d20app/models/export_trt_engine.py "
            f"{variant} (see models/README.md, #82).")
    return _TensorRtRunner(epath)


def load_net(variant: str = DEFAULT_VARIANT, accelerator: str = "cpu"):
    """Build an inference runner for a YOLO11 variant on the chosen accelerator.

    Returns an object with ``.infer(blob) -> (1, 84, N)``. Raises ``ValueError``
    for an unknown variant/accelerator, ``FileNotFoundError`` if the model is
    missing, or ``RuntimeError`` if an OpenVINO device can't be brought up (the
    caller decides whether to fall back to ``cpu``).
    """
    import cv2

    if variant in DROPPED_MODELS:
        raise ValueError(
            f"model {variant!r} was {DROPPED_MODELS[variant]}. Pick one of "
            f"{[v for v, m in MODELS.items() if m.get('selectable', True)]} instead "
            "(see the benchmark, issue #70)."
        )
    if variant not in MODELS:
        raise ValueError(f"unknown YOLO variant {variant!r}; known: {sorted(MODELS)}")
    accel = (accelerator or "cpu").lower()
    if accel not in ACCELERATORS:
        raise ValueError(f"unknown accelerator {accel!r}; known: {list(ACCELERATORS)}")

    fb_reason = ""          # why a fallback happened (#90): surfaced, never silent
    if accel == "tensorrt":
        # Opt-in engine path (#82). Runs BEFORE the ONNX existence check — a
        # machine can legitimately hold only the .engine (26x's ONNX is
        # export-only). Any failure falls back to 'auto' (verified CUDA, else
        # CPU — both say so out loud): same model, same accuracy, just slower,
        # which is the established accelerator-degradation stance.
        try:
            return _annotate(_load_tensorrt(variant), "tensorrt")
        except RuntimeError as exc:
            _log.warning("tensorrt unavailable (%s) — falling back to 'auto' "
                         "(verified CUDA, else CPU)", exc)
            fb_reason = f"requested tensorrt — unavailable: {exc}"
            accel = "auto"

    path = model_path(variant)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"YOLO model {os.path.relpath(path)} is missing. "
            "See d20app/models/README.md to export it, or pick a bundled model "
            "(yolo11n / yolo26m) in the GUI."
        )

    if accel == "auto":
        # The benchmark-settled default (#71): CUDA when it genuinely binds —
        # the runner below verifies the provider actually selected, so "auto"
        # can never silently run slow — else CPU, said out loud.
        try:
            return _annotate(_OnnxRuntimeRunner(path), "onnx-cuda", fb_reason)
        except Exception as exc:            # noqa: BLE001 — any CUDA absence → CPU
            _log.warning(
                "accelerator 'auto': CUDA unavailable (%s) — running %s on CPU",
                exc, variant)
            reason = "; ".join(r for r in
                               (fb_reason, f"CUDA unavailable: {exc}") if r)
            return _annotate(_CvDnnRunner(cv2.dnn.readNetFromONNX(path)),
                             "cpu", reason)

    if accel in ("openvino-gpu", "openvino-auto"):
        device = "GPU" if accel == "openvino-gpu" else "AUTO"
        try:
            return _annotate(_OpenVinoRunner(path, device), accel)
        except Exception as exc:            # noqa: BLE001 — surface a clear, actionable error
            raise RuntimeError(
                f"OpenVINO {device} backend unavailable ({exc}). Install the optional "
                "'openvino' package and the Intel GPU compute drivers, or use the CPU "
                "accelerator."
            ) from exc

    if accel == "onnx-cuda":
        try:
            return _annotate(_OnnxRuntimeRunner(path), "onnx-cuda")
        except ImportError as exc:          # onnxruntime not installed at all
            raise RuntimeError(
                "onnx-cuda needs the optional 'onnxruntime-gpu' package (CUDA-12 build). "
                "Install it on the NVIDIA host, or use the CPU accelerator."
            ) from exc
        # other failures (no CUDA / silent-CPU) already raise a clear RuntimeError above;
        # the caller (PersonDetector._ensure_net) retries the same model on CPU.

    net = cv2.dnn.readNetFromONNX(path)
    if accel == "opencl":
        # OpenCV silently runs on CPU if there's no OpenCL device, so this is safe.
        return _annotate(_CvDnnRunner(net, target=cv2.dnn.DNN_TARGET_OPENCL_FP16),
                         "opencl")
    return _annotate(_CvDnnRunner(net), "cpu")


def _letterbox(frame, size: int):
    """Resize ``frame`` into a ``size``x``size`` square, padding to keep aspect.

    Returns ``(canvas, ratio, pad_x, pad_y)`` so boxes can be mapped back to the
    original frame's pixel coordinates.
    """
    import cv2
    import numpy as np

    h, w = frame.shape[:2]
    r = min(size / h, size / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    resized = cv2.resize(frame, (nw, nh))
    canvas = np.full((size, size, 3), 114, np.uint8)
    pad_x, pad_y = (size - nw) // 2, (size - nh) // 2
    canvas[pad_y:pad_y + nh, pad_x:pad_x + nw] = resized
    return canvas, r, pad_x, pad_y


def detect_boxes(net, frame, floor: float, size: int = INPUT_SIZE) -> list:
    """Run a YOLO11 runner on one BGR ``frame``; return ``[(label, score, box)]``.

    ``net`` is a runner from :func:`load_net` (``.infer``); a raw ``cv2.dnn`` net
    is also accepted for back-compat. ``floor`` is the confidence below which
    detections are dropped. Coordinates are pixels within ``frame``.
    """
    import cv2
    import numpy as np

    lb, r, pad_x, pad_y = _letterbox(frame, size)
    blob = cv2.dnn.blobFromImage(lb, 1 / 255.0, (size, size), swapRB=True, crop=False)
    if hasattr(net, "infer"):
        out = net.infer(blob)                 # (1, 84, N) — runner (cv2.dnn or OpenVINO)
    else:                                     # back-compat: a bare cv2.dnn net
        net.setInput(blob)
        out = net.forward()
    # Golden-export guard (#71): this decoder reads the RAW (1, 84, N) head. An
    # end2end export (YOLO26's default) emits (1, 300, 6) — NMS baked in — which
    # this decode silently mis-reads as garbage scores, quietly costing 4–9 recall
    # points. Refuse it loudly instead.
    if getattr(out, "ndim", 0) != 3 or out.shape[1] > out.shape[2]:
        raise RuntimeError(
            f"YOLO output shape {getattr(out, 'shape', None)} is not the raw "
            "(1, 84, N) detection head — this looks like an end2end (NMS-baked) "
            "export, which mis-decodes here. Re-export with end2end=False (the "
            "golden recipe: scripts/export_yolo.py, issue #71)."
        )
    out = np.squeeze(out, 0).T                 # (N, 84): [cx, cy, w, h, 80 class scores]
    class_scores = out[:, 4:]
    class_ids = class_scores.argmax(axis=1)
    confs = class_scores.max(axis=1)
    keep = confs >= floor
    out, class_ids, confs = out[keep], class_ids[keep], confs[keep]
    if len(out) == 0:
        return []

    rects, scores, ids = [], [], []
    for row, cid, cf in zip(out, class_ids, confs):
        cx, cy, bw, bh = row[:4]
        x1 = (cx - bw / 2 - pad_x) / r
        y1 = (cy - bh / 2 - pad_y) / r
        rects.append([int(x1), int(y1), int(bw / r), int(bh / r)])
        scores.append(float(cf))
        ids.append(int(cid))

    boxes = []
    idxs = cv2.dnn.NMSBoxes(rects, scores, float(floor), _NMS_IOU)
    for i in np.array(idxs).reshape(-1):
        x, y, w, h = rects[i]
        cid = ids[i]
        label = COCO_CLASSES[cid] if 0 <= cid < len(COCO_CLASSES) else str(cid)
        boxes.append((label, scores[i], (x, y, x + w, y + h)))
    return boxes


def merge_nms(boxes: list, floor: float) -> list:
    """Per-class NMS over ``[(label, score, (x1, y1, x2, y2))]`` boxes.

    Model-agnostic — used to de-duplicate detections collected from overlapping
    tiles (a subject in a seam region is found by two tiles; NMS keeps one). Reuses
    the same :data:`_NMS_IOU` as the single-pass path.
    """
    import cv2
    import numpy as np

    by_label: dict = {}
    for label, score, box in boxes:
        by_label.setdefault(label, []).append((float(score), box))
    merged = []
    for label, items in by_label.items():
        rects = [[b[0], b[1], b[2] - b[0], b[3] - b[1]] for _, b in items]
        scores = [s for s, _ in items]
        idxs = cv2.dnn.NMSBoxes(rects, scores, float(floor), _NMS_IOU)
        for i in np.array(idxs).reshape(-1):
            merged.append((label, items[i][0], items[i][1]))
    return merged
