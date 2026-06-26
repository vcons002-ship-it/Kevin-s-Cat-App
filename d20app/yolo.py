"""YOLO11 object detection via OpenCV's ``cv2.dnn`` (ONNX) — no PyTorch at runtime.

A drop-in alternative backend for :class:`d20app.detector.PersonDetector`: it
produces the same ``[(label, score, (x1, y1, x2, y2))]`` boxes the MobileNet-SSD
path does, so the rest of the pipeline (person trigger, ``cat`` labelling,
annotated snapshots) is unchanged.

Why offer it: the bundled MobileNet-SSD is fast but weak in low light / odd poses
(it scored 0.00 on a real dim night frame). YOLO11n scored ~0.87 on the same
frame at ~1.4x the CPU — a much better night/occlusion detector for a modest cost.

Two variants are available (see :data:`MODELS`):

- ``yolo11n`` — nano at 320x320 (~11 MB, ~28 ms/frame). The default; it already
  handles the night case well.
- ``yolo11m`` — medium at 640x640 (~77 MB, ~500 ms/frame on CPU). Bigger and
  slower; on our own night/day frames it did **not** beat nano on the night case
  that motivated it, so it's offered as an option, not the default. Worth trying
  if you have CPU headroom and want the extra capacity on hard scenes.

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
  2–4× CPU on Intel hardware and the thing that makes the heavier ``yolo11m``
  practical. **Intel-only**; needs the host Intel GPU compute drivers.

Whatever the backend, inference returns the same ``(1, 84, N)`` tensor, so the
letterbox + NMS decode below is shared. ``load_net`` returns a small *runner*
(``.infer(blob) -> ndarray``) wrapping whichever engine was selected.
"""

from __future__ import annotations

import os

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
MODELS = {
    "yolo11n": {"file": "yolo11n.onnx", "size": 320},
    "yolo11m": {"file": "yolo11m.onnx", "size": 640},
}
DEFAULT_VARIANT = "yolo11n"

# Where each accelerator runs the conv layers.
ACCELERATORS = ("cpu", "opencl", "openvino-gpu", "openvino-auto")

# Back-compat aliases for the single-model era (some tests/callers import these).
ONNX_PATH = os.path.join(_MODELS_DIR, MODELS[DEFAULT_VARIANT]["file"])
INPUT_SIZE = MODELS[DEFAULT_VARIANT]["size"]
_NMS_IOU = 0.45


def model_path(variant: str = DEFAULT_VARIANT) -> str:
    """Absolute path to a variant's ONNX file (no existence check)."""
    return os.path.join(_MODELS_DIR, MODELS[variant]["file"])


def input_size(variant: str = DEFAULT_VARIANT) -> int:
    """The fixed square input size the given variant was exported at."""
    return MODELS[variant]["size"]


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


def load_net(variant: str = DEFAULT_VARIANT, accelerator: str = "cpu"):
    """Build an inference runner for a YOLO11 variant on the chosen accelerator.

    Returns an object with ``.infer(blob) -> (1, 84, N)``. Raises ``ValueError``
    for an unknown variant/accelerator, ``FileNotFoundError`` if the model is
    missing, or ``RuntimeError`` if an OpenVINO device can't be brought up (the
    caller decides whether to fall back to ``cpu``).
    """
    import cv2

    if variant not in MODELS:
        raise ValueError(f"unknown YOLO variant {variant!r}; known: {sorted(MODELS)}")
    accel = (accelerator or "cpu").lower()
    if accel not in ACCELERATORS:
        raise ValueError(f"unknown accelerator {accel!r}; known: {list(ACCELERATORS)}")
    path = model_path(variant)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"YOLO model {os.path.relpath(path)} is missing. "
            "See d20app/models/README.md to export it, or use the MobileNet-SSD model."
        )

    if accel in ("openvino-gpu", "openvino-auto"):
        device = "GPU" if accel == "openvino-gpu" else "AUTO"
        try:
            return _OpenVinoRunner(path, device)
        except Exception as exc:            # noqa: BLE001 — surface a clear, actionable error
            raise RuntimeError(
                f"OpenVINO {device} backend unavailable ({exc}). Install the optional "
                "'openvino' package and the Intel GPU compute drivers, or use the CPU "
                "accelerator."
            ) from exc

    net = cv2.dnn.readNetFromONNX(path)
    if accel == "opencl":
        # OpenCV silently runs on CPU if there's no OpenCL device, so this is safe.
        return _CvDnnRunner(net, target=cv2.dnn.DNN_TARGET_OPENCL_FP16)
    return _CvDnnRunner(net)


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
