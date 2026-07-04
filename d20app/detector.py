"""Person detection: cheap motion pre-filter + COCO YOLO (YOLO11 / YOLO26).

The goal is to trigger **only when a person enters frame and to ignore the
cats**. We do this in two stages to keep CPU low on a GPU-less NAS:

1. A frame-difference *motion pre-filter* — skip the neural net entirely while
   the scene is static.
2. When motion is seen, run a YOLO model (COCO) via :mod:`d20app.yolo`. COCO has
   ``person`` and ``cat`` as separate classes, so we report a trigger only for a
   ``person`` box above the confidence threshold and ignore ``cat`` (and the rest).

The box-decode core lives in :mod:`d20app.yolo` and is unit-testable without a
camera. (MobileNet-SSD was removed in 0.25.0 — YOLO won every benchmark.)
"""

from __future__ import annotations

import logging
import os
import sys
import threading
import time
from dataclasses import dataclass, field

_log = logging.getLogger(__name__)

# Make OpenCV's FFmpeg backend behave like VLC/ffplay for RTSP cameras:
#   * rtsp_transport;tcp — use TCP (retransmitted, in-order packets) instead of
#     the lossy UDP default, which eliminates most "error while decoding MB…"
#     and "missing reference picture" decoder spam.
#   * timeout;5000000 — fail a dead/unreachable camera after 5s (microseconds)
#     instead of blocking the loop on the OS default of a minute or more.
#   * a quiet log level — stop libavcodec printing cosmetic decode errors and
#     repeated "401 Unauthorized" lines straight to the console; the app reports
#     real failures in its own Activity log instead.
# All use setdefault() so an advanced user can override them from the shell,
# e.g. OPENCV_FFMPEG_LOGLEVEL=24 to see warnings again while debugging.
os.environ.setdefault(
    "OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp|timeout;5000000"
)
os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "8")   # 8 = AV_LOG_FATAL

# Confidence floor for *naming* a non-person mover (e.g. "cat") in the log.
# Lower than the person threshold so distant/uncertain cats still get identified.
_LABEL_FLOOR = 0.3


class CameraError(RuntimeError):
    """The camera stream could not be opened or read (bad URL, auth, network)."""


@dataclass
class FrameOutcome:
    """What one processed frame contained."""

    motion: bool             # did the cheap motion pre-filter trigger?
    person: bool             # was a person detected above the threshold?
    labels: tuple = ()       # other classes seen (e.g. ("cat",)), best score first


_cv2_quieted = False


def _quiet_cv2_logs(cv2) -> None:
    """Hush OpenCV's own WARN chatter (e.g. the videoio backend warning) once.

    This is separate from OPENCV_FFMPEG_LOGLEVEL, which only governs FFmpeg.
    """
    global _cv2_quieted
    if _cv2_quieted:
        return
    try:
        cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
    except Exception:
        pass
    _cv2_quieted = True


def parse_local_index(source):
    """Return device index N when ``source`` is a local ``"usb:N"`` string, else None."""
    if isinstance(source, str) and source.startswith("usb:"):
        try:
            return int(source[4:])
        except ValueError:
            return None
    return None


def _open_capture(source):
    """Open ``source`` with the right OpenCV backend.

    A local ``usb:N`` camera opens by device index with the platform backend
    (DirectShow on Windows, V4L2 on Linux); everything else is a network stream
    or file opened through FFmpeg (so RTSP auth behaves like VLC).
    """
    import cv2

    idx = parse_local_index(source)
    if idx is not None:
        if sys.platform.startswith("win"):
            backend = cv2.CAP_DSHOW
        elif sys.platform.startswith("linux"):
            backend = cv2.CAP_V4L2
        else:
            backend = cv2.CAP_ANY
        return cv2.VideoCapture(idx, backend)
    return cv2.VideoCapture(source, cv2.CAP_FFMPEG)


def grab_frame_jpeg(source: str, skip: int = 4):
    """Open ``source``, grab one frame, and return it as JPEG bytes (or None).

    Used by the GUI's region-of-interest picker to show a still from the camera.
    A few frames are skipped so the returned image isn't the codec's first
    (often grey/partial) frame.
    """
    import cv2

    _quiet_cv2_logs(cv2)
    cap = _open_capture(source)
    if not cap.isOpened():
        cap.release()
        return None
    frame = None
    for _ in range(skip + 1):
        ok, f = cap.read()
        if ok and f is not None:
            frame = f
    cap.release()
    if frame is None:
        return None
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return buf.tobytes() if ok else None


def apply_image_adjustments(frame, gamma: float = 1.0, brightness: int = 0,
                            contrast: float = 1.0, saturation: float = 1.0):
    """Return ``frame`` with gamma / brightness / contrast / saturation applied.

    All defaults are no-ops, and a no-op skips the work and returns the input
    unchanged (no copy). Applied to the frame *before* the net runs, so it can lift
    a too-dark or washed-out feed into the detector's comfort zone. Order: gamma
    (lifts shadows) → brightness/contrast (linear) → saturation (HSV).
    """
    import cv2
    import numpy as np

    out = frame
    if gamma and gamma > 0 and abs(gamma - 1.0) > 1e-3:
        inv = 1.0 / float(gamma)
        lut = np.array([((i / 255.0) ** inv) * 255 for i in range(256)],
                       dtype=np.uint8)
        out = cv2.LUT(out, lut)
    if abs(float(contrast) - 1.0) > 1e-3 or int(brightness) != 0:
        out = cv2.convertScaleAbs(out, alpha=float(contrast), beta=float(brightness))
    if saturation is not None and saturation >= 0 and abs(float(saturation) - 1.0) > 1e-3:
        hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV).astype("float32")
        hsv[..., 1] = np.clip(hsv[..., 1] * float(saturation), 0, 255)
        out = cv2.cvtColor(hsv.astype("uint8"), cv2.COLOR_HSV2BGR)
    return out


def sample_video_frames(path: str, n: int = 8) -> list:
    """Decode ``path`` and return up to ``n`` evenly-spaced BGR frames.

    For the GUI's "test a video" tool: gives a filmstrip to run detection on
    without processing every frame. Falls back to sequential sampling when the
    container doesn't report a frame count. Returns ``[]`` if it can't be read.
    """
    import cv2

    _quiet_cv2_logs(cv2)
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        cap.release()
        return []
    n = max(1, int(n))
    frames = []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    try:
        if total > 1:
            count = min(n, total)
            idxs = [int(round(i * (total - 1) / max(1, count - 1)))
                    for i in range(count)] if count > 1 else [0]
            for idx in idxs:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ok, fr = cap.read()
                if ok and fr is not None:
                    frames.append(fr)
        if not frames:
            # No reliable frame count (some streams/codecs): read sequentially and
            # keep a bounded, evenly-thinned sample.
            grabbed = []
            for _ in range(n * 30):
                ok, fr = cap.read()
                if not ok or fr is None:
                    break
                grabbed.append(fr)
            if grabbed:
                step = max(1, len(grabbed) // n)
                frames = grabbed[::step][:n]
    finally:
        cap.release()
    return frames


def mask_credentials(url: str) -> str:
    """Hide the password in an ``rtsp://user:pass@host`` URL for safe logging."""
    if "://" not in url or "@" not in url:
        return url
    scheme, rest = url.split("://", 1)
    creds, host = rest.rsplit("@", 1)        # host never contains '@'
    if ":" in creds:
        user, _ = creds.split(":", 1)
        creds = f"{user}:***"
    return f"{scheme}://{creds}@{host}"


class MotionPrefilter:
    """Detect motion by comparing consecutive grayscale frames.

    Returns True only when there's a **solid, compact** region of change — not
    just enough changed pixels. This rejects two common false triggers:

    * **Sensor noise / compression grain** — removed by a median blur (which
      also erases thin specks without smearing real edges) plus a morphological
      opening of the change mask.
    * **Decode-artifact bands** — a corrupt camera frame often shows a long,
      *thin* line of bad pixels. That line has lots of changed pixels but is
      only a few pixels tall, so we reject any blob whose shorter side is below
      ``min_blob_px``.

    The first frame reports **no** motion (nothing to compare against yet), so a
    static scene never triggers detection until something really moves.
    """

    def __init__(self, min_area_frac: float = 0.003, diff_threshold: int = 25,
                 min_blob_px: int = 14) -> None:
        self.min_area_frac = min_area_frac
        self.diff_threshold = diff_threshold
        self.min_blob_px = min_blob_px
        self._prev = None
        self._kernel = None
        # WHERE the last update saw motion (the escalation ladder's "look here"
        # hints, #66-era data we used to throw away): up to _max_blobs solid-blob
        # boxes as (x1, y1, x2, y2) ints in the same (ROI-cropped) frame coords the
        # detections use, largest first, plus the wall-clock time of that update.
        # Includes sub-min-area blobs — a distant cat's small mover is exactly the
        # hint the zoom rung wants — but never thin lines/specks (min_blob_px still
        # applies), and the motion VERDICT below is unchanged. The list is rebound
        # on each update, never mutated, so readers on other threads may hold a
        # stale-but-consistent list without a lock.
        self.last_blobs: list = []
        self.last_blobs_ts: float = 0.0
        self._max_blobs = 8

    def update(self, gray) -> bool:
        import cv2  # local import: keep module importable without OpenCV

        # Median blur kills salt-and-pepper noise and thin corruption lines
        # without widening real edges (a Gaussian blur would smear a 1px line
        # into a band that survives later filtering).
        clean = cv2.medianBlur(gray, 5)
        if self._prev is None:
            self._prev = clean
            self.last_blobs = []
            return False
        delta = cv2.absdiff(self._prev, clean)
        self._prev = clean
        _, thresh = cv2.threshold(delta, self.diff_threshold, 255, cv2.THRESH_BINARY)
        if self._kernel is None:
            self._kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, self._kernel)

        h, w = thresh.shape[:2]
        min_area = self.min_area_frac * h * w
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        moved = False
        blobs = []                                # (area, (x1, y1, x2, y2))
        for c in contours:
            x, y, bw, bh = cv2.boundingRect(c)
            if min(bw, bh) < self.min_blob_px:    # thin line/speck — not even a hint
                continue
            area = cv2.contourArea(c)
            blobs.append((area, (x, y, x + bw, y + bh)))
            if area >= min_area:                  # the original verdict, unchanged
                moved = True
        blobs.sort(key=lambda t: -t[0])
        self.last_blobs = [b for _, b in blobs[:self._max_blobs]]
        self.last_blobs_ts = time.time()
        return moved


class PersonDetector:
    """Open a camera stream and report when a person (not a cat) is present.

    Combines :class:`MotionPrefilter` with a YOLO network. Handles stream
    reconnection with backoff so a flaky camera doesn't kill the loop.
    """

    def __init__(self, source: str, confidence: float = 0.5, roi=None,
                 detect_size: int = 300, label_floor: float = _LABEL_FLOOR,
                 motion_min_area_frac: float = 0.003, motion_diff_threshold: int = 25,
                 motion_min_blob_px: int = 14, motion_gate: bool = True,
                 model: str = "yolo11n",
                 accelerator: str = "cpu", smooth_feed: bool = False,
                 gamma: float = 1.0, brightness: int = 0,
                 contrast: float = 1.0, saturation: float = 1.0,
                 cat_scan_tiling: str = "off", cat_scan_tile_overlap: float = 0.2,
                 cat_scan_imgsz: int = 0, cat_scan_frames: int = 1,
                 cat_confidence: float = 0.5,
                 locator_classes=None, track_fusion: bool = True) -> None:
        self.source = source
        self.confidence = confidence
        # The locator ("the cat") path: which classes count, and the confidence they
        # must clear — independent of the person threshold and of label_floor. A house
        # with no dog can include "dog" since the model often mislabels a cat as one.
        self.cat_confidence = float(cat_confidence)
        self.locator_classes = tuple(locator_classes) if locator_classes else ("cat",)
        # Locator-scan resolution knobs (used only by the still-cat forced scan, not
        # the fast treat path). See :meth:`_detect_locator`.
        self.cat_scan_tiling = cat_scan_tiling or "off"
        self.cat_scan_tile_overlap = float(cat_scan_tile_overlap)
        self.cat_scan_imgsz = int(cat_scan_imgsz or 0)
        self.cat_scan_frames = max(1, min(int(cat_scan_frames or 1),
                                          self._AVG_MAX_FRAMES))
        self.track_fusion = bool(track_fusion)
        self._locator_runner = None     # lazily-loaded larger-input YOLO net (Option A)
        self._locator_size = None
        self._locator_model = None
        self._locator_tried = False
        # Image adjustments applied to each frame before the net (and the live feed
        # in non-smooth mode) sees it. Defaults are no-ops. See
        # :func:`apply_image_adjustments`.
        self.gamma = float(gamma)
        self.brightness = int(brightness)
        self.contrast = float(contrast)
        self.saturation = float(saturation)
        # Which YOLO model to run: "yolo11n" (default — low light/odd poses, ~1.4x
        # CPU), "yolo26m"/"yolo26x" (bigger/stronger), etc. A model that
        # can't load raises a clear error (MobileNet-SSD was removed in 0.25.0).
        self.model = model or "yolo11n"
        # Where the YOLO conv layers run: "cpu" (default), "opencl" (iGPU via
        # OpenCL), or "openvino-gpu"/"openvino-auto" (Intel OpenVINO runtime). If a
        # GPU backend can't start, we retry the same model on CPU before giving up.
        self.accelerator = accelerator or "cpu"
        self._yolo = None       # the YOLO inference runner, lazily loaded
        self._yolo_size = None  # the loaded variant's fixed input size
        self.roi = roi          # optional [x, y, w, h]
        # Legacy MobileNet-SSD net input size. Ignored since 0.25.0 — YOLO uses its
        # exported fixed size; every deployed model is a 640 export (#70/#80).
        # Kept as a no-op so old callers/configs don't break.
        self.detect_size = int(detect_size) if detect_size else 300
        # Min confidence to NAME a non-person mover (cat/pottedplant/…) in the log
        # and draw it on a snapshot. Higher = fewer stray labels; doesn't affect
        # whether a person triggers a treat (that's `confidence`).
        self.label_floor = float(label_floor)
        self._cap = None
        self._read_fails = 0
        self.frame_size = None  # (w, h) of the last good frame; None until one reads
        self._last_frame = None    # last frame the net analysed (for snapshots)
        self._last_boxes = []      # detections on that frame: [(label, score, box)]
        # Live-feed state: the most recent **raw** frame read (live_jpeg crops it
        # to the ROI on demand) and when the boxes were last refreshed by the net.
        # ``_live_version`` bumps on every new frame / box update so the stream can
        # skip re-encoding an unchanged frame. Guarded by a lock because the web
        # request thread reads them while the loop (or grab thread) writes.
        self._live_frame = None
        self._live_boxes_at = 0.0
        self._cat_last_seen = 0.0       # monotonic time the net last saw a cat (any motion)
        self._live_published_at = 0.0   # monotonic time of the last frame publish
        self._live_version = 0
        self._live_lock = threading.Lock()
        # Smooth feed: when on, a dedicated thread reads the camera continuously so
        # the live feed runs at camera rate, decoupled from (slow) inference. Off
        # by default — the loop reads frames itself, one per analysed iteration.
        # ``smooth_feed`` is the *actual* state (the grabber is started lazily by
        # the loop thread); ``_smooth_desired`` is the requested state (set here or
        # later from the web thread). They differ until the loop reconciles them,
        # so the grabber is started/stopped by one thread only — never racing the
        # camera read.
        self.smooth_feed = False
        self._smooth_desired = bool(smooth_feed)
        self._grab_thread = None
        self._grab_stop = threading.Event()
        self._grab_error = ""
        self._grab_fails = 0    # grab thread's own empty-read counter (not shared)
        # #96: motion-gate off = run the net on EVERY frame (cheap with the
        # accelerated models). The motion pre-filter still runs — its verdict
        # still feeds outcome.motion (rolls need a real entrance), the trail,
        # and the null-frame bookkeeping — it just stops gating the net.
        self.motion_gate = bool(motion_gate)
        self._motion = MotionPrefilter(
            min_area_frac=motion_min_area_frac,
            diff_threshold=motion_diff_threshold,
            min_blob_px=motion_min_blob_px,
        )
        # The "cat trail" (#67): null-frame silhouettes coloured by recency, fed by
        # every frame read. Cheap (capped working resolution) and thread-safe.
        from .trail import TrailTracker
        self._trail = TrailTracker(diff_threshold=motion_diff_threshold)
        # Person exclusion for the trail (0.41.0): the freshest person boxes the
        # net produced, so human movement doesn't stamp cat-sized blobs. Written
        # and read on the camera worker thread only — no lock needed.
        self._person_boxes: list = []
        self._person_boxes_at = 0.0
        # Frame ring buffer (#68): the last few seconds of (downscaled) frames for
        # the temporal VLM mosaic ("did a cat pass through?"). ~8 frames spaced ≥1 s
        # at ≤480 px ≈ well under 3 MB per camera. Guarded by _live_lock.
        from collections import deque
        self._ring: "deque" = deque(maxlen=self._RING_FRAMES)
        self._ring_last = 0.0
        # Temporal score fusion (0.37.0): weak locator hits accumulated across
        # frames — a consistently-moving 0.35-confidence cat becomes one confirmed
        # sighting no single frame could produce. Pure YOLO evidence (0.33.0 rules).
        from .fusion import TrackFuser
        self._fusion = TrackFuser()
        self._fused_hit = None          # last unclaimed confirmation (see take_fused_hit)
        # Targeted boost (0.42.0): an unconfirmed lead's box — forced scans zoom a
        # full-resolution crop around it and run the heaviest available model on
        # it (#70: heavier model helps; raw >640 input doesn't — the crop is the
        # resolution lever that measured well). Hint written by web threads, read
        # by the worker: the tuple is rebound, never mutated (lock-free pattern).
        self._boost_hint = None         # ((x1, y1, x2, y2), monotonic deadline)
        self._boost_runner = None       # heavier net, loaded once on first use
        self._boost_size = 0
        self._boost_model = ""
        self._boost_tried = False
        # Newest confirmed cat position (box, label, score, wall ts) — feeds the
        # "last known location" overlay at per-detection freshness (0.42.1).
        # Guarded by _live_lock; never written to the sightings log from here.
        self._cat_live_confirmed = None

    # -- model / stream lifecycle -------------------------------------------
    def _ensure_net(self):
        if self._yolo is not None:
            return self._yolo
        from . import yolo
        # Precision is resolved from the accelerator (#90): the logical model
        # ("yolo26x") maps to the FP16 file on onnxruntime/TensorRT paths and
        # the FP32 file on cv2.dnn — never a user-facing _fp16 choice.
        resolved = yolo.resolve_variant(self.model, self.accelerator)
        try:
            self._yolo = yolo.load_net(resolved, self.accelerator)
        except ValueError as exc:
            # Unknown or DROPPED model (#79): a config error, not a camera hiccup —
            # surface it through the loop's fatal path (same as a missing file) so
            # it stops loudly instead of retrying forever.
            raise FileNotFoundError(str(exc)) from exc
        except Exception as exc:            # noqa: BLE001 — degrade the *accelerator*…
            # A failed accelerator (e.g. no GPU/driver) shouldn't cost the model:
            # retry the same YOLO on CPU. If even that fails, raise clearly — there's
            # no MobileNet-SSD fallback anymore (removed in 0.25.0).
            if self.accelerator != "cpu":
                _log.warning("%s on %s unavailable (%s) — retrying on CPU",
                             self.model, self.accelerator, exc)
                reason = f"requested {self.accelerator} — failed: {exc}"
                self.accelerator = "cpu"
                resolved = yolo.resolve_variant(self.model, "cpu")   # back to FP32
                self._yolo = yolo.load_net(resolved, "cpu")
                self._yolo.fallback_reason = reason
            else:
                raise RuntimeError(
                    f"YOLO model {self.model!r} failed to load ({exc}). Check the "
                    "ONNX file in d20app/models/ (see models/README.md to export it)."
                ) from exc
        self._yolo_size = yolo.input_size(resolved)
        self._resolved_model = resolved
        return self._yolo

    def effective_accelerator(self):
        """``(effective, reason)`` for what the net ACTUALLY runs on (#90) —
        ``("tensorrt", "")`` on a clean engine load, or e.g. ``("cpu",
        "requested tensorrt — unavailable: …")`` after a fallback. ``(None, "")``
        until the net has loaded. Fallbacks must be visible, never re-labelled."""
        net = self._yolo
        if net is None:
            return None, ""
        return (getattr(net, "effective_accelerator", self.accelerator),
                getattr(net, "fallback_reason", ""))

    def _ensure_cap(self):
        import cv2

        _quiet_cv2_logs(cv2)
        if self._cap is None or not self._cap.isOpened():
            # A local USB camera opens by index; a network stream is forced onto
            # FFmpeg so RTSP auth behaves like VLC (see _open_capture).
            cap = _open_capture(self.source)
            if not cap.isOpened():
                cap.release()
                idx = parse_local_index(self.source)
                if idx is not None:
                    raise CameraError(
                        f"could not open USB camera {idx} on this PC — is it "
                        "plugged in and not in use by another app?"
                    )
                raise CameraError(
                    f"could not open the camera stream "
                    f"{mask_credentials(self.source)} — check the URL, and the "
                    "username/password if the camera needs a login"
                )
            self._cap = cap
        return self._cap

    def _crop(self, frame):
        if not self.roi:
            return frame
        x, y, w, h = self.roi
        return frame[y:y + h, x:x + w]

    def _adjust(self, frame):
        """Apply this detector's image adjustments (no-op returns the input)."""
        return apply_image_adjustments(
            frame, gamma=self.gamma, brightness=self.brightness,
            contrast=self.contrast, saturation=self.saturation)

    # -- inference -----------------------------------------------------------
    def _run_net(self, img, floor: float, size: int | None = None) -> list:
        """One YOLO forward pass on an already-cropped/tiled BGR ``img``.

        Returns ``[(label, score, (x1, y1, x2, y2))]`` in ``img`` pixel coords.
        ``size`` overrides the YOLO input size (used by the larger-input locator net);
        otherwise the model's native size is used.
        """
        from . import yolo

        net = self._ensure_net()
        return yolo.detect_boxes(net, img, floor, size=size or self._yolo_size)

    def _detect_boxes(self, frame, floor: float) -> list:
        """Single full-frame detection on the ROI crop — the fast treat path."""
        return self._run_net(self._crop(frame), floor)

    # -- locator path: higher effective resolution for the still-cat scan -----
    _TILE_GRID = {"off": 1, "2x2": 2, "3x3": 3, "4x4": 4}

    def _tiling_grid(self) -> int:
        return self._TILE_GRID.get(str(self.cat_scan_tiling or "off"), 1)

    def _locator_net(self):
        """A larger-input YOLO runner for the locator scan, or ``None`` to use the
        camera's normal net. Loaded once; missing/failed export → ``None`` (the scan
        then leans on tiling at the native size)."""
        if self.cat_scan_imgsz <= 0:
            return None
        variant = f"{self.model}_{self.cat_scan_imgsz}"
        from . import yolo
        if variant not in yolo.MODELS:
            return None
        if not self._locator_tried:
            self._locator_tried = True
            try:
                self._locator_runner = yolo.load_net(variant, self.accelerator)
                self._locator_size = yolo.input_size(variant)
                self._locator_model = variant
            except Exception as exc:        # noqa: BLE001 — degrade to tiling, don't crash
                _log.warning("locator model %s unavailable (%s) — using %s + tiling",
                             variant, exc, self.model)
        return self._locator_runner

    def _detect_locator(self, frame, floor: float) -> list:
        """Higher-resolution detection for the periodic still-cat scan.

        Tiling (Option B) splits the ROI crop into an overlapping grid so a small
        cat fills more of the net's input; an optional larger YOLO input (Option A)
        stacks on top. Both default to off → identical to :meth:`_detect_boxes`.
        """
        from . import yolo

        cropped = self._crop(frame)
        runner = self._locator_net()        # larger YOLO net, or None
        loc_size = self._locator_size

        def single(img):
            if runner is not None:        # larger-input YOLO export (Option A)
                return yolo.detect_boxes(runner, img, floor, size=loc_size)
            return self._run_net(img, floor)   # camera's own model at native size

        grid = self._tiling_grid()
        if grid <= 1:
            return single(cropped)

        h, w = cropped.shape[:2]
        bw, bh = w / grid, h / grid
        ow, oh = bw * self.cat_scan_tile_overlap, bh * self.cat_scan_tile_overlap
        collected = []
        for gy in range(grid):
            for gx in range(grid):
                x0 = int(max(0, gx * bw - ow))
                y0 = int(max(0, gy * bh - oh))
                x1 = int(min(w, (gx + 1) * bw + ow))
                y1 = int(min(h, (gy + 1) * bh + oh))
                tile = cropped[y0:y1, x0:x1]
                if tile.size == 0:
                    continue
                for label, score, (a, b, c, d) in single(tile):
                    collected.append((label, score, (a + x0, b + y0, c + x0, d + y0)))
        return yolo.merge_nms(collected, floor)

    # -- targeted boost: zoom + the heaviest available model (0.42.0) ----------
    def set_boost_hint(self, box, seconds: float) -> None:
        """Aim forced scans at ``box`` (ROI-crop coords) for ``seconds``.

        Used when an unconfirmed lead (a VLM "yes") names a *place*: the boost
        shouldn't just look harder everywhere, it should look hardest there."""
        self._boost_hint = (tuple(int(v) for v in box),
                            time.monotonic() + max(0.0, float(seconds)))

    def boost_hint(self):
        """The live targeted-boost box (ROI-crop coords), or None if expired."""
        hint = self._boost_hint
        if hint and time.monotonic() < hint[1]:
            return hint[0]
        return None

    def _boost_net(self):
        """The heaviest available model for targeted boosts, or ``None`` to use
        the camera's own net. Picked per #70 (heavier model beats bigger input),
        loaded once, and degrades with a logged warning — never crashes a scan."""
        if not self._boost_tried:
            self._boost_tried = True
            from . import yolo
            variant = yolo.boost_variant(self.model, self.accelerator)
            if variant:
                try:
                    self._boost_runner = yolo.load_net(variant, self.accelerator)
                    self._boost_size = yolo.input_size(variant)
                    self._boost_model = variant
                    _log.info("targeted boost runs %s (camera model: %s)",
                              variant, self.model)
                except Exception as exc:   # noqa: BLE001 — degrade to the camera's net
                    _log.warning("boost model %s unavailable (%s) — targeted "
                                 "boost uses %s", variant, exc, self.model)
        return self._boost_runner

    def _detect_hint(self, cropped, hint_box, floor: float, boxes: list) -> list:
        """Zoom-scan a full-res crop around ``hint_box`` and merge the mapped-back
        results into ``boxes``.

        The crop makes the suspected cat large in the net's input — the
        resolution lever #70 proved out (tiling/zoom), where raw >640 input
        measured worse — and the pass runs on the heaviest model on disk."""
        from . import yolo
        from .escalation import map_box_to_frame, square_crop_box

        h, w = cropped.shape[:2]
        cx1, cy1, cx2, cy2 = square_crop_box(hint_box, (w, h))
        crop = cropped[cy1:cy2, cx1:cx2]
        if crop.size == 0:
            return boxes
        runner = self._boost_net()
        if runner is not None:
            found = yolo.detect_boxes(runner, crop, floor, size=self._boost_size)
        else:
            found = self._run_net(crop, floor)
        if not found:
            return boxes
        mapped = [(lab, s, map_box_to_frame(b, (cx1, cy1, cx2, cy2)))
                  for lab, s, b in found]
        return yolo.merge_nms(list(boxes) + mapped, floor)

    @staticmethod
    def _best(boxes, label: str) -> float:
        return max((s for lab, s, _ in boxes if lab == label), default=0.0)

    def best_box(self, label: str):
        """``(score, (x1, y1, x2, y2))`` for the strongest ``label`` box in the

        last analysed frame, or ``None`` if that label wasn't seen. Used to log
        *where* a cat was, in the same coords as the snapshot boxes.
        """
        best = None
        for lab, score, box in self._last_boxes:
            if lab == label and (best is None or score > best[0]):
                best = (score, box)
        return best

    def detect_in_frame(self, frame) -> bool:
        """Return True if a person is present in ``frame`` above the threshold."""
        boxes = self._detect_boxes(frame, floor=min(0.3, self.confidence))
        return self._best(boxes, "person") >= self.confidence

    # -- per-class gating -----------------------------------------------------
    def _floor_for(self, label: str) -> float:
        """The confidence a box of ``label`` must clear to count/draw: person →
        ``confidence`` (treats), a locator class (the cat) → ``cat_confidence``,
        anything else → ``label_floor`` (just naming a mover in the log)."""
        if label == "person":
            return self.confidence
        if label in self.locator_classes:
            return self.cat_confidence
        return self.label_floor

    def _is_locator_hit(self, label: str, score: float) -> bool:
        """True if ``label`` counts as 'the cat' for the locator at its threshold."""
        return label in self.locator_classes and score >= self.cat_confidence

    # Colours (BGR) for drawing boxes: person = green, cat/locator = orange, other = grey.
    _BOX_COLORS = {"person": (80, 220, 80), "cat": (40, 170, 240)}

    def _draw_boxes(self, img, boxes) -> None:
        """Draw labelled detection boxes onto ``img`` in place.

        Only boxes that actually count are drawn — a person at the trigger
        threshold, other classes at the label floor — so a corrupt frame's
        low-confidence guesses don't litter the image.
        """
        import cv2

        for label, score, (x1, y1, x2, y2) in boxes:
            if score < self._floor_for(label):
                continue
            # A locator class (the cat — maybe a "dog" standing in for it) draws in
            # the cat colour even if it isn't literally "cat".
            color = self._BOX_COLORS.get(label) or (
                self._BOX_COLORS["cat"] if label in self.locator_classes else (160, 160, 160))
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            tag = f"{label} {score:.2f}"
            ty = max(y1 - 6, 12)
            cv2.putText(img, tag, (x1, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    def annotated_jpeg(self) -> bytes | None:
        """JPEG of the last analysed frame with labelled detection boxes drawn."""
        import cv2

        if self._last_frame is None:
            return None
        img = self._last_frame.copy()
        self._draw_boxes(img, self._last_boxes)
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
        return buf.tobytes() if ok else None

    def detect_image(self, frame):
        """Run detection on one BGR frame (no camera, no motion gate).

        Applies this detector's image adjustments + ROI exactly as the live path
        does, then returns ``(annotated_jpeg_bytes, detections)`` where each
        detection is ``{"label", "score", "box": [x1,y1,x2,y2]}`` for boxes that
        clear their floor (person → ``confidence``, others → ``label_floor``).
        Powers the GUI's "Test detection" tool.
        """
        import cv2

        adjusted = self._adjust(frame)
        floor = min(self.label_floor, self.confidence, self.cat_confidence)
        # Use the locator (tiled / larger-input) path when either is enabled, so the
        # tester measures exactly what the still-cat scan would do.
        use_locator = self._tiling_grid() > 1 or self.cat_scan_imgsz > 0
        boxes = self._detect_locator(adjusted, floor) if use_locator else self._detect_boxes(adjusted, floor)
        img = self._crop(adjusted).copy()
        self._draw_boxes(img, boxes)
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        dets = [
            {"label": label, "score": round(float(score), 3),
             "box": [int(v) for v in box]}
            for label, score, box in boxes
            if score >= self._floor_for(label)
        ]
        dets.sort(key=lambda d: -d["score"])
        return (buf.tobytes() if ok else None), dets

    # How long detection boxes stay drawn on the live feed after their last
    # refresh — long enough to ride out the gap between analysed frames, short
    # enough that boxes don't linger over an empty frame once a subject leaves.
    _LIVE_BOX_TTL = 1.5

    # In smooth mode, how long the published frame may go un-refreshed while the
    # grab thread reports an error before we surface that error to the loop (so a
    # camera that dies after a good frame is still noticed, not silently frozen).
    _GRAB_STALE_SECONDS = 2.0

    def last_confirmed(self):
        """The newest cat position this detector *confirmed* (any locator-class
        box ≥ ``cat_confidence``, or a track-fusion confirm), or ``None``.

        ``{"box", "label", "score", "age_s"}`` — the drawing-level track behind
        the "last known location" overlay (0.42.1): seconds-fresh, updated on
        every confirming detection, and entirely separate from the throttled
        sightings log."""
        with self._live_lock:
            rec = self._cat_live_confirmed
        if rec is None:
            return None
        box, label, score, ts = rec
        return {"box": box, "label": label, "score": score,
                "age_s": max(0.0, time.time() - ts)}

    @staticmethod
    def _age_label(seconds: float) -> str:
        """'42s' / '7m' / '3h' — compact age for on-feed overlay labels."""
        if seconds < 90:
            return f"{int(seconds)}s"
        if seconds < 5400:
            return f"{int(round(seconds / 60))}m"
        return f"{int(round(seconds / 3600))}h"

    def live_jpeg(self, trail: bool = False, last_known=None) -> bytes | None:
        """JPEG of the most recent frame, with recent detection boxes overlaid.

        Drives the live GUI stream. Returns the latest read frame (not just the
        last *analysed* one, so the feed stays smooth at scan rate); boxes are
        drawn only if the net refreshed them within :data:`_LIVE_BOX_TTL`, so a
        person who has left doesn't leave a box hanging. ``trail=True`` (0.39.0)
        composites the cat trail as a live overlay — the recency ribbon + route
        line update in place as the cat moves — with detection boxes drawn on
        top; no trail this episode simply shows the plain feed. ``None`` until a
        frame has been read. Thread-safe: the web request thread calls this
        while the detection loop writes the underlying frame.

        ``last_known`` (0.42.0) — ``{"box", "label", "age_s"}`` for the camera's
        newest *confirmed* cat position: drawn as a grey box with an age label,
        so the feed always answers "where was she last?" even when nothing is
        detected now. Suppressed while a fresh locator-class box is on screen
        (0.42.1) — a live cat box IS the answer; a grey echo of it would read
        as a second cat. A live targeted-boost hint is drawn too (orange,
        "checking (lead)"), so you can see where the boost is aiming. Live
        detection boxes go on top.
        """
        import cv2

        with self._live_lock:
            frame = self._live_frame
            boxes = self._last_boxes
            fresh = (time.monotonic() - self._live_boxes_at) <= self._LIVE_BOX_TTL
        if frame is None:
            return None
        # Crop to the ROI here (the buffer holds the raw frame); the copy also
        # detaches us from the grab thread swapping the buffer underneath.
        img = self._crop(frame).copy()
        if trail:
            overlaid = self._trail.render(img)
            if overlaid is not None:
                img = overlaid
        cat_on_screen = fresh and any(
            self._is_locator_hit(lab, s) for lab, s, _ in boxes)
        if last_known and not cat_on_screen:
            x1, y1, x2, y2 = (int(v) for v in last_known["box"])
            cv2.rectangle(img, (x1, y1), (x2, y2), (170, 170, 170), 1)
            cv2.putText(img,
                        f"last known: {last_known.get('label', 'cat')} "
                        f"({self._age_label(last_known.get('age_s', 0))} ago)",
                        (x1, max(14, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (190, 190, 190), 1, cv2.LINE_AA)
        hint = self.boost_hint()
        if hint is not None:
            x1, y1, x2, y2 = hint
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 165, 255), 1)
            cv2.putText(img, "checking (lead)", (x1, max(14, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 165, 255), 1,
                        cv2.LINE_AA)
        if fresh:
            self._draw_boxes(img, boxes)
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
        return buf.tobytes() if ok else None

    def live_version(self) -> int:
        """Monotonic counter bumped on each new frame / box update.

        The stream uses it to avoid re-encoding an unchanged frame and to serve
        at the rate frames actually arrive (camera rate in smooth mode).
        """
        with self._live_lock:
            return self._live_version

    def cat_present(self) -> bool:
        """True if a locator-class box (the cat) cleared its threshold in the last
        fresh frame.

        Drives the GUI's flashing "Show cat" button. Bounded by ``_LIVE_BOX_TTL``
        so it clears shortly after the cat leaves (or detection stops refreshing).
        """
        with self._live_lock:
            fresh = (time.monotonic() - self._live_boxes_at) <= self._LIVE_BOX_TTL
            boxes = self._last_boxes
        return fresh and any(
            self._is_locator_hit(lab, score) for lab, score, _ in boxes
        )

    def cat_last_seen(self) -> float:
        """Monotonic time the net last detected a cat (0.0 if never).

        Unlike :meth:`cat_present` (a short, box-TTL window), this isn't bounded —
        the loop decides the freshness window so a *still* cat re-found by a periodic
        forced scan keeps the GUI's flash on between scans.
        """
        with self._live_lock:
            return self._cat_last_seen

    def motion_hint_boxes(self) -> list:
        """WHERE motion last happened: ``[(x1, y1, x2, y2)]`` in the same
        (ROI-cropped) frame coords the detections use, largest blob first.

        The escalation ladder's "look here" hints (#66). Lock-free: the prefilter
        rebinds (never mutates) ``last_blobs`` each update, so a reader on another
        thread sees a stale-but-consistent list — the same pattern as the loop's
        ``_detectors`` dict.
        """
        return self._motion.last_blobs

    def motion_hint_ts(self) -> float:
        """Wall-clock time of the last motion-blob update (0.0 if never)."""
        return self._motion.last_blobs_ts

    # Ring-buffer tuning (#68): frames spaced ≥ _RING_SPACING s, ≤ _RING_MAX_DIM px.
    _RING_FRAMES = 8
    _RING_SPACING = 1.0
    _RING_MAX_DIM = 480

    def _push_ring(self, frame) -> None:
        import cv2

        now = time.time()
        if now - self._ring_last < self._RING_SPACING:
            return
        h, w = frame.shape[:2]
        scale = max(h, w) / float(self._RING_MAX_DIM)
        small = frame if scale <= 1.0 else cv2.resize(
            frame, (int(round(w / scale)), int(round(h / scale))))
        with self._live_lock:
            self._ring.append((now, small.copy()))
            self._ring_last = now

    def recent_frames(self, n: int | None = None) -> list:
        """The last ≤n ring-buffer frames, oldest first: ``[(ts, bgr_copy)]``.

        The temporal VLM mosaic's raw material (#68) — a few seconds of history at
        reduced resolution, copies so callers can't race the worker."""
        with self._live_lock:
            items = list(self._ring)
        if n:
            items = items[-n:]
        return [(ts, f.copy()) for ts, f in items]

    # Temporal-fusion weak floor: locator-class boxes at [_FUSE_FLOOR, cat_confidence)
    # are invisible everywhere except the fuser. Below ~0.2 the models false-fire
    # (#70's measured detection floor was 0.25; 0.2 keeps a little headroom for the
    # genuinely-weak-but-real hits fusion exists to accumulate).
    _FUSE_FLOOR = 0.2

    # Trail person-exclusion freshness (0.41.0): person boxes older than this stop
    # masking the trail. Short on purpose — long enough to ride out a per-frame
    # detection flicker, short enough that a spot a person just left is trail-able
    # again quickly. Honest limit: during the cooldown detection-pause the net is
    # skipped, so no fresh boxes exist and a person moving through the pause can
    # still stamp (erase_recent() cleans up what detection sees when it resumes).
    _PERSON_EXCLUDE_SECS = 3.0

    def take_fused_hit(self):
        """Pop the latest temporal-fusion confirmation (or None). The loop claims it
        exactly once and records it as an ordinary sighting (source "track")."""
        with self._live_lock:
            fused, self._fused_hit = self._fused_hit, None
        return fused

    # Still-scan frame averaging: burst cap, stillness-probe width, and the mean
    # gray diff (on the probe) above which the scene counts as moving. Averaging
    # a still scene drops sensor noise ~√N — real signal recovery, the opposite
    # of SR's synthesized detail (#69) — but averaging a mover would smear it,
    # so any movement mid-burst aborts to the single sharp frame.
    _AVG_MAX_FRAMES = 8
    _AVG_PROBE_W = 160
    _AVG_DIFF_MAX = 4.0

    def _read_still_average(self, cap, first):
        """Average ``cat_scan_frames`` back-to-back frames for the still-cat scan.

        Reads up to ``cat_scan_frames - 1`` extra frames from ``cap`` and returns
        the per-pixel mean when every extra frame matches ``first`` (cheap
        downscaled-gray diff ≤ ``_AVG_DIFF_MAX``). A read failure, a size change,
        or any movement mid-burst returns ``first`` unchanged. Sync path only —
        in smooth mode the grab thread is the capture's sole reader. The scan
        path is not latency-sensitive (a sleeping cat waits); the treat path
        never calls this.
        """
        import cv2
        import numpy as np

        n = self.cat_scan_frames
        if n <= 1:
            return first
        h, w = first.shape[:2]
        pw = self._AVG_PROBE_W
        ph = max(1, round(h * pw / float(w)))
        probe0 = cv2.cvtColor(cv2.resize(first, (pw, ph)), cv2.COLOR_BGR2GRAY)
        acc = first.astype(np.float32)
        got = 1
        for _ in range(n - 1):
            ok, f = cap.read()
            if not ok or f is None or f.shape != first.shape:
                return first
            probe = cv2.cvtColor(cv2.resize(f, (pw, ph)), cv2.COLOR_BGR2GRAY)
            if float(cv2.absdiff(probe, probe0).mean()) > self._AVG_DIFF_MAX:
                return first        # something moved — keep the sharp frame
            acc += f.astype(np.float32)
            got += 1
        return (acc / got).astype(np.uint8)

    def trail_endpoint(self):
        """The cat trail's newest silhouette (frame coords + interior flag), or
        None. An interior endpoint means the last movement ended in-view — the
        ladder's strongest hint, and the basis of "probable location" (#67)."""
        return self._trail.endpoint()

    def trail_jpeg(self, frame_bgr):
        """The recency-coloured trail composited over ``frame_bgr`` as JPEG bytes,
        or None when there's no trail this episode."""
        import cv2

        img = self._trail.render(frame_bgr)
        if img is None:
            return None
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
        return buf.tobytes() if ok else None

    def latest_frame(self):
        """A copy of the most recent raw frame (camera-native, pre-ROI), or None.

        The on-demand escalation ladder inspects this; the copy detaches the caller
        from the capture thread swapping the buffer (same as :meth:`live_jpeg`).
        """
        with self._live_lock:
            frame = self._live_frame
        return None if frame is None else frame.copy()

    def _publish_frame(self, frame) -> None:
        """Store the latest raw frame for the live feed and bump the version."""
        with self._live_lock:
            self._live_frame = frame
            self.frame_size = (frame.shape[1], frame.shape[0])
            self._live_version += 1
            self._live_published_at = time.monotonic()

    # -- smooth feed: a dedicated capture thread ----------------------------
    def _grab_loop(self) -> None:
        """Continuously read the camera so the feed runs at camera rate.

        Active only in smooth mode, where it is the **sole** reader of the
        capture (``read_and_detect`` then samples the published buffer instead of
        reading the camera itself). Handles its own reconnect with the same
        empty-read tolerance as the synchronous path, surfacing a persistent
        failure via ``_grab_error`` so the loop can report it.
        """
        while not self._grab_stop.is_set():
            try:
                cap = self._ensure_cap()
            except CameraError as exc:
                with self._live_lock:
                    self._grab_error = str(exc)
                self._grab_stop.wait(1.0)
                continue
            ok, frame = cap.read()
            if self._grab_stop.is_set():
                break                       # stop requested during the blocking read
            if not ok or frame is None:
                self._grab_fails += 1       # own counter — not shared with the sync path
                if self._grab_fails >= 3:
                    with self._live_lock:
                        self._grab_error = (
                            f"lost the camera stream {mask_credentials(self.source)} "
                            "(no frames received)"
                        )
                    self._cap = None        # force a reconnect next iteration
                self._grab_stop.wait(0.1)
                continue
            self._grab_fails = 0
            if self._grab_error:
                with self._live_lock:
                    self._grab_error = ""
            self._publish_frame(frame)

    def _start_grab(self) -> None:
        """Spawn the grab thread (loop thread only). Resets its retry/error state."""
        self._grab_stop.clear()
        self._grab_fails = 0
        with self._live_lock:
            self._grab_error = ""
        self._grab_thread = threading.Thread(
            target=self._grab_loop, name="cam-grab", daemon=True
        )
        self._grab_thread.start()

    def _apply_smooth(self, desired: bool) -> None:
        """Start or stop the grab thread. **Loop thread only**, so the capture is

        never read by two threads at once (the web thread merely sets the desired
        flag and lets the loop reconcile it here).
        """
        if desired and not self.smooth_feed:
            self._start_grab()
            self.smooth_feed = True
            _log.info("Smooth live feed ON (dedicated capture thread)")
        elif not desired and self.smooth_feed:
            self._grab_stop.set()
            thread = self._grab_thread
            if thread is not None:
                thread.join(timeout=5)
                if thread.is_alive():
                    # The grabber is wedged in a blocking read (stalled camera).
                    # Keep it as the live reader rather than risk two readers, and
                    # **clear the stop** so it doesn't silently exit when its read
                    # returns (which would freeze the feed with no reader). The
                    # user can toggle off again once the camera unwedges.
                    self._grab_stop.clear()
                    self._smooth_desired = True
                    _log.warning("Smooth-feed grab thread didn't stop — staying smooth")
                    return
            self._grab_thread = None
            self._read_fails = 0
            self.smooth_feed = False
            _log.info("Smooth live feed OFF")

    def read_and_detect(self, detect: bool = True, force: bool = False) -> FrameOutcome:
        """Grab one frame, apply the motion pre-filter, then classify it.

        Returns a :class:`FrameOutcome`. ``motion`` is False when nothing moved
        (or a frame couldn't be read); when motion is seen, ``person`` says
        whether a person cleared the threshold and ``labels`` lists the other
        things seen (e.g. ``("cat",)``) so the caller can report *what* moved.
        Raises :class:`CameraError` when the stream is really gone.

        With ``detect=False`` the frame is still read (so the stream stays warm
        and a dead camera is still noticed) and the motion baseline is refreshed,
        but the neural net is **skipped** entirely and a neutral, no-motion
        outcome is returned. The loop uses this to idle cheaply during the
        between-rolls cooldown, when nothing the net sees could trigger anyway.

        With ``force=True`` the net runs **even when nothing moved** (and even when
        ``detect`` is False), returning the real ``person``/``labels`` with
        ``motion=False``. This is the periodic still-cat scan: a sleeping cat makes
        no motion to trip the pre-filter, so the loop forces an occasional look.
        """
        import cv2

        # Reconcile a smooth-mode toggle here, on the loop thread, so starting or
        # stopping the grab thread never races the camera read below.
        if self._smooth_desired != self.smooth_feed:
            self._apply_smooth(self._smooth_desired)

        if self.smooth_feed:
            # Self-heal: if the grabber died (e.g. it unwedged after a stop while
            # the stop event was briefly set), respawn it so the feed never stays
            # frozen with no reader.
            if self._grab_thread is None or not self._grab_thread.is_alive():
                self._start_grab()
            # The grab thread is the sole reader; sample its latest frame.
            with self._live_lock:
                frame = self._live_frame
                err = self._grab_error
                age = time.monotonic() - self._live_published_at
            # Surface a persistent failure even while a stale frame is still held
            # (the camera died after a good frame) — mirrors the sync path's
            # "give up after a run of empty reads" so the loop can report it.
            if frame is None or (err and age >= self._GRAB_STALE_SECONDS):
                if err:
                    raise CameraError(err)
                return FrameOutcome(motion=False, person=False)   # warming up
        else:
            cap = self._ensure_cap()    # raises CameraError if it can't open
            ok, frame = cap.read()
            if not ok or frame is None:
                self._read_fails += 1
                self._cap = None        # force a reconnect next call
                # Tolerate a brief hiccup, but a run of empty reads means the
                # stream is really gone — surface it so the loop can back off.
                if self._read_fails >= 3:
                    raise CameraError(
                        f"lost the camera stream {mask_credentials(self.source)} "
                        "(no frames received)"
                    )
                return FrameOutcome(motion=False, person=False)
            self._read_fails = 0
            if force and self.cat_scan_frames > 1:
                # Still-cat scans average a short burst of frames — noise drops
                # ~√N on a still scene, and the burst aborts to the single frame
                # the moment anything moves. Forced scans are not
                # latency-sensitive; the motion/treat path never does this.
                frame = self._read_still_average(cap, frame)

        # Image adjustments are applied to the frame the net analyses; in the
        # synchronous path the live feed shows the adjusted frame too (publish after
        # adjusting). In smooth mode the grab thread publishes the raw frame, so the
        # feed there stays unadjusted while the net still sees the adjusted pixels.
        frame = self._adjust(frame)
        if not self.smooth_feed:
            # Publish every read frame for the live feed, even when the net is
            # skipped (no motion / cooldown pause), so the feed stays warm.
            self._publish_frame(frame)

        cropped = self._crop(frame)
        gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        moved = self._motion.update(gray)      # keep the baseline fresh even when paused
        # Cat-trail silhouettes + endpoint (#67). Fresh person boxes are blanked
        # out of the stamp (0.41.0): a human arm moves in cat-sized blobs, and
        # the trail is a cat trail. The boxes are the PREVIOUS net run's (the net
        # goes after the trail each frame) — erase_recent() below scrubs the
        # frame or two that slip through when a person first appears.
        exclude = ()
        if self._person_boxes and (time.monotonic() - self._person_boxes_at
                                   <= self._PERSON_EXCLUDE_SECS):
            exclude = self._person_boxes
        self._trail.update(gray, moved, exclude=exclude)
        self._push_ring(cropped)               # temporal-mosaic frame history (#68)
        # Run the net on real motion, when a forced still-cat scan asks for it,
        # or on every frame when the motion gate is off (#96).
        if not force and (not detect or not (moved or not self.motion_gate)):
            return FrameOutcome(motion=False, person=False)

        floor = min(self.label_floor, self.confidence, self.cat_confidence)
        # A forced (still-cat) scan uses the higher-resolution locator path; the
        # motion/treat path stays fast at the native size.
        fused = None
        if force:
            boxes = self._detect_locator(frame, floor)
            hint = self.boost_hint()
            if hint is not None:
                # Targeted boost (0.42.0): also zoom the suspected spot at full
                # resolution with the heaviest available model.
                boxes = self._detect_hint(cropped, hint, floor, boxes)
        elif self.track_fusion:
            # Temporal fusion (0.37.0): decode down to the weak floor in the SAME
            # forward pass (the floor is a post-filter, not extra inference). Weak
            # locator hits feed the fuser only — everything the rest of the app
            # sees (boxes, live feed, labels) is still thresholded at `floor`, so
            # nothing weak leaks out except as a fused, movement-checked confirm.
            raw_boxes = self._detect_boxes(frame, min(floor, self._FUSE_FLOOR))
            boxes = [b for b in raw_boxes if b[1] >= floor]
            self._fusion.frame_size = (cropped.shape[1], cropped.shape[0])
            fused = self._fusion.update(None, [
                (s, b) for lab, s, b in raw_boxes
                if lab in self.locator_classes and s >= self._FUSE_FLOOR])
        else:
            boxes = self._detect_boxes(frame, floor)
        self._last_frame = cropped             # what the net saw (box coords match)
        now = time.monotonic()
        person_boxes = [b for lab, _s, b in boxes if lab == "person"]
        if person_boxes:
            self._person_boxes = person_boxes
            self._person_boxes_at = now
            # The trail stamped this frame BEFORE the net ran — scrub the
            # person's just-made stamps retroactively (0.41.0).
            self._trail.erase_recent(person_boxes)
        cat_seen = any(self._is_locator_hit(lab, score) for lab, score, _ in boxes)
        # Track the newest confirmed cat position for the "last known" overlay
        # (0.42.1): every ≥cat_confidence hit updates it — a drawing-level track,
        # NOT a log write, so the sightings log keeps its throttled cadence. A
        # fused (track) confirmation counts too; a real box outranks it.
        best_cat = None
        for lab, score, box in boxes:
            if self._is_locator_hit(lab, score) and (best_cat is None
                                                     or score > best_cat[2]):
                best_cat = (tuple(box), lab, score)
        if best_cat is None and fused:
            best_cat = (tuple(fused["box"]), "cat", fused["score"])
        with self._live_lock:
            self._last_boxes = boxes           # fresh detections (possibly empty)
            self._live_boxes_at = now
            if cat_seen or fused:
                self._cat_last_seen = now
            if best_cat is not None:
                self._cat_live_confirmed = (*best_cat, time.time())
            if fused:
                self._fused_hit = fused        # claimed (and recorded) by the loop
            self._live_version += 1            # boxes changed → stream re-renders
        person = self._best(boxes, "person") >= self.confidence
        # Identify non-person movers: a locator class (the cat) at cat_confidence,
        # any other mover at label_floor (just to name it in the log) — keeping stray
        # "pottedplant"/"sofa" guesses out.
        labels = tuple(
            label
            for label, score, _ in sorted(boxes, key=lambda b: -b[1])
            if label != "person" and (
                self._is_locator_hit(label, score)
                or (label not in self.locator_classes and score >= self.label_floor))
        )
        # ``motion`` reflects the pre-filter: False on a forced still-cat scan, so
        # the loop knows not to treat a forced look as real movement (it never rolls).
        return FrameOutcome(motion=moved, person=person, labels=labels)

    def release(self) -> None:
        # Stop the grab thread (if any) before releasing the capture it reads.
        self._grab_stop.set()
        thread = self._grab_thread
        still_alive = False
        if thread is not None:
            thread.join(timeout=5)
            still_alive = thread.is_alive()
        self._grab_thread = None
        if still_alive:
            # The grabber is wedged in a blocking read on self._cap; releasing the
            # capture out from under it is cv2/FFmpeg undefined behaviour. It's a
            # daemon thread, so leak the capture — the process reclaims it on exit.
            _log.warning("Smooth-feed grab thread didn't stop — leaking the capture")
            return
        if self._cap is not None:
            self._cap.release()
            self._cap = None
