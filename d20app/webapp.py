"""Flask web GUI: the single page where Kevin configures and runs everything.

Serves the config page plus JSON endpoints:

  GET  /api/speakers   -> auto-detected Cast devices
  GET  /api/cameras    -> auto-detected ONVIF cameras
  GET  /api/sounds     -> sound files available to cast
  POST /api/sounds     -> upload a custom sound
  GET  /api/config     -> current saved settings
  POST /api/config     -> save settings
  POST /api/test       -> force a treat sound on the chosen speaker
  POST /api/start      -> start the detection loop
  POST /api/stop       -> stop the detection loop
  GET  /api/status     -> live loop status (running, last roll, counts)
  GET  /api/stream     -> live MJPEG feed (?camera=<name> selects which camera)
  POST /api/live/smooth-> toggle the smooth (decoupled-capture) live feed
  POST /api/cameras/active -> set which saved cameras are watched at once
  GET  /api/cats       -> cat sightings (last seen, today's count, recent)
"""

from __future__ import annotations

import base64
import os
import threading
import time
import uuid
from collections import OrderedDict
from html import escape as esc

from flask import Flask, Response, jsonify, request, send_from_directory
from werkzeug.utils import secure_filename

from . import __version__
from . import config as config_mod
from . import discovery
from .detector import PersonDetector, grab_frame_jpeg, sample_video_frames
from .loop import DetectionLoop, _camera_source

ALLOWED_SOUND_EXT = {".wav", ".mp3", ".ogg", ".m4a", ".aac"}

# --- "Test detection" tool: upload a photo/video, run the net with adjustable
# settings, draw boxes. Frames are kept briefly in memory keyed by a session id so
# the GUI can re-run on each slider tweak without re-uploading. ---------------
_TEST_SESSIONS: "OrderedDict[str, list]" = OrderedDict()   # id -> [BGR frame, ...]
_TEST_SESSIONS_LOCK = threading.Lock()
_TEST_MAX_SESSIONS = 4
_TEST_VIDEO_FRAMES = 8
_test_detectors: dict = {}              # (model, accelerator) -> reusable PersonDetector
_test_detect_lock = threading.Lock()    # serialise test detection + detector reuse
_TEST_MAX_UPLOAD = 256 * 1024 * 1024    # 256 MB cap on a test upload


def _thumb_data_url(frame, width: int = 200) -> str:
    import cv2

    h, w = frame.shape[:2]
    if w > width:
        frame = cv2.resize(frame, (width, max(1, int(h * width / w))))
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
    if not ok:
        return ""
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


def _decode_test_upload(data: bytes, filename: str) -> list:
    """Return BGR frames from uploaded image *or* video bytes ([] if unreadable)."""
    import cv2
    import numpy as np

    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is not None:
        return [img]
    # Not an image — try it as a video (cv2 needs a real path).
    import tempfile

    suffix = os.path.splitext(filename or "")[1] or ".mp4"
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    try:
        tmp.write(data)
        tmp.flush()
        tmp.close()
        return sample_video_frames(tmp.name, _TEST_VIDEO_FRAMES)
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass


def _run_test_detection(frame, settings: dict):
    """Run :meth:`PersonDetector.detect_image` with override settings.

    Reuses one detector per (model, accelerator) so the net isn't reloaded on every
    slider tweak; the cheap per-call knobs (confidence/size/floor/ROI/adjustments)
    are set per request under a lock.
    """
    model = settings.get("model") or "yolo11n"
    accel = settings.get("accelerator") or "cpu"

    def _f(key, default):
        try:
            return float(settings.get(key, default))
        except (TypeError, ValueError):
            return float(default)

    with _test_detect_lock:
        key = (model, accel)
        det = _test_detectors.get(key)
        if det is None:
            det = PersonDetector(source="__test__", model=model, accelerator=accel)
            _test_detectors[key] = det
        det.confidence = _f("person_confidence", 0.5)
        det.detect_size = int(_f("detect_size", 300)) or 300
        det.label_floor = _f("label_floor", 0.55)
        det.cat_confidence = _f("cat_confidence", 0.5)
        lc = settings.get("locator_classes")
        det.locator_classes = tuple(lc) if isinstance(lc, (list, tuple)) and lc else ("cat",)
        roi = settings.get("roi")
        det.roi = list(roi) if (isinstance(roi, (list, tuple)) and len(roi) == 4) else None
        det.gamma = _f("gamma", 1.0)
        det.brightness = int(_f("brightness", 0))
        det.contrast = _f("contrast", 1.0)
        det.saturation = _f("saturation", 1.0)
        det.cat_scan_tiling = settings.get("tiling") or "off"
        det.cat_scan_tile_overlap = _f("tile_overlap", 0.2)
        det.cat_scan_imgsz = int(_f("imgsz", 0))
        det._locator_tried = False        # re-resolve the larger-input net on setting change
        det._locator_runner = None
        t0 = time.perf_counter()
        annotated, dets = det.detect_image(frame)
        return annotated, dets, round((time.perf_counter() - t0) * 1000.0, 1)


# --- "Benchmark this image": sweep models × tiling on one frame, emit a shareable
# self-contained HTML report + an optional XLSX (needs openpyxl). ---------------
_BENCHMARKS: "OrderedDict[str, dict]" = OrderedDict()     # id -> {html, xlsx, xlsx_error}
_BENCH_LOCK = threading.Lock()
_BENCH_MAX_REPORTS = 8
_BENCH_TILINGS = ["off", "2x2", "3x3", "4x4"]
_BENCH_MAX_RUNS = 24            # cap the matrix so one request can't run forever


def _benchmark_models():
    """Model names worth offering in a sweep: the YOLO variants whose ONNX is
    present, plus MobileNet. (A 960 export only appears if it was produced.)"""
    from . import yolo

    out = [v for v in ("yolo11n", "yolo11m", "yolo11m_960")
           if v in yolo.MODELS and os.path.exists(yolo.model_path(v))]
    out.append("mobilenet_ssd")
    return out


def _model_native_size(model: str) -> int:
    from . import yolo

    if model in yolo.MODELS:
        return yolo.input_size(model)
    if "@" in model:
        try:
            return int(model.split("@", 1)[1])
        except ValueError:
            pass
    return 300


def _bench_thumb(jpeg: bytes, width: int = 240):
    """(data-url, raw-jpeg-bytes) thumbnail of an annotated frame for the reports."""
    import cv2
    import numpy as np

    img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return "", b""
    h, w = img.shape[:2]
    if w > width:
        img = cv2.resize(img, (width, max(1, int(h * width / w))))
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 72])
    raw = buf.tobytes() if ok else b""
    return ("data:image/jpeg;base64," + base64.b64encode(raw).decode("ascii")), raw


def _run_benchmark(frame, models: list, tilings: list, cat_threshold: float,
                   accelerator: str) -> list:
    """Sweep ``models × tilings`` on one frame. Each run reuses ``_run_test_detection``
    (no parallel detection path) with a very low floor + cat/dog locator so the
    **raw** best cat/dog score is captured; ``detected`` marks it against
    ``cat_threshold``. Sorted by combined (cat-or-dog) score, best first."""
    runs = []
    for model in models:
        for tiling in tilings:
            settings = {
                "model": model, "tiling": tiling, "tile_overlap": 0.2,
                "accelerator": accelerator, "person_confidence": 0.5,
                "cat_confidence": 0.01, "label_floor": 0.01,
                "locator_classes": ["cat", "dog"],
            }
            annotated, dets, ms = _run_test_detection(frame, settings)
            best_cat = max((d["score"] for d in dets if d["label"] == "cat"), default=0.0)
            best_dog = max((d["score"] for d in dets if d["label"] == "dog"), default=0.0)
            combined = max(best_cat, best_dog)
            thumb, raw = _bench_thumb(annotated)
            runs.append({
                "model": model, "size": _model_native_size(model), "tiling": tiling,
                "tile_overlap": 0.2, "accelerator": accelerator,
                "cat_score": round(best_cat, 3), "dog_score": round(best_dog, 3),
                "combined_score": round(combined, 3),
                "detected": bool(combined >= cat_threshold),
                "boxes": sum(1 for d in dets if d["label"] in ("cat", "dog")),
                "inference_ms": ms, "thumb": thumb, "_raw": raw,
            })
    runs.sort(key=lambda r: -r["combined_score"])
    return runs


def _score_color(score: float) -> str:
    """Red→amber→green for a 0..1 score (report cell background)."""
    s = max(0.0, min(1.0, float(score)))
    r, g = (220, int(60 + 160 * s)) if s < 0.5 else (int(220 - 320 * (s - 0.5)), 200)
    return f"rgb({r},{g},70)"


def _benchmark_html(runs: list, meta: dict) -> str:
    """A self-contained HTML report — every thumbnail inlined, no external assets,
    so it renders for a remote viewer and emails cleanly."""
    rows = []
    for r in runs:
        hit = "✓" if r["detected"] else "·"
        rows.append(
            f"<tr><td><img src='{r['thumb']}'></td>"
            f"<td>{esc(r['model'])}</td><td>{r['size']}</td><td>{esc(r['tiling'])}</td>"
            f"<td style='background:{_score_color(r['combined_score'])}'>"
            f"<b>{r['combined_score']:.2f}</b></td>"
            f"<td>{r['cat_score']:.2f}</td><td>{r['dog_score']:.2f}</td>"
            f"<td>{hit}</td><td>{r['inference_ms']:.0f} ms</td></tr>")
    fixed = (f"cat threshold <b>{meta['cat_threshold']:.2f}</b> · accelerator "
             f"<b>{esc(meta['accelerator'])}</b> · tile overlap 0.2 · person 0.50 · "
             f"locator [cat, dog]")
    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<title>Cat detection benchmark</title><style>"
        "body{font-family:system-ui,Arial,sans-serif;margin:24px;color:#111}"
        "table{border-collapse:collapse;width:100%}th,td{border:1px solid #ddd;"
        "padding:6px 8px;text-align:center;font-size:14px}th{background:#f3f3f3}"
        "img{height:90px;border-radius:4px;display:block}"
        ".fixed{background:#f7f7f9;border:1px solid #e3e3e8;border-radius:8px;"
        "padding:10px 14px;margin:10px 0;font-size:14px}</style></head><body>"
        f"<h2>🐱 Cat detection benchmark — {esc(meta['image'])}</h2>"
        f"<p class='muted'>{esc(meta['ts'])} · {len(runs)} runs</p>"
        f"<div class='fixed'><b>Settings held fixed for every run:</b> {fixed}</div>"
        "<table><tr><th>frame</th><th>model</th><th>size</th><th>tiling</th>"
        "<th>cat-or-dog</th><th>cat</th><th>dog</th><th>found?</th><th>time</th></tr>"
        + "".join(rows) + "</table>"
        "<p style='color:#777;font-size:13px'>“cat-or-dog” is the combined locator "
        "score (the number that predicts locator reliability with cat+dog); “found?” "
        "marks it against the cat threshold above. Time is the total for the run "
        "(summed across tiles).</p></body></html>")


def _benchmark_xlsx(runs: list, meta: dict) -> bytes:
    """XLSX with embedded thumbnails. Raises if openpyxl isn't installed."""
    try:
        import openpyxl
        from openpyxl.drawing.image import Image as XLImage
        from openpyxl.styles import Font
    except Exception as exc:        # noqa: BLE001 — optional dependency
        raise RuntimeError(
            "XLSX export needs the 'openpyxl' package — re-run setup (say yes to "
            "spreadsheet export) or: pip install openpyxl") from exc
    import io

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "benchmark"
    ws["A1"] = f"Cat detection benchmark — {meta['image']} ({meta['ts']})"
    ws["A1"].font = Font(bold=True)
    ws["A2"] = (f"Fixed: cat threshold {meta['cat_threshold']:.2f} · accelerator "
                f"{meta['accelerator']} · tile overlap 0.2 · person 0.50 · locator [cat, dog]")
    headers = ["frame", "model", "size", "tiling", "cat-or-dog", "cat", "dog",
               "found?", "time (ms)"]
    ws.append([])
    ws.append(headers)
    for c in ws[ws.max_row]:
        c.font = Font(bold=True)
    head_row = ws.max_row
    for i, r in enumerate(runs):
        row = head_row + 1 + i
        ws.cell(row, 2, r["model"]); ws.cell(row, 3, r["size"])
        ws.cell(row, 4, r["tiling"]); ws.cell(row, 5, r["combined_score"])
        ws.cell(row, 6, r["cat_score"]); ws.cell(row, 7, r["dog_score"])
        ws.cell(row, 8, "yes" if r["detected"] else "no")
        ws.cell(row, 9, r["inference_ms"])
        ws.row_dimensions[row].height = 70
        if r.get("_raw"):
            img = XLImage(io.BytesIO(r["_raw"]))
            img.width, img.height = 120, 90
            ws.add_image(img, f"A{row}")
    ws.column_dimensions["A"].width = 18
    ws.column_dimensions["B"].width = 16
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


def _mask_cameras(cameras, cfg=None) -> list:
    """Saved cameras with full per-camera settings, raw passwords stripped.

    Each entry is coerced to a complete camera dict (missing settings filled from
    the global defaults) so the GUI editor always has every field; the password is
    replaced by a ``has_password`` flag.
    """
    out = []
    for c in cameras or []:
        if not isinstance(c, dict):
            continue
        full = config_mod.coerce_camera(c, cfg)
        full.pop("password", None)
        full["has_password"] = bool(c.get("password"))
        out.append(full)
    return out


def _public_config(cfg) -> dict:
    """Config as a browser-safe dict: strip passwords, expand the camera list."""
    d = cfg.asdict()
    d.pop("camera_password", None)
    d["cameras"] = _mask_cameras(d.get("cameras"), cfg)
    return d


def create_app(loop: DetectionLoop | None = None) -> Flask:
    app = Flask(__name__)
    app.config["loop"] = loop or DetectionLoop()
    app.config["MAX_CONTENT_LENGTH"] = _TEST_MAX_UPLOAD   # cap test-video uploads

    # -- page ---------------------------------------------------------------
    @app.get("/")
    def index():
        return send_from_directory(app.template_folder, "index.html")

    @app.get("/api/version")
    def api_version():
        return jsonify({"version": __version__})

    # -- detection snapshots (annotated images shown in the activity log) ----
    @app.get("/snapshots/<path:name>")
    def snapshot(name):
        directory = app.config["loop"].snapshots.directory
        if not os.path.exists(os.path.join(directory, name)):
            return jsonify({"error": "not found"}), 404
        return send_from_directory(directory, name)

    # -- live preview frame (for the region-of-interest picker) -------------
    @app.get("/api/preview")
    def api_preview():
        cfg = config_mod.load()
        name = request.args.get("camera")
        if name:
            cam = next((c for c in (cfg.cameras or [])
                        if isinstance(c, dict) and c.get("name") == name), None)
            if cam is None:
                return jsonify({"error": "camera not found"}), 404
            source = config_mod.camera_source(
                cam.get("url", ""), cam.get("username", ""), cam.get("password", ""))
        elif cfg.camera_url:
            source = _camera_source(cfg)
        else:
            return jsonify({"error": "No camera configured yet."}), 400
        jpeg = grab_frame_jpeg(source)
        if jpeg is None:
            return jsonify({"error": "Couldn't grab a frame from the camera."}), 502
        return Response(jpeg, mimetype="image/jpeg")

    # -- live detection feed (MJPEG of what the running loop sees) -----------
    @app.get("/api/stream")
    def api_stream():
        loop = app.config["loop"]
        if not loop.is_running():
            return jsonify(
                {"error": "Start watching to see the live detection feed."}
            ), 409

        name = request.args.get("camera")     # which camera's feed (default: streamed one)

        def frames():
            # One JPEG per part; the browser renders this directly in an <img>.
            # Encode only when the frame/box version changes, so we never re-send
            # an unchanged frame and the feed runs at whatever rate frames arrive
            # — the loop's scan rate normally, or camera rate with the smooth feed.
            # The 0.03 s poll is the only ceiling (~30 fps); cheap when idle.
            last_ver = -1
            while loop.is_running():
                loop.note_viewing(name)   # keep this camera awake under round-robin
                ver = loop.live_version(name)
                if ver != last_ver:
                    jpeg = loop.live_jpeg(name)
                    if jpeg is not None:
                        last_ver = ver
                        yield (b"--frame\r\nContent-Type: image/jpeg\r\n"
                               b"Content-Length: " + str(len(jpeg)).encode() + b"\r\n\r\n"
                               + jpeg + b"\r\n")
                time.sleep(0.03)

        return Response(frames(),
                        mimetype="multipart/x-mixed-replace; boundary=frame")

    @app.post("/api/live/smooth")
    def api_live_smooth():
        # Persist the choice and, if watching, apply it live (the loop reconciles
        # it on its next frame). Off → on costs a little extra CPU/bandwidth.
        on = bool((request.get_json(silent=True) or {}).get("on"))
        config_mod.update({"smooth_live_feed": on})
        app.config["loop"].set_smooth(on)
        return jsonify({"ok": True, "smooth_live_feed": on})

    # -- discovery ----------------------------------------------------------
    @app.get("/api/speakers")
    def api_speakers():
        return jsonify(discovery.discover_speakers())

    @app.get("/api/cameras")
    def api_cameras():
        return jsonify(discovery.discover_cameras())

    @app.get("/api/cameras/local")
    def api_cameras_local():
        # USB/built-in cameras on the machine running the app.
        return jsonify(discovery.probe_local_cameras())

    # -- sounds -------------------------------------------------------------
    @app.get("/api/sounds")
    def api_sounds():
        files = sorted(
            f for f in os.listdir(config_mod.SOUNDS_DIR)
            if os.path.splitext(f)[1].lower() in ALLOWED_SOUND_EXT
        )
        return jsonify(files)

    @app.post("/api/sounds")
    def api_upload_sound():
        if "file" not in request.files:
            return jsonify({"error": "no file uploaded"}), 400
        f = request.files["file"]
        name = secure_filename(f.filename or "")
        if not name or os.path.splitext(name)[1].lower() not in ALLOWED_SOUND_EXT:
            return jsonify({"error": "unsupported file type"}), 400
        f.save(os.path.join(config_mod.SOUNDS_DIR, name))
        return jsonify({"saved": name})

    # -- config -------------------------------------------------------------
    @app.get("/api/config")
    def api_get_config():
        return jsonify(_public_config(config_mod.load()))

    @app.post("/api/config")
    def api_set_config():
        values = request.get_json(silent=True) or {}
        # Don't overwrite a stored password with an empty form field.
        if not values.get("camera_password"):
            values.pop("camera_password", None)
        # The saved-camera store is managed only via the /api/cameras/saved
        # endpoints, so the main settings save can't clobber it (or its passwords).
        values.pop("cameras", None)
        cfg = config_mod.update(values)
        return jsonify(_public_config(cfg))

    # -- saved cameras (full per-camera config + credentials) ----------------
    @app.get("/api/cameras/saved")
    def api_cameras_saved():
        cfg = config_mod.load()
        return jsonify(_mask_cameras(cfg.cameras, cfg))

    @app.post("/api/cameras/saved")
    def api_cameras_save():
        data = request.get_json(silent=True) or {}
        name = (data.get("name") or "").strip()
        url = (data.get("url") or "").strip()
        if not name or not url:
            return jsonify({"error": "A camera needs a name and a stream URL."}), 400
        cfg = config_mod.load()
        cams = [c for c in (cfg.cameras or []) if isinstance(c, dict)]
        existing = next((c for c in cams if c.get("name") == name), None)
        # Start from the existing entry (so unspecified settings persist), overlay
        # the incoming fields, then coerce to a complete per-camera dict.
        merged = dict(existing or {})
        merged.update(data)
        merged["name"], merged["url"] = name, url
        entry = config_mod.coerce_camera(merged, cfg)
        # A blank password on re-save keeps the previously-stored one.
        if not (data.get("password") or "").strip() and existing is not None:
            entry["password"] = existing.get("password", "")
        if existing is not None:
            cams[cams.index(existing)] = entry
        else:
            cams.append(entry)
        config_mod.update({"cameras": cams})
        return jsonify(_mask_cameras(cams, cfg))

    @app.post("/api/cameras/saved/select")
    def api_cameras_select():
        name = ((request.get_json(silent=True) or {}).get("name") or "").strip()
        cfg = config_mod.load()
        cam = next((c for c in (cfg.cameras or [])
                    if isinstance(c, dict) and c.get("name") == name), None)
        if cam is None:
            return jsonify({"error": "camera not found"}), 404
        config_mod.update({
            "camera_name": cam.get("name", ""),
            "camera_url": cam.get("url", ""),
            "camera_username": cam.get("username", ""),
            "camera_password": cam.get("password", ""),
        })
        return jsonify(_public_config(config_mod.load()))

    @app.post("/api/cameras/saved/delete")
    def api_cameras_delete():
        name = ((request.get_json(silent=True) or {}).get("name") or "").strip()
        cfg = config_mod.load()
        cams = [c for c in (cfg.cameras or [])
                if isinstance(c, dict) and c.get("name") != name]
        # Also drop it from the watched set if present.
        active = [n for n in (cfg.active_cameras or []) if n != name]
        config_mod.update({"cameras": cams, "active_cameras": active})
        return jsonify(_mask_cameras(cams, cfg))

    @app.post("/api/cameras/active")
    def api_cameras_active():
        """Set which saved cameras are watched simultaneously (multi-camera)."""
        names = (request.get_json(silent=True) or {}).get("names") or []
        cfg = config_mod.load()
        valid = {c.get("name") for c in (cfg.cameras or []) if isinstance(c, dict)}
        active = [n for n in names if n in valid]
        config_mod.update({"active_cameras": active})
        return jsonify({"active_cameras": active})

    # -- control ------------------------------------------------------------
    @app.post("/api/test")
    def api_test():
        try:
            app.config["loop"].test_cast()
            return jsonify({"ok": True})
        except Exception as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400

    @app.post("/api/start")
    def api_start():
        started = app.config["loop"].start()
        return jsonify({"running": True, "started": started})

    @app.post("/api/stop")
    def api_stop():
        app.config["loop"].stop()
        return jsonify({"running": False})

    @app.get("/api/status")
    def api_status():
        loop = app.config["loop"]
        s = loop.status
        return jsonify(
            {
                "running": loop.is_running(),
                "last_error": s.last_error,
                "last_roll": s.last_roll,
                "last_roll_at": s.last_roll_at,
                "rolls": s.rolls,
                "treats": s.treats,
                "cameras": loop.cam_status(),   # per-camera connected/error + roles
            }
        )

    # -- activity log -------------------------------------------------------
    @app.get("/api/log")
    def api_log():
        limit = request.args.get("limit", default=200, type=int)
        return jsonify(app.config["loop"].activity.entries(limit=limit))

    @app.post("/api/log/clear")
    def api_log_clear():
        app.config["loop"].activity.clear()
        return jsonify({"ok": True})

    # -- cat sightings ("show cat") -----------------------------------------
    @app.get("/api/cats")
    def api_cats():
        limit = request.args.get("limit", default=20, type=int)
        loop = app.config["loop"]
        return jsonify({
            "last": loop.cats.last(),
            "today": loop.cats.today_count(),
            "present": loop.cat_present(),     # cat on camera right now → flash the button
            "cameras": loop.cats_present_cameras(),   # cameras seeing a cat now → Show-cat rotation
            "recent": loop.cats.recent(limit=limit),
        })

    @app.post("/api/cats/clear")
    def api_cats_clear():
        app.config["loop"].cats.clear()
        return jsonify({"ok": True})

    # -- "Test detection" tool (upload a photo/video, tune settings, draw boxes) --
    @app.post("/api/test/upload")
    def api_test_upload():
        if "file" not in request.files:
            return jsonify({"error": "no file uploaded"}), 400
        data = request.files["file"].read()
        if not data:
            return jsonify({"error": "empty file"}), 400
        frames = _decode_test_upload(data, request.files["file"].filename or "")
        if not frames:
            return jsonify({"error": "Couldn't read that as an image or video."}), 400
        sid = uuid.uuid4().hex
        with _TEST_SESSIONS_LOCK:
            _TEST_SESSIONS[sid] = frames
            while len(_TEST_SESSIONS) > _TEST_MAX_SESSIONS:
                _TEST_SESSIONS.popitem(last=False)     # drop the oldest session
        h, w = frames[0].shape[:2]
        return jsonify({
            "id": sid, "count": len(frames), "width": int(w), "height": int(h),
            "kind": "video" if len(frames) > 1 else "image",
            "thumbs": [_thumb_data_url(fr) for fr in frames],
        })

    @app.post("/api/test/detect")
    def api_test_detect():
        body = request.get_json(silent=True) or {}
        with _TEST_SESSIONS_LOCK:
            frames = _TEST_SESSIONS.get(body.get("id"))
        if not frames:
            return jsonify({"error": "Upload expired — pick the file again."}), 404
        try:
            idx = int(body.get("frame_index", 0))
        except (TypeError, ValueError):
            idx = 0
        idx = max(0, min(idx, len(frames) - 1))
        annotated, dets, ms = _run_test_detection(frames[idx], body.get("settings") or {})
        if annotated is None:
            return jsonify({"error": "Detection failed on that frame."}), 500
        return jsonify({
            "annotated": "data:image/jpeg;base64," + base64.b64encode(annotated).decode("ascii"),
            "detections": dets, "frame_index": idx, "inference_ms": ms,
        })

    @app.get("/api/test/benchmark/models")
    def api_benchmark_models():
        return jsonify({"models": _benchmark_models(), "tilings": _BENCH_TILINGS})

    @app.post("/api/test/benchmark")
    def api_test_benchmark():
        body = request.get_json(silent=True) or {}
        with _TEST_SESSIONS_LOCK:
            frames = _TEST_SESSIONS.get(body.get("id"))
        if not frames:
            return jsonify({"error": "Upload expired — pick the file again."}), 404
        try:
            idx = max(0, min(int(body.get("frame_index", 0)), len(frames) - 1))
        except (TypeError, ValueError):
            idx = 0
        models = [m for m in (body.get("models") or _benchmark_models()) if m]
        tilings = [t for t in (body.get("tilings") or _BENCH_TILINGS) if t]
        if not models or not tilings:
            return jsonify({"error": "Pick at least one model and tiling."}), 400
        if len(models) * len(tilings) > _BENCH_MAX_RUNS:
            return jsonify({"error": f"Too many runs ({len(models) * len(tilings)}); "
                            f"trim to ≤ {_BENCH_MAX_RUNS}."}), 400
        try:
            cat_threshold = float(body.get("cat_threshold",
                                           config_mod.load().cat_confidence))
        except (TypeError, ValueError):
            cat_threshold = 0.5
        accelerator = body.get("accelerator") or "cpu"
        runs = _run_benchmark(frames[idx], models, tilings, cat_threshold, accelerator)
        h, w = frames[idx].shape[:2]
        meta = {"cat_threshold": cat_threshold, "accelerator": accelerator,
                "image": body.get("name") or "uploaded image",
                "ts": time.strftime("%Y-%m-%d %H:%M"), "width": int(w), "height": int(h)}
        html = _benchmark_html(runs, meta)
        try:
            xlsx = _benchmark_xlsx(runs, meta)
            xlsx_error = None
        except Exception as exc:        # noqa: BLE001 — optional dep / report failure
            xlsx, xlsx_error = None, str(exc)
        rid = uuid.uuid4().hex
        with _BENCH_LOCK:
            _BENCHMARKS[rid] = {"html": html, "xlsx": xlsx}
            while len(_BENCHMARKS) > _BENCH_MAX_REPORTS:
                _BENCHMARKS.popitem(last=False)
        public = [{k: v for k, v in r.items() if k != "_raw"} for r in runs]
        return jsonify({
            "id": rid, "runs": public, "meta": meta,
            "html_url": f"/api/test/benchmark/{rid}.html",
            "xlsx_url": (f"/api/test/benchmark/{rid}.xlsx" if xlsx else None),
            "xlsx_error": xlsx_error,
        })

    @app.get("/api/test/benchmark/<rid>.html")
    def api_benchmark_html(rid):
        rep = _BENCHMARKS.get(rid)
        if not rep:
            return jsonify({"error": "report expired"}), 404
        return Response(rep["html"], mimetype="text/html")

    @app.get("/api/test/benchmark/<rid>.xlsx")
    def api_benchmark_xlsx(rid):
        rep = _BENCHMARKS.get(rid)
        if not rep or not rep.get("xlsx"):
            return jsonify({"error": "no XLSX (install openpyxl) or report expired"}), 404
        return Response(
            rep["xlsx"],
            mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": f'attachment; filename="benchmark_{rid}.xlsx"'})

    @app.post("/api/cats/boost")
    def api_cats_boost():
        # "Show cat" jumped the live feed to this camera — run its detector
        # continuously for a short window so the feed boxes the cat while you look.
        name = (request.get_json(silent=True) or {}).get("camera")
        ok = app.config["loop"].boost_detection(name) if name else False
        return jsonify({"ok": bool(ok)})

    return app
