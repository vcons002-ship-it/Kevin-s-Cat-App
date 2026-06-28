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

    @app.post("/api/cats/boost")
    def api_cats_boost():
        # "Show cat" jumped the live feed to this camera — run its detector
        # continuously for a short window so the feed boxes the cat while you look.
        name = (request.get_json(silent=True) or {}).get("camera")
        ok = app.config["loop"].boost_detection(name) if name else False
        return jsonify({"ok": bool(ok)})

    return app
