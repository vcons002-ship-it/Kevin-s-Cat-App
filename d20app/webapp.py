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
  GET  /api/stream     -> live MJPEG feed of the detector's annotated frames
"""

from __future__ import annotations

import os
import time

from flask import Flask, Response, jsonify, request, send_from_directory
from werkzeug.utils import secure_filename

from . import __version__
from . import config as config_mod
from . import discovery
from .detector import grab_frame_jpeg
from .loop import DetectionLoop, _camera_source

ALLOWED_SOUND_EXT = {".wav", ".mp3", ".ogg", ".m4a", ".aac"}


def _mask_cameras(cameras) -> list:
    """Saved cameras without raw passwords — a ``has_password`` flag instead."""
    out = []
    for c in cameras or []:
        if not isinstance(c, dict):
            continue
        out.append({
            "name": c.get("name", ""),
            "url": c.get("url", ""),
            "username": c.get("username", ""),
            "has_password": bool(c.get("password")),
        })
    return out


def _public_config(cfg_dict: dict) -> dict:
    """Strip every stored password before sending config to the browser."""
    cfg_dict.pop("camera_password", None)
    cfg_dict["cameras"] = _mask_cameras(cfg_dict.get("cameras"))
    return cfg_dict


def create_app(loop: DetectionLoop | None = None) -> Flask:
    app = Flask(__name__)
    app.config["loop"] = loop or DetectionLoop()

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
        if not cfg.camera_url:
            return jsonify({"error": "No camera configured yet."}), 400
        jpeg = grab_frame_jpeg(_camera_source(cfg))
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

        def frames():
            # One JPEG per part; the browser renders this directly in an <img>.
            # Capped at ~10 fps — the loop only reads at scan_fps anyway, so this
            # adds an encode per served frame and nothing when no one's watching.
            while loop.is_running():
                jpeg = loop.live_jpeg()
                if jpeg is not None:
                    yield (b"--frame\r\nContent-Type: image/jpeg\r\n"
                           b"Content-Length: " + str(len(jpeg)).encode() + b"\r\n\r\n"
                           + jpeg + b"\r\n")
                time.sleep(0.1)

        return Response(frames(),
                        mimetype="multipart/x-mixed-replace; boundary=frame")

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
        return jsonify(_public_config(config_mod.load().asdict()))

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
        return jsonify(_public_config(cfg.asdict()))

    # -- saved cameras (a dropdown of manually-added cameras + credentials) --
    @app.get("/api/cameras/saved")
    def api_cameras_saved():
        return jsonify(_mask_cameras(config_mod.load().cameras))

    @app.post("/api/cameras/saved")
    def api_cameras_save():
        data = request.get_json(silent=True) or {}
        name = (data.get("name") or "").strip()
        url = (data.get("url") or "").strip()
        if not name or not url:
            return jsonify({"error": "A camera needs a name and a stream URL."}), 400
        cfg = config_mod.load()
        cams = [c for c in (cfg.cameras or []) if isinstance(c, dict)]
        entry = {
            "name": name,
            "url": url,
            "username": (data.get("username") or "").strip(),
            "password": data.get("password") or "",
        }
        existing = next((c for c in cams if c.get("name") == name), None)
        if existing is not None:
            # A blank password on re-save keeps the previously-stored one.
            if not entry["password"]:
                entry["password"] = existing.get("password", "")
            cams[cams.index(existing)] = entry
        else:
            cams.append(entry)
        config_mod.update({"cameras": cams})
        return jsonify(_mask_cameras(cams))

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
        return jsonify(_public_config(config_mod.load().asdict()))

    @app.post("/api/cameras/saved/delete")
    def api_cameras_delete():
        name = ((request.get_json(silent=True) or {}).get("name") or "").strip()
        cfg = config_mod.load()
        cams = [c for c in (cfg.cameras or [])
                if isinstance(c, dict) and c.get("name") != name]
        config_mod.update({"cameras": cams})
        return jsonify(_mask_cameras(cams))

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

    return app
