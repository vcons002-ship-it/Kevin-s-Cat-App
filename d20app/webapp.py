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
"""

from __future__ import annotations

import os

from flask import Flask, jsonify, request, send_from_directory
from werkzeug.utils import secure_filename

from . import config as config_mod
from . import discovery
from .loop import DetectionLoop

ALLOWED_SOUND_EXT = {".wav", ".mp3", ".ogg", ".m4a", ".aac"}


def create_app(loop: DetectionLoop | None = None) -> Flask:
    app = Flask(__name__)
    app.config["loop"] = loop or DetectionLoop()

    # -- page ---------------------------------------------------------------
    @app.get("/")
    def index():
        return send_from_directory(app.template_folder, "index.html")

    # -- discovery ----------------------------------------------------------
    @app.get("/api/speakers")
    def api_speakers():
        return jsonify(discovery.discover_speakers())

    @app.get("/api/cameras")
    def api_cameras():
        return jsonify(discovery.discover_cameras())

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
        cfg = config_mod.load().asdict()
        cfg.pop("camera_password", None)        # never echo the password back
        return jsonify(cfg)

    @app.post("/api/config")
    def api_set_config():
        values = request.get_json(silent=True) or {}
        # Don't overwrite a stored password with an empty form field.
        if not values.get("camera_password"):
            values.pop("camera_password", None)
        cfg = config_mod.update(values)
        out = cfg.asdict()
        out.pop("camera_password", None)
        return jsonify(out)

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
