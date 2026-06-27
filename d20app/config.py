"""Configuration: load/save the single ``config.yaml`` the GUI writes.

The detection loop reads this; the web GUI writes it. Defaults live here so the
app runs out-of-the-box before anything is configured.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from urllib.parse import quote

import yaml

# config.yaml lives at the repo root (one level up from this package).
_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_PKG_DIR)
CONFIG_PATH = os.environ.get("D20_CONFIG", os.path.join(_REPO_ROOT, "config.yaml"))
SOUNDS_DIR = os.path.join(_PKG_DIR, "sounds")


@dataclass
class Config:
    """All user-tunable settings. Mirrors the GUI form fields."""

    # --- Camera ---
    camera_url: str = ""              # RTSP/MJPEG/HTTP stream URL (the active camera)
    camera_name: str = ""             # friendly name (from discovery, for display)
    camera_username: str = ""         # optional; only if the stream needs auth
    camera_password: str = ""
    # Saved cameras the user has added. Each is a full per-camera config dict
    # (see `camera_defaults`): identity (name/url/username/password), roles
    # (roll/track_cats), and its own detection settings (model, confidence, roi,
    # motion, ...). Passwords are kept in plaintext locally (same as camera_password).
    cameras: list = field(default_factory=list)
    # Names of saved cameras to watch simultaneously (multi-camera). Empty = the
    # legacy single active camera (the camera_* fields above). Mirrors speaker_names.
    active_cameras: list = field(default_factory=list)

    # --- Speaker (Google Home / Cast) ---
    speaker_name: str = ""            # legacy single speaker (kept for back-compat)
    speaker_names: list = field(default_factory=list)   # one or more Cast device names

    # --- Sound / speech ---
    sound_file: str = "treat_chime.wav"   # filename within d20app/sounds/
    use_speech: bool = False          # speak a message instead of playing the chime
    speech_text: str = "Give the cat a treat!"   # what to say when use_speech is on

    # --- Game rules (GUI-tunable) ---
    dice_sides: int = 20             # D20, D100, ...
    dc: int = 20                     # treat when roll >= dc (e.g. natural 20)
    cooldown_seconds: int = 600      # frequency interval between rolls

    # --- Detection tuning ---
    detector_model: str = "yolo11n"  # "yolo11n" (better low-light/odd-pose, ~1.4x CPU; default), "yolo11m" (medium, bigger/slower, ~5-18x CPU), or "mobilenet_ssd" (lightest, bundled); falls back to mobilenet_ssd if YOLO can't load
    accelerator: str = "cpu"         # where the YOLO model runs: "cpu" (default), "opencl" (iGPU via OpenCL, no extra deps), or "openvino-gpu"/"openvino-auto" (Intel OpenVINO, needs the optional 'openvino' pkg + Intel GPU drivers); auto-falls back to CPU if a GPU backend can't start
    person_confidence: float = 0.5   # min DNN confidence to count as a person (0.5: clean person/cat split on stills, keeps hard poses ≥0.71)
    confirm_frames: int = 4          # require a person in this many frames in a row (4 guards against a moving cat's transient high-confidence spike)
    detect_size: int = 300           # net input size; 300 = reliable for people (512 = distant cats, heavier)
    scan_fps: float = 10.0           # frames/sec to read from the camera (lower = less CPU)
    smooth_live_feed: bool = False   # dedicated capture thread so the live feed runs at camera rate (decoupled from inference); costs a little extra CPU/bandwidth
    roi: list | None = None          # optional [x, y, w, h] crop of the frame (set in the GUI)
    label_floor: float = 0.55        # min confidence to NAME a non-person mover in the log/snapshot (higher = fewer stray "pottedplant"/"sofa" labels; no effect on treats)

    # --- Motion pre-filter (cheap gate before the neural net runs) ---
    motion_sensitivity: str = "medium"   # "low"|"medium"|"high"|"custom" — GUI preset that drives the three knobs below
    motion_min_area_frac: float = 0.003  # fraction of the frame that must change to count as motion (higher = less sensitive)
    motion_diff_threshold: int = 25      # per-pixel brightness change to count a pixel as moved (higher = less sensitive)
    motion_min_blob_px: int = 14         # reject change regions thinner than this (rejects thin artifact lines)

    # --- CPU saving ---
    pause_during_cooldown: bool = True   # skip the neural net while in the between-rolls cooldown (nothing it sees can trigger anyway); resumes just before the window reopens

    # --- Quiet time (no chimes during this daily window; "" = disabled) ---
    quiet_start: str = ""            # "HH:MM", e.g. "22:00"
    quiet_end: str = ""              # "HH:MM", e.g. "07:00" (may wrap past midnight)

    # --- Casting behaviour ---
    dont_interrupt_playback: bool = False   # skip a treat if media is playing
    keep_speakers_warm: bool = False        # loop a silent clip so the Cast receiver stays loaded and there's no "connecting" chime (holds the speaker active)

    # --- Server ---
    web_port: int = 8080
    file_server_port: int = 8081     # serves the sound to the speaker

    def asdict(self) -> dict:
        return asdict(self)


def speaker_targets(cfg: "Config") -> list:
    """Cast device names to play on: the new list, else the legacy single name."""
    names = [n for n in (cfg.speaker_names or []) if n]
    if not names and cfg.speaker_name:
        names = [cfg.speaker_name]
    return names


# --- Multi-camera: per-camera config dicts ---------------------------------
# Identity + role fields and their literal defaults (a new camera does both roles).
_CAMERA_BASE = {
    "name": "", "url": "", "username": "", "password": "",
    "roll": True, "track_cats": True,
}
# Per-camera detection fields → the global Config attribute that supplies the
# default (so a new/old camera inherits the current global detection settings).
_CAMERA_FROM_CFG = {
    "model": "detector_model", "accelerator": "accelerator",
    "person_confidence": "person_confidence", "confirm_frames": "confirm_frames",
    "detect_size": "detect_size", "scan_fps": "scan_fps", "label_floor": "label_floor",
    "smooth_feed": "smooth_live_feed", "roi": "roi",
    "motion_sensitivity": "motion_sensitivity",
    "motion_min_area_frac": "motion_min_area_frac",
    "motion_diff_threshold": "motion_diff_threshold",
    "motion_min_blob_px": "motion_min_blob_px",
}


def camera_source(url: str, username: str = "", password: str = "") -> str:
    """Inject percent-encoded credentials into an rtsp:// URL given separately.

    Leaves a bare URL (or a ``usb:N`` source, which has no ``://``) unchanged.
    """
    if username and "://" in url and "@" not in url:
        scheme, rest = url.split("://", 1)
        cred = quote(username, safe="")
        if password:
            cred += ":" + quote(password, safe="")
        return f"{scheme}://{cred}@{rest}"
    return url


def camera_defaults(cfg: "Config" | None = None) -> dict:
    """A full camera dict with defaults — identity/roles + global detection settings."""
    cfg = cfg or Config()
    out = dict(_CAMERA_BASE)
    for key, attr in _CAMERA_FROM_CFG.items():
        out[key] = getattr(cfg, attr)
    return out


def coerce_camera(raw: dict, cfg: "Config" | None = None) -> dict:
    """Build a complete camera dict from ``raw``, filling and type-coercing missing
    keys from the defaults (so partial GUI payloads and old saved entries upgrade).
    """
    defaults = camera_defaults(cfg)
    out = {}
    for key, default in defaults.items():
        # ``roi`` is explicitly nullable: a present ``roi: None`` means "whole
        # frame", so don't treat it as missing and inherit the global default.
        present = key in raw and (raw[key] is not None or key == "roi")
        out[key] = _coerce(raw[key], default) if present else default
    return out


def camera_targets(cfg: "Config") -> list:
    """The cameras to watch, each a full spec dict plus a resolved ``source``.

    Uses ``active_cameras`` (multi-camera) when set, else falls back to the single
    legacy active camera. Returns ``[]`` if nothing is configured.
    """
    saved = {c.get("name"): c for c in (cfg.cameras or [])
             if isinstance(c, dict) and c.get("name")}
    names = []
    for n in (cfg.active_cameras or []):       # de-dup: one detector/thread per name
        if n in saved and n not in names:
            names.append(n)
    if names:
        specs = [coerce_camera(saved[n], cfg) for n in names]
    elif cfg.camera_url:
        specs = [coerce_camera({
            "name": cfg.camera_name or cfg.camera_url,
            "url": cfg.camera_url,
            "username": cfg.camera_username,
            "password": cfg.camera_password,
        }, cfg)]
    else:
        return []
    for spec in specs:
        spec["source"] = camera_source(spec["url"], spec["username"], spec["password"])
    return specs


_KNOWN_FIELDS = set(Config().asdict().keys())


def load(path: str = CONFIG_PATH) -> Config:
    """Load config from ``path``; return defaults if it doesn't exist yet."""
    if not os.path.exists(path):
        return Config()
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    # Ignore unknown keys so an old/edited file never crashes startup.
    clean = {k: v for k, v in data.items() if k in _KNOWN_FIELDS}
    return Config(**clean)


def save(cfg: Config, path: str = CONFIG_PATH) -> None:
    """Persist config to ``path`` as YAML."""
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg.asdict(), fh, sort_keys=False, default_flow_style=False)


def update(values: dict, path: str = CONFIG_PATH) -> Config:
    """Merge ``values`` into the saved config, persist, and return it.

    Only known fields are applied; types are coerced to match the dataclass
    defaults so values arriving as strings from an HTML form land correctly.
    """
    cfg = load(path)
    defaults = Config().asdict()
    for key, raw in values.items():
        if key not in _KNOWN_FIELDS:
            continue
        setattr(cfg, key, _coerce(raw, defaults[key]))
    save(cfg, path)
    return cfg


def _coerce(raw, default):
    """Coerce ``raw`` (often a string from a form) to ``default``'s type."""
    if default is None:
        return raw
    if isinstance(default, bool):
        if isinstance(raw, str):
            return raw.strip().lower() in ("1", "true", "yes", "on")
        return bool(raw)
    if isinstance(default, int):
        return int(float(raw))
    if isinstance(default, float):
        return float(raw)
    if isinstance(default, str):
        return "" if raw is None else str(raw)
    return raw
