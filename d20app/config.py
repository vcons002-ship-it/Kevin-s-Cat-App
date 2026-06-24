"""Configuration: load/save the single ``config.yaml`` the GUI writes.

The detection loop reads this; the web GUI writes it. Defaults live here so the
app runs out-of-the-box before anything is configured.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field

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
    camera_url: str = ""              # RTSP/MJPEG/HTTP stream URL
    camera_name: str = ""             # friendly name (from discovery, for display)
    camera_username: str = ""         # optional; only if the stream needs auth
    camera_password: str = ""

    # --- Speaker (Google Home / Cast) ---
    speaker_name: str = ""            # Cast device friendly name

    # --- Sound ---
    sound_file: str = "treat_chime.wav"   # filename within d20app/sounds/

    # --- Game rules (GUI-tunable) ---
    dice_sides: int = 20             # D20, D100, ...
    dc: int = 20                     # treat when roll >= dc (e.g. natural 20)
    cooldown_seconds: int = 600      # frequency interval between rolls

    # --- Detection tuning ---
    person_confidence: float = 0.5   # min DNN confidence to count as a person
    confirm_frames: int = 3          # require a person in this many frames in a row
    detect_size: int = 512           # net input size; 512 spots distant cats (300 = lighter CPU)
    scan_fps: float = 10.0           # frames/sec to read from the camera (lower = less CPU)
    roi: list | None = None          # optional [x, y, w, h] crop of the frame (set in the GUI)

    # --- Quiet time (no chimes during this daily window; "" = disabled) ---
    quiet_start: str = ""            # "HH:MM", e.g. "22:00"
    quiet_end: str = ""              # "HH:MM", e.g. "07:00" (may wrap past midnight)

    # --- Casting behaviour ---
    dont_interrupt_playback: bool = False   # skip a treat if media is playing

    # --- Server ---
    web_port: int = 8080
    file_server_port: int = 8081     # serves the sound to the speaker

    def asdict(self) -> dict:
        return asdict(self)


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
