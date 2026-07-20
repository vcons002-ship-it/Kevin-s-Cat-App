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
    detector_model: str = "yolo11n"  # the benchmark-settled lineup (#70/#71): "yolo11n" (floor, 640 since #80, default), "yolo26m" (lightweight), "yolo26x"/fp16 variants (workhorse, export-only). Dropped models (yolo11m/..., #79) raise a loud, actionable error — no silent worse detector.
    accelerator: str = "auto"        # where the YOLO model runs: "auto" (default, #71: CUDA when it genuinely binds — verified, never a silent CPU fallback — else CPU, logged loudly), "cpu", "opencl" (iGPU via OpenCL), "openvino-gpu"/"openvino-auto" (Intel OpenVINO, optional 'openvino' pkg), or "onnx-cuda" (NVIDIA via optional 'onnxruntime-gpu' CUDA-12 build; raises loudly if CUDA doesn't bind).
    person_confidence: float = 0.5   # min DNN confidence to count as a person (0.5: clean person/cat split on stills, keeps hard poses ≥0.71)
    confirm_frames: int = 4          # require a person in this many frames in a row (4 guards against a moving cat's transient high-confidence spike)
    detect_size: int = 300           # legacy MobileNet-SSD net input size — ignored since 0.25.0 (YOLO uses its exported size). Kept so old configs still load.
    scan_fps: float = 10.0           # frames/sec to read from the camera (lower = less CPU)
    smooth_live_feed: bool = False   # dedicated capture thread so the live feed runs at camera rate (decoupled from inference); costs a little extra CPU/bandwidth
    roi: list | None = None          # optional [x, y, w, h] crop of the frame (set in the GUI)
    label_floor: float = 0.55        # min confidence to NAME a non-person mover in the log/snapshot (higher = fewer stray "pottedplant"/"sofa" labels; no effect on treats)
    cat_confidence: float = 0.5      # min confidence for a locator-class detection (the cat) to count — independent of person_confidence and of label_floor. Gates the track_cats/Show-cat path only; the treat/roll path is untouched.
    locator_classes: list = field(default_factory=lambda: ["cat"])   # COCO classes that count as "the cat" for the locator. Default cat-only. A no-dog household can use ["cat", "dog"] since the model often (confidently) mislabels a cat as a dog and more resolution raises "dog", not "cat".

    # --- Image adjustments (applied to the frame BEFORE the net runs; can rescue a
    # too-dark / washed-out feed). Defaults are all no-ops. Tune them in the GUI's
    # "Test detection" tool against a real photo, then save to a camera. ---
    gamma: float = 1.0               # >1 brightens midtones (lifts shadows), <1 darkens; 1.0 = off
    brightness: int = 0              # add to every pixel (-100..100); 0 = off
    contrast: float = 1.0            # multiply around mid-grey (0.5..2.0); 1.0 = off
    saturation: float = 1.0          # scale colour intensity (0 = greyscale, 1.0 = off, >1 = punchier)

    # --- Per-camera LIVE-detection tiling (#101), independent of the still-scan's.
    live_tiling: str = "off"             # live-detection tiling (#101) — independent of the scan's; default off (multiplies per-frame cost)
    live_tile_overlap: float = 0.2       # overlap for live tiling when on

    # --- Still-cat scan settings (#101/#102): GLOBAL, not per-camera — one
    # coherent "check for still cat" group. The scan gets ONE hard static look, so
    # it wants its own heavy model/tiling regardless of a camera's live settings.
    cat_scan_model: str = ""             # "" = each camera's own live model; else a specific heavy model for the scan (#94)
    cat_scan_confidence: float = 0.0     # cat threshold for the scan; 0 = use each camera's cat_confidence

    # --- "Show me the cat" active find-scan (#92): search on click, don't just
    # jump to the last sighting. Off by default (click keeps today's behavior).
    # Its own full settings (#101): model/tiling/overlap/confidence.
    find_scan: bool = False              # run a real detection pass across the find cameras on click
    find_model: str = ""                 # the model the find pass runs — "" = each camera's own
    find_tiling: str = "3x3"             # find-scan tiling (a thorough look for a still cat)
    find_tile_overlap: float = 0.35      # overlap for find tiling
    find_confidence: float = 0.0         # cat threshold for find; 0 = each camera's cat_confidence
    find_cameras: list = field(default_factory=list)   # cameras to sweep; [] = all watched
    motion_sensitivity: str = "medium"   # "low"|"medium"|"high"|"custom"|"off" — GUI preset that drives the three knobs below ("off" = detect every frame, #96)
    motion_min_area_frac: float = 0.003  # fraction of the frame that must change to count as motion (higher = less sensitive)
    motion_diff_threshold: int = 25      # per-pixel brightness change to count a pixel as moved (higher = less sensitive)
    motion_min_blob_px: int = 14         # reject change regions thinner than this (rejects thin artifact lines)
    motion_hold_seconds: float = 2.0  # after motion fires, keep running the net for this long, refreshed by every further motion frame. Without it the net only ever sees frames that MOVED — the motion-blurred, mid-stride ones — and a cat that pauses mid-walk runs no inference during the pause. Respects the cooldown pause and round-robin (it gates like motion, not like a forced scan). 0 = motion frames only (pre-0.57 behaviour).
    motion_reference_ms: int = 200   # compare each frame against the newest one at least this old IN THE VIDEO, rather than whatever arrived last. RTSP frames queue, so consecutive reads were measured only ~33ms apart in video time even though the loop reads every 200ms — and the motion test is an AREA test, so a cat covers ~6x less of it and slow/distant cats were missed. 0 = previous frame (pre-0.56 behaviour).

    # --- Cat check (still/sleeping cats produce no motion to trigger the net) ---
    cat_scan_interval: float = 30.0  # how often (sec) a cat-tracking camera runs the net even with no motion, to catch a still/sleeping cat: >0 = every N seconds (default 30), 0 = always on (net every frame; most CPU), <0 = off (motion only, legacy behaviour)

    # --- Locator resolution (the still-cat scan only; the fast treat path is untouched).
    # A sleeping cat shrunk into a 640 frame can be too small to detect at all; tiling
    # and/or a larger input give the locator scan more effective resolution. ---
    cat_scan_tiling: str = "3x3"     # "off" | "2x2" | "3x3" | "4x4" — split the frame into an overlapping grid, detect per tile, merge. Default 3x3 per the full benchmark (#70): 26x@3x3 = 91%/0% FP; 4x4 adds ~1 cat at 1.7x the latency (and high-overlap 4x4 buys recall with false positives).
    cat_scan_tile_overlap: float = 0.35  # fraction each tile overlaps its neighbour so a cat on a seam still lands whole in one tile. Benchmark-settled (#70): 3x3 is best at 0.35 (2x2 prefers 0.20).
    cat_scan_imgsz: int = 0          # LEGACY, no effect since 0.40.0 (#79): the >640 "locator input" hypothesis was benchmarked and rejected — 960/1280 measured WORSE than 640 (#70); tiling is the resolution lever. Kept so old configs still load; any value falls back to native + tiling.
    cat_scan_frames: int = 3         # still-cat scan only: average this many back-to-back frames before the net (sensor noise drops ~sqrt(N) on a still scene — helps dim/noisy frames; any motion mid-burst falls back to the single frame). 1 = off; capped at 8. The fast treat path never averages.
    track_fusion: bool = True        # temporal score fusion: a string of WEAK cat detections (below cat_confidence) that chain smoothly and actually MOVE across the frame is confirmed as one sighting (source "track") — the recall-raising mirror of confirm_frames. Pure YOLO evidence; the movement requirement is the decoy guard (a cushion never travels). Off = judge every frame alone (pre-0.37.0 behaviour).
    # --- Follow mode: feeds are assigned by sighting recency and HELD until a cat
    # turns up outside the pair. Both knobs below are safeguards for overlapping
    # camera views (a cat near a shared edge tripping two cameras); 0 = off, which
    # is the plain recency behaviour.
    swap_confirm_count: int = 0              # detections a NEW camera needs before it may take a feed. 0/1 = the first detection swaps. Raise it to ignore a transient pass through an overlap.
    camera_reuse_cooldown_seconds: float = 0.0   # how long a camera that just lost its feed is barred from taking one again. Breaks a two-camera ping-pong, which the confirm count alone can't (both are genuinely detecting).
    fusion_debug: bool = False       # troubleshooting only (#110): write one structured record per fusion decision to fusion_events.jsonl — per-hit score/label/box, net travel vs the threshold, and whether a strong box coexisted. OFF by default and adds NOTHING to the activity feed; it exists to tell a genuine weak-cat recovery apart from a redundant misfire.

    # --- CPU saving ---
    pause_during_cooldown: bool = True   # skip the neural net while in the between-rolls cooldown (nothing it sees can trigger anyway); resumes just before the window reopens

    # --- Round-robin (cap CPU when watching many cameras: run only a few at a time,
    # rotating sets on an interval — high camera counts at the CPU of a few). ---
    round_robin: bool = False        # rotate detection across the watched cameras instead of all-at-once
    round_robin_size: int = 2        # how many cameras detect at once (the rest rest, releasing their capture)
    round_robin_interval: float = 15.0   # seconds each set is watched before rotating to the next

    # --- Quiet time (no chimes during this daily window; "" = disabled) ---
    quiet_start: str = ""            # "HH:MM", e.g. "22:00"
    quiet_end: str = ""              # "HH:MM", e.g. "07:00" (may wrap past midnight)

    # --- Casting behaviour ---
    dont_interrupt_playback: bool = False   # skip a treat if media is playing
    keep_speakers_warm: bool = False        # loop a silent clip so the Cast receiver stays loaded and there's no "connecting" chime (holds the speaker active)

    # --- VLM (moondream) cat-presence tester ---
    moondream_api_key: str = ""      # paste once in the GUI; authenticates the one-time local weight download AND cloud inference. Never logged; masked in the API. Env MOONDREAM_API_KEY also works (headless).
    vlm_escalation: bool = False     # allow the "find the cat" escalation ladder (zoom crops -> YOLO -> VLM detect/query) to run against LIVE cameras on demand. Off by default; the fast treat path never touches it, and Test-tool escalation on uploaded frames works regardless of this flag.

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
    "always_watch": False,        # exclude from round-robin — this camera never rests
    # Semantic zones (#68): named rectangles drawn on the full preview frame (like
    # the ROI) — [{name, box: [x, y, w, h], exit: bool}]. Sightings inside a zone
    # get its name ("the couch"); zones marked exit (doorways) refine the trail's
    # "may have left the view" check.
    "zones": [],
}
# Per-camera detection fields → the global Config attribute that supplies the
# default (so a new/old camera inherits the current global detection settings).
_CAMERA_FROM_CFG = {
    "model": "detector_model", "accelerator": "accelerator",
    "person_confidence": "person_confidence", "confirm_frames": "confirm_frames",
    "scan_fps": "scan_fps", "label_floor": "label_floor",
    "smooth_feed": "smooth_live_feed", "roi": "roi",
    "gamma": "gamma", "brightness": "brightness",
    "contrast": "contrast", "saturation": "saturation",
    "cat_confidence": "cat_confidence", "locator_classes": "locator_classes",
    # Still-scan settings moved to a GLOBAL group (#101/#102) — no longer
    # per-camera. Old camera dicts that still carry cat_scan_* keep loading
    # (coerce_camera ignores unknown extras); the loop reads the global values.
    "live_tiling": "live_tiling",
    "live_tile_overlap": "live_tile_overlap",
    "track_fusion": "track_fusion",
    "fusion_debug": "fusion_debug",
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
        # Pass the field's current value so a blank/bad numeric keeps it (H1/M3),
        # while its *type* is still taken from the dataclass default.
        setattr(cfg, key, _coerce(raw, defaults[key], getattr(cfg, key)))
    save(cfg, path)
    return cfg


def _coerce(raw, default, current=None):
    """Coerce ``raw`` (often a string from a form) to ``default``'s type.

    ``raw`` frequently arrives from an auto-saving HTML control, so a numeric
    field may be blank (``""``) or ``null`` when the user clears its input. Treat
    those — and any unparseable value — as "keep this field's current value"
    instead of raising: an unguarded ``int(float(""))`` / ``float(None)`` would
    500 the whole ``POST /api/config`` (audit H1). ``current`` is the value to
    keep on a blank/bad input (the field's existing saved value, passed by
    ``update()``); when it's ``None`` — no prior value, e.g. building a fresh
    camera dict — we fall back to ``default``. Keeping the current value stops a
    spurious blank auto-save from clobbering a real setting (audit M3).
    """
    if default is None:
        return raw
    if isinstance(default, bool):
        if isinstance(raw, str):
            return raw.strip().lower() in ("1", "true", "yes", "on")
        return bool(raw)
    if isinstance(default, (int, float)):   # bool handled above (bool is an int)
        keep = current if current is not None else default
        if raw is None or (isinstance(raw, str) and raw.strip() == ""):
            return keep
        try:
            return int(float(raw)) if isinstance(default, int) else float(raw)
        except (TypeError, ValueError):
            return keep
    if isinstance(default, str):
        return "" if raw is None else str(raw)
    return raw
