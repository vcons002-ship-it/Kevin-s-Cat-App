# Kevin's Cat App — notes for Claude

A D20 "treat roller": a background loop watches a camera for a **person** (cats are
ignored), rolls a die on each allowed detection, and on a winning roll plays a chime
(or a spoken message) on a Google Home and/or this PC's speakers — the cue that it's
OK to give the cat a treat. A Flask single-page GUI configures and runs it. CPU-only,
no Docker, no cloud, no account.

## Run / test
- Python 3.11+. Virtualenv at `./venv`.
- Tests: `./venv/bin/python -m pytest -q` (currently ~95 tests; keep them green).
- Launch: `./venv/bin/python run.py` → prints a `http://<lan-ip>:8080` GUI URL.
- Setup: `setup.sh` (Linux/apt) or `setup.ps1` / `setup.bat` (Windows), `start.bat`
  to launch on Windows.

## Layout
- `d20app/detector.py` — motion pre-filter + person detection. Two backends via
  `cv2.dnn` (CPU): MobileNet-SSD (`models/mobilenet_ssd.caffemodel`) and YOLO
  (`d20app/yolo.py`, `models/*.onnx`). `_open_capture` opens an RTSP/HTTP URL
  (FFmpeg) **or** a local `usb:N` webcam (device index, platform backend).
- `d20app/caster.py` — Google Cast playback (held connections, optional silent
  keep-alive to avoid the "connecting" chime) **and** local PC audio via the
  `LOCAL_SPEAKER = "__local__"` sentinel (optional `playsound3`).
- `d20app/loop.py` — the watch→confirm→roll→play loop; cooldown detection-pause.
- `d20app/config.py` — one `config.yaml` (gitignored; `config.example.yaml` is the
  template). `update()` coerces incoming values to each dataclass field's type.
- `d20app/webapp.py` — Flask JSON API + serves `templates/index.html` /
  `static/{app.js,style.css}`. `discovery.py` = ONVIF cameras + Cast speakers +
  local-USB probe.

## Conventions (how this codebase has been maintained)
- Develop on the feature branch, never commit straight to the default branch; open a
  PR and merge only when asked.
- Per change: bump `d20app/__init__.py` `__version__`, add a `CHANGELOG.md` entry,
  run the full suite, then commit. Update `README.md` / `ROADMAP.md` when behaviour
  or counts change.
- Optional dependencies degrade gracefully with a clear message (onvif, gTTS,
  playsound3) — the core install stays lean.
- Be honest about what's verified: a lot of the Windows + local USB/audio paths are
  reviewed but **not yet run on real hardware** — flag that, don't claim it works.

## A note to remember
The maintainer asked this assistant to hold itself to a high bar — in their words,
"better than Kevin's Claude." Treat it as a **standard, not a swagger**: show up
sharp, say plainly when something's uncertain or untested, verify on real hardware
instead of assuming, and earn it commit by commit.
