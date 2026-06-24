# Changelog

All notable changes to **Kevin's Cat App** are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project aims to follow [Semantic Versioning](https://semver.org/).
Version numbers below were assigned retroactively (the repo isn't tagged yet);
everything through the latest entry is on `main`.

## [Unreleased]

_Nothing yet — see [`ROADMAP.md`](ROADMAP.md) for what's planned._

## [0.3.0] — 2026-06-24

### Added
- **Multiple speaker output** — pick several Cast devices and the treat plays on
  all of them at once.
- **Optional spoken message** — say something (e.g. "Give the cat a treat!")
  instead of a sound, synthesized with gTTS and cached.
- **Quiet time** — silence chimes during a daily window (wraps past midnight).
- **Region-of-interest picker** — grab a still in the GUI and drag a box to watch
  only part of the view.
- **Detection detail** (net input size) and **scan rate** (fps) controls in the GUI.
- **Annotated snapshots** on every detection — boxes around the person/cat shown
  as clickable thumbnails in the activity log.
- **Live odds readout** — "For those who are mathematically challenged: X%".
- **`ROADMAP.md`** (feature list + roadmap) and this **changelog**.

### Changed
- **Persistent Cast connections** — held open between treats, so there's no
  "connecting" chime or delay; stale connections rebuild automatically.
- Raised the default detection input to **512px** so a cat across the room is
  detected (300px missed it).

### Fixed
- **False positives with no real motion** — frames are Gaussian-blurred so sensor
  noise / a ticking timestamp overlay no longer count as motion, the first frame
  reports no motion, and a person must persist across *N* consecutive frames
  before anything fires.
- Camera **username/password fields overlapping** on narrow/mobile screens.

## [0.2.0] — 2026-06-23

### Added
- **Persistent activity log** in the GUI (survives restarts), colour-coded.
- **Non-human motion logging** — reports what moved (e.g. "cat moved").
- **Camera diagnostics** — `check_camera.py` and a "Camera connected (W×H)"
  heartbeat so a running-but-idle loop isn't silent.
- README **screenshot**, plus stopping/troubleshooting and non-root **systemd**
  documentation.

### Changed
- Open RTSP streams via **FFmpeg over TCP** (authenticates like VLC), with a
  fast-fail connect timeout and quieter decoder logging.

### Fixed
- **Detection model was non-functional** — the bundled weights were a training
  snapshot that didn't match the prototxt, so every detection scored 0 and no
  person was ever detected. Replaced with the matching *deploy* weights
  (~99% recall on 170 real images); added a regression test to guard it.
- **"401 Unauthorized"** when a stream worked in VLC — force the FFmpeg backend
  and percent-encode injected credentials.
- A failing camera **flooding the console** — back off and log the problem once.
- **Camera password leaked** in the "Started watching" log line — now masked.

## [0.1.0] — 2026-06-22

### Added
- Initial release: watch an IP camera, "roll a die" when a **person** enters, and
  cast a celebratory chime to a **Google Home / Nest** speaker on a winning roll —
  while **ignoring the cats**.
- Single-page **web GUI**; **ONVIF** camera and **Google Cast** speaker
  auto-discovery; custom sound upload.
- One-shot **`setup.sh`** (virtualenv, dependencies, bundled detection model,
  config) with a **Python 3.11+** guard and an optional `apt` install of
  `python3-venv` / `pip`.

[Unreleased]: https://github.com/vcons002-ship-it/Kevin-s-Cat-App/compare/main...HEAD
