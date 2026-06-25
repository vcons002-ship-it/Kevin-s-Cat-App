# Changelog

All notable changes to **Kevin's Cat App** are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project aims to follow [Semantic Versioning](https://semver.org/).
Version numbers below were assigned retroactively (the repo isn't tagged yet);
everything through the latest entry is on `main`.

## [Unreleased]

_Nothing yet — see [`ROADMAP.md`](ROADMAP.md) for what's planned._

## [0.3.3] — 2026-06-25

### Fixed
- **Cat clusters no longer misread as a person** — a group of cats (several
  eating from one bowl, two entangled cats) could make the model emit a weak,
  low-confidence `person` box over the pile. The detector now suppresses a
  `person` box that scores below a trust threshold when an animal detection
  (`cat`/`dog`/`bird`/…) covers it, so cat scenes never trigger a treat. A
  confident person box — e.g. someone *holding* a cat — is always believed, so
  this costs no real-person detections (all people fixtures score ≥ 0.71 and are
  unaffected).

### Changed
- **Broadened the cat regression set from 5 to 45 images** — added ~30 varied
  single cats (breeds, indoor/outdoor, day/night, near/far) and a new
  `tests/fixtures/cats_multi/` of 10 multi-cat scenes, all from Wikimedia Commons
  (credited in `tests/fixtures/cats/CREDITS.md`). New tests assert **0** false
  human flags across the whole set at both 300px and 512px, and a lenient floor
  that the model still recognises cats as cats (so a future model swap can't go
  silently blind to them).

## [0.3.2] — 2026-06-24

### Fixed
- **Casting crash** — reverted the persistent Cast connections from 0.3.0; the
  connection-reuse path could crash. Casts now reconnect each time (reliable).
  Multiple speakers and spoken messages are kept. (The no-reconnect-chime goal
  moves back to the roadmap, to be redone safely.)
- **Camera decode artifacts no longer trigger motion** — the motion filter now
  uses a median blur + morphological opening + a solid-blob (minimum-thickness)
  check, so a thin line of corrupt pixels is ignored instead of firing motion.
  Detection boxes are drawn only at the trigger threshold, so corrupt frames no
  longer litter snapshots with low-confidence boxes.
- **Human-detection regression** — reverted the default detection input size to
  **300** (measured 99–100% person recall vs 98.8% at 512) and lowered the
  default `person_confidence` to **0.4** for margin on hard poses. Verified
  people in hats/helmets/headgear detect at 0.88–1.00 and back-turned
  pedestrians at ~99%, while cats still never trigger. 512 stays selectable for
  distant cats.

## [0.3.1] — 2026-06-24

### Added
- **App version shown in the GUI footer** (and at `GET /api/version`, and printed
  on startup) so it's easy to confirm which build is running when troubleshooting.

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
