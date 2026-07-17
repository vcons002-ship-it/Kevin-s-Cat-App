# Kevin's Cat App — notes for Claude

> **Reviewing this repo?** Read [`DESIGN_RATIONALE.md`](DESIGN_RATIONALE.md)
> first — it explains the *why* behind the architecture, the standing invariants
> (treat path is sacred, cheap-first, never record unconfirmed claims, priors not
> state), the rejected ideas, and what is deliberately still unverified on real
> hardware. For *what the app does and how the features are used* — the default
> workflow, every button, every toggle — read
> [`docs/USER_GUIDE.md`](docs/USER_GUIDE.md) (served in-app at `/guide`).
>
> **Also read [`LIVE_TESTING_CONTEXT.md`](LIVE_TESTING_CONTEXT.md)** — findings from
> running the app on 7 real cameras that a static review can't see (issues 30-34, the
> current architectural state, and a KNOWN CONTRADICTION between the code audit and
> live testing about NMS that must be resolved before touching either fix).

A D20 "treat roller": a background loop watches a camera for a **person** (cats never
roll, but are **tracked** - see `cats.py` / the "Show cat" feature), rolls a die on
each allowed detection, and on a winning roll plays a chime (or a spoken message) on a
Google Home and/or this PC's speakers - the cue that it's OK to give the cat a treat. A
Flask single-page GUI configures and runs it. No Docker, no cloud, no account.

## Current state (updated - the app moved past the earlier CPU-only design)
- **GPU inference is now the primary path.** Detection runs **YOLO 26x** on GPU, with
  **TensorRT** engines (FP16, golden-head `(1,84,8400)` no-NMS, Ultralytics metadata
  header stripped by the app, locked to the exact TRT version + GPU). Untiled+TRT
  ~18ms, 3x3-tiled+TRT ~93ms. TensorRT made 26x cheap enough to use everywhere, so the
  model tier effectively collapsed to one model (26x); 11n is not used on GPU in any
  config. onnxruntime/`cv2.dnn` CPU paths still exist as fallbacks.
- **Production host (NAS):** RTX 3070, headless (no display - driver display-risks are
  irrelevant), driver 610.43.02 / CUDA UMD 13.3, torch 2.12.1+cu130, TensorRT
  11.1.0.106. Survives kernel updates via DKMS (headers must track the kernel). The NAS
  is a pull-only consumer - it runs the app, never develops on it.
- **Benchmarking host:** a separate RTX 5090 box (where dev + Claude Code run).
- A VLM (moondream) validator tier exists; motion pre-filtering gates detection.
- Version was 0.51.0 at the dev-handoff (PR #114). See
  `docs/reviews/2026-07-09-audit-fixes-handoff.md`.

## Run / test
- Python 3.11+. Virtualenv at `./venv`.
- Tests: `./venv/bin/python -m pytest -q` (~410 tests; keep them green, add tests with
  fixes).
- Launch: `./venv/bin/python run.py` -> prints a `http://<lan-ip>:8080` GUI URL.
- Setup: `setup.sh` (Linux/apt) or `setup.ps1` / `setup.bat` (Windows).

## Layout
- `d20app/detector.py` - motion pre-filter + person/cat detection. YOLO backend
  (`d20app/yolo.py`); GPU via TensorRT/onnxruntime, CPU via `cv2.dnn`. `_open_capture`
  opens an RTSP/HTTP URL (FFmpeg) **or** a local `usb:N` webcam. Motion verdict =
  `contourArea(largest blob) >= min_area_frac*H*W` after a 5x5 MORPH_OPEN (NOTE: this
  under-triggers on thin/distant cat motion - see LIVE_TESTING_CONTEXT.md).
- `d20app/caster.py` - Google Cast playback + local PC audio (`LOCAL_SPEAKER`
  sentinel).
- `d20app/loop.py` - the watch->confirm->roll->play loop; cooldown detection-pause.
  Multi-camera orchestrator: one `PersonDetector` + worker thread per watched camera
  (`config.camera_targets`), shared cooldown gate. Per-camera **roles** gate behaviour
  (`roll` -> treats, `track_cats` -> sightings). Per-camera live feeds
  (`/api/stream?camera=`); records sightings via `cats.py`.
- `d20app/cats.py` - `CatTracker`: file-backed sightings (when/camera/where + snapshot)
  behind `/api/cats`; `describe_region()` maps a box to a thirds-grid location.
- `d20app/config.py` - one `config.yaml` (gitignored; `config.example.yaml` is the
  template - NOTE: audit found it's missing ~17 fields, M6). `update()` coerces values
  to each dataclass field's type (audit found `_coerce` raises on blank/None -> HTTP 500
  on auto-save, H1).
- `d20app/webapp.py` - Flask JSON API + serves `templates/index.html` /
  `static/{app.js,style.css}`. `discovery.py` = ONVIF cameras + Cast speakers + USB
  probe.
- `d20app/provision.py` - model/engine provisioning + manifest (issue 30: should
  verify-and-adopt existing valid files instead of regenerating; needs a gitignored
  local manifest).

## Who's working here now
Kevin (GitHub: OmarTheHippo) is the developer. He has 20+ years engineering experience,
has validated this codebase line-by-line for weeks, and has the ONLY live-camera
environment - so runtime-confirmation tasks are his. The original maintainer now
reviews PRs.

**Working style:** verify against the actual current code/runtime before asserting -
Kevin reliably catches over-eager hypotheses; give specific, falsifiable claims he can
check, and say plainly when something is a guess or is untested. Do NOT hardcode
model/setting *usage* inferred from descriptive labels (e.g. "workhorse"/"light"/
"heavy") - expose neutral, configurable mechanisms; hardcoding-from-labels has caused
real bugs here.

## Conventions
- Branching: `main` is the trunk. Kevin develops on his branch (`kev`) and opens PRs to
  `main` for the maintainer's review. Don't commit straight to `main`. (GitHub merge
  commits show as "Unverified" - expected, don't rewrite.)
- Per change: bump `d20app/__init__.py` `__version__`, add a `CHANGELOG.md` entry, run
  the full suite, then commit. Update `README.md` / `ROADMAP.md` when behaviour or
  counts change.
- Optional dependencies degrade gracefully with a clear message (onvif, gTTS,
  playsound3) - the core install stays lean.
- Be honest about what's verified: several Windows + local USB/audio paths are reviewed
  but **not yet run on real hardware** - flag that, don't claim it works. Same for the
  memory-leak fix, which is merged but NOT yet runtime-confirmed against the real 30-min
  crash (Kevin's to confirm - he has the cameras).

## The standard
Show up sharp: say plainly when something's uncertain or untested, verify on real
hardware instead of assuming, and earn it commit by commit. Correctness and honesty over
confidence.
