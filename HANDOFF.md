# Kevin's Cat App — Handoff

> A pick-up-cold orientation for the next engineer (or the next Claude session).
> For the *why* behind the architecture read [`DESIGN_RATIONALE.md`](DESIGN_RATIONALE.md);
> for *what every button does* read [`docs/USER_GUIDE.md`](docs/USER_GUIDE.md) (served at `/guide`);
> for maintenance conventions read [`CLAUDE.md`](CLAUDE.md). This file is the
> current-state snapshot that ties them together.

_Snapshot date: 2026-07-08 · version **0.50.0** · `main` @ `a714dec`_

---

## 1. What it is

A CPU-only "D20 treat roller." A background loop watches one or more cameras for a
**person** (cats are *tracked* but never trigger a roll), rolls a die on each allowed
detection, and on a winning roll plays a chime or spoken message on a Google Home
and/or the local PC speakers — the cue that it's OK to give the cat a treat. A Flask
single-page GUI configures and runs everything. No Docker, no cloud, no account.

---

## 2. Current state

| | |
|---|---|
| Version | **0.50.0** (`d20app/__init__.py`) |
| Default branch | `main` @ `a714dec` (all shipped work is merged here) |
| Integration branch | `Dev` — develop here, PR `Dev → main`, merge only when asked |
| Tests | **404 passing** across 35 files — `./venv/bin/python -m pytest -q` (~85 s) |
| Python | 3.11+, virtualenv at `./venv` |
| Working tree | clean; local `main` and `Dev` synced to origin |

**Run / setup**
- Launch: `./venv/bin/python run.py` → prints `http://<lan-ip>:8080` GUI URL.
- Setup: `setup.sh` (Linux/apt), `setup.ps1` / `setup.bat` + `start.bat` (Windows).
- Camera diagnostic: `check_camera.py`.

---

## 3. Repo map (where things live)

| Module | LOC | Role |
|---|---:|---|
| `d20app/webapp.py` | 2157 | Flask JSON API + serves `templates/index.html`, `static/{app.js,style.css}`. Every `/api/*` endpoint. |
| `d20app/detector.py` | 1416 | Motion pre-filter + person/cat detection; ROI, adjustments, tiling, still-scan/locator path, live frame publishing, `reconfigure()` hot-reload. |
| `d20app/loop.py` | 935 | Orchestrator: one `PersonDetector` + worker thread per watched camera; shared cooldown gate; per-camera roles (`roll`/`track_cats`); still-scan scheduler; hot-reload cadence. |
| `d20app/yolo.py` | 695 | YOLO runner abstraction over `cv2.dnn` (CPU) + accelerator backends (onnxruntime-CUDA, OpenVINO, TensorRT); `resolve_variant`, `detect_boxes[_tiled]`, `boost_variant`. |
| `d20app/trail.py` | 381 | `TrailTracker`: null-frame silhouettes coloured by recency ("cat trail"), path + legend, person-box exclusion. |
| `d20app/moondream.py` | 379 | Optional VLM (`query`, `detect_regions`); local moondream2 / cloud moondream3; majority voting. |
| `d20app/escalation.py` | 366 | Pure crop-math + the VLM escalation "ladder" (zoom+YOLO → VLM detect → voted query); CPU velocity predictor. |
| `d20app/caster.py` | 344 | Google Cast playback (held connections, silent keep-alive) + local PC audio (`__local__` sentinel). |
| `d20app/config.py` | 298 | One `config.yaml` (gitignored; `config.example.yaml` is the template). `update()` coerces to each dataclass field's type. |
| `d20app/provision.py` | 286 | Model provisioning (`models_manifest.json`); audit statuses ok/missing/stale/unverified. |
| `d20app/cats.py` | 241 | `CatTracker`: file-backed sightings (when/camera/where/source + snapshot); `describe_region()` thirds-grid. |
| `discovery.py` / `fusion.py` / `activitylog.py` / `dice.py` / `snapshots.py` / `heatmap.py` | | ONVIF+Cast+USB discovery / track fusion / activity log / RollGate + roll logic / snapshot store / sighting heat-map. |

Models (`models/*.onnx`) are **gitignored** — provisioned locally via `provision.py` /
`models/export_*.py`. A model that can't load raises a clear error (no silent fallback).

---

## 4. Conventions (how this repo is maintained)

- **Branching:** never commit straight to `main`. Develop on `Dev`, PR `Dev → main`,
  merge only when the maintainer asks. (GitHub's merge commits show "Unverified" — expected.)
- **Per change:** bump `d20app/__init__.py` `__version__`, add a `CHANGELOG.md` entry,
  run the full suite, then commit. Update `README.md` / `ROADMAP.md` / `CLAUDE.md` test
  counts when behaviour or counts change.
- **Optional deps degrade gracefully** with a clear message (onvif, gTTS, playsound3,
  moondream, onnxruntime-gpu, openvino) — the core install stays lean.
- **Honesty bar:** be explicit about what's verified vs. reviewed-but-not-run-on-hardware
  (see §7). The maintainer's standard: "better than Kevin's Claude" — say plainly when
  something is uncertain or untested.

---

## 5. Design invariants (do not break these)

From `DESIGN_RATIONALE.md`, the four questions any change must pass:

1. **The treat path is sacred.** The always-running watch→roll→play loop is the critical
   path. New features ship on-demand / behind toggles and must not add risk to it.
2. **Cheap-first.** Try the free/CPU thing before the expensive one (motion gate before
   YOLO; zoom+YOLO before the VLM; the escalation ladder stops at the first confirmed hit).
3. **Never record an unconfirmed claim.** A bare VLM "yes" is demoted to `probable`
   (never written as a sighting); only YOLO confirms. (0.33.0 closed this gap everywhere.)
4. **Priors, not state.** No fragile room-to-room state machine; history is a *hint*, never
   a belief (issue #65 rejected house-graph tracking).

---

## 6. Recent arc (what shipped lately)

| Version | Issues | Summary |
|---|---|---|
| 0.43.0 | #82 | TensorRT accelerator + driver guard + fallback surfacing. |
| 0.44.0 | #85–#88 | Guard regex, dropdown sync, checkbox CSS, model provisioning manifest. |
| 0.45.0 | #90 | Precision-from-accelerator (FP32 cv2.dnn vs FP16 onnx/trt), toggle-bug fix. |
| 0.46.0 | #94 | Still-cat scan gets a **dedicated model** + last-run indicator. |
| 0.47.0 | #92 #93 | Find-my-cat **active scan**; cats-only sightings log with source tags; lightbox. |
| 0.47.1 | #95 #97 | Escalation moved to its own card; GUI layout/consistency pass. |
| 0.48.0 | #100–#106 | **Config hot-reload** (workers re-read every ~2 s, `reconfigure()`); **motion-off runs the live path** (not the scan path); per-camera **live tiling**; find/GUI fixes. |
| 0.49.0 | #101 #102 | **Per-mode settings restructure** — still-scan + find each a global settings group (model/tiling/overlap/confidence + "each camera's own"); per-camera scan-model removed; last-scan indicator moved to Cat-cam. |
| 0.50.0 | #102 §2 | **Uniform GUI save behavior** (every control auto-saves on change); bottom button → "Save all settings" + explained; **cooldown applies live** (`_apply_shared_reload`); honest labels on the two genuinely start-time settings. |

---

## 7. Verified vs. NAS-only (the honesty split)

**CI-verified:** all detection/crop/mapping/ladder-decision logic, config coercion,
endpoint behaviour, GUI wiring (headless Playwright render checks + behavioral audits).

**NOT yet run on real hardware** (the "NAS validation queue" in `DESIGN_RATIONALE.md`
§277, and much of the Windows + local-USB/audio path). Do not claim these work:
0. Track fusion on real walking-cat footage.
1. Whether moondream can reason over a temporal frame-mosaic at all.
2. `detect()` real-world quality + coordinate orientation. (Context: the maintainer's
   benchmarks put VLMs at 37–42% false positives on the decoy set vs tuned yolo26x at
   91% recall / 0% FP — which is *why* invariant #3 exists.)
3. VRAM co-residency: moondream2 (~4.6 GB) + CUDA YOLO on the 8 GB 3070 (the ladder's
   confirm rung defaults to **CPU YOLO** for exactly this reason).
4. Trail behaviour over hours (ghosting, lighting drift).
5–7. Zone-drawing UX on real previews; heat-map readability; per-camera RSS with buffers.

Hardware targets: **3070 (NAS, 8 GB)** — moondream2 + YOLO, on-demand co-residency;
**5090 (testing)** — moondream3 / video-LLM / re-ID experiments, all opt-in, nothing
shipped depends on it.

---

## 8. Open issues — status

**32 issues are "open," but all are already addressed and merged.** The maintainer merges
the PRs but doesn't close the issues, so the open list overstates outstanding work. The
newest is **#106** (GUI layout pass #2) — verified complete against the tree, including its
two "claimed-done-but-didn't-take" items (escalation toggle now in `escalation-card`; camera
Edit arrow flips ▾/▴). Issues #91–#106 map to versions 0.44.0–0.50.0 above.

There is **no un-actioned issue and no un-answered reporter feedback** as of this snapshot.
An offer stands with the maintainer to bulk-**close** the shipped issues (`state_reason:
completed`) so the open list reflects reality — not yet done (closing others' issues is a
heavier action, left for explicit go-ahead).

**Watch for the recurring pattern:** issue reporter `OmarTheHippo` re-files items "marked
done that didn't take." Verify fixes against the actual tree before claiming done — don't
trust a prior comment.

---

## 9. Known warts (owned, not bugs to chase)

- **Numbering wart:** some 0.31.0/0.32.0 comments/CHANGELOG cite `#68` (the PR), and one
  pushed commit title says `#69` for issues never opened. History left unrewritten on purpose.
- **Zone drawing v1** uses `prompt()`/`confirm()` — functional, unpolished, known.
- **Two settings are start-time by design:** speaker choice and keep-connection-warm take
  effect on next Start (Cast keepalive/target lifecycle). Labeled inline, not silent (0.50.0).
- **CatTracker frame_size-vs-crop coordinate quirk** predates the escalation work; matched
  and noted, not fixed.

---

## 10. Standing constraints (session/agent operating rules)

- **Model identity** (`claude-opus-4-8`) must **never** appear in commits, PR titles/bodies,
  code, or any pushed artifact — chat replies only.
- **Commit trailer** ends with the `Co-Authored-By` + `Claude-Session` lines; **PR bodies**
  end with the "🤖 Generated with Claude Code" + session-link footer.
- **GitHub scope:** only `vcons002-ship-it/kevin-s-cat-app`. Use `mcp__github__*` tools
  (no `gh` CLI). Don't open a PR unless asked; be frugal with issue/PR comments.
- **Never** disable TLS or unset `HTTPS_PROXY`; outbound HTTPS goes through the agent proxy.
- **`moondream_api_key`** is never logged or returned by any endpoint.
- **Push:** `git push -u origin Dev`; retry network failures with backoff (2/4/8/16 s).

---

## 11. Suggested pick-up points

Nothing is blocking. If the maintainer wants more work, natural next steps:

1. **Close the shipped issues** on GitHub (quick, pending the maintainer's go-ahead).
2. **NAS validation pass** — the queue in §7 is the highest-value unknown; a single real
   frame with a known cat position validates `detect()` orientation first.
3. **Optional per-mode granularity** — the maintainer was offered per-camera (vs global)
   still-scan/find settings, and fully-live speaker/keep-warm changes; both deferred as
   "say the word."
4. **`caption()` report cards, cat re-ID (5090 benchmark), BLE-collar identity** — noted
   future ideas in `ROADMAP.md`, none committed.
