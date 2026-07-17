# Kevin's Cat App — Handoff

> A pick-up-cold orientation for the next engineer (or the next Claude session).
> For the *why* behind the architecture read [`DESIGN_RATIONALE.md`](DESIGN_RATIONALE.md);
> for *what every button does* read [`docs/USER_GUIDE.md`](docs/USER_GUIDE.md) (served at `/guide`);
> for maintenance conventions read [`CLAUDE.md`](CLAUDE.md). This file is the
> current-state snapshot that ties them together.

_Snapshot date: 2026-07-17 · version **0.51.0** · `main` @ `afc7746` (PR #114)_

> Also read [`LIVE_TESTING_CONTEXT.md`](LIVE_TESTING_CONTEXT.md) — findings from running
> the app on 7 real cameras that a static review can't see (live issues 30–34, the current
> GPU/TensorRT architecture, the motion-verdict weakness, and the resolved NMS question).

---

## 1. What it is

A "D20 treat roller." A background loop watches one or more cameras for a **person**
(cats are *tracked* but never trigger a roll), rolls a die on each allowed detection,
and on a winning roll plays a chime or spoken message on a Google Home and/or the local
PC speakers — the cue that it's OK to give the cat a treat. A Flask single-page GUI
configures and runs everything. No Docker, no cloud, no account.

Inference now runs on **GPU** as the primary path — **YOLO 26x** on a **TensorRT** FP16
engine (golden `(1,84,8400)` no-NMS head; the app strips Ultralytics' metadata header;
engines are locked to the exact TRT version + GPU). TensorRT made 26x cheap enough
(~18 ms untiled, ~93 ms 3×3-tiled on the NAS 3070) that the model tier collapsed to one
model — 11n is no longer used on GPU in any config. onnxruntime-CUDA and `cv2.dnn` CPU
remain as fallbacks. A moondream VLM validator tier and a motion pre-filter still gate
detection. See [`CLAUDE.md`](CLAUDE.md) for the full driver/stack specifics.

---

## 2. Current state

| | |
|---|---|
| Version | **0.51.0** (`d20app/__init__.py`) |
| Trunk | `main` @ `afc7746` (all shipped work is merged here; PR #114) |
| Dev branch | `kev` — Kevin develops here, PRs `kev → main` for the maintainer's review |
| Tests | **~410 passing** across 36 files — `./venv/bin/python -m pytest -q` (~3 min) |
| Python | 3.11+, virtualenv at `./venv` |
| Working tree | clean |

**Run / setup**
- Launch: `./venv/bin/python run.py` → prints `http://<lan-ip>:8080` GUI URL.
- Setup: `setup.sh` (Linux/apt), `setup.ps1` / `setup.bat` + `start.bat` (Windows).
- Camera diagnostic: `check_camera.py`.

**Hosts:** the production host is the **NAS** (RTX 3070, headless/compute-only, driver
610.43.02 / CUDA UMD 13.3, torch 2.12.1+cu130, TensorRT 11.1.0.106) — a pull-only
consumer that runs the app but never develops on it. Dev + benchmarking happen on a
separate **RTX 5090** box.

---

## 3. Repo map (where things live)

| Module | LOC | Role |
|---|---:|---|
| `d20app/webapp.py` | 2157 | Flask JSON API + serves `templates/index.html`, `static/{app.js,style.css}`. Every `/api/*` endpoint. |
| `d20app/detector.py` | 1416 | Motion pre-filter + person/cat detection; ROI, adjustments, tiling, still-scan/locator path, live frame publishing, `reconfigure()` hot-reload. |
| `d20app/loop.py` | 935 | Orchestrator: one `PersonDetector` + worker thread per watched camera; shared cooldown gate; per-camera roles (`roll`/`track_cats`); still-scan scheduler; hot-reload cadence. |
| `d20app/yolo.py` | 696 | YOLO runner abstraction. **GPU-primary:** TensorRT engine (the settled path) → onnxruntime-CUDA; `cv2.dnn`/OpenVINO CPU/iGPU as fallbacks. `resolve_variant`, `detect_boxes`, `merge_nms` (per-class), `boost_variant`, engine-metadata strip. |
| `d20app/trail.py` | 381 | `TrailTracker`: null-frame silhouettes coloured by recency ("cat trail"), path + legend, person-box exclusion. |
| `d20app/moondream.py` | 379 | Optional VLM (`query`, `detect_regions`); local moondream2 / cloud moondream3; majority voting. |
| `d20app/escalation.py` | 366 | Pure crop-math + the VLM escalation "ladder" (zoom+YOLO → VLM detect → voted query); CPU velocity predictor. |
| `d20app/caster.py` | 344 | Google Cast playback (held connections, silent keep-alive) + local PC audio (`__local__` sentinel). |
| `d20app/config.py` | 298 | One `config.yaml` (gitignored; `config.example.yaml` is the template). `update()` coerces to each dataclass field's type. |
| `d20app/provision.py` | 286 | Model provisioning (`models_manifest.json`); audit statuses ok/missing/stale/unverified. |
| `d20app/cats.py` | 241 | `CatTracker`: file-backed sightings (when/camera/where/source + snapshot); `describe_region()` thirds-grid. |
| `discovery.py` / `fusion.py` / `activitylog.py` / `dice.py` / `snapshots.py` / `heatmap.py` | | ONVIF+Cast+USB discovery / track fusion / activity log / RollGate + roll logic / snapshot store / sighting heat-map. |

Models (`models/*.onnx`) and TensorRT engines (`models/*.engine`) are **gitignored** —
provisioned locally via `provision.py` / `models/export_*.py` / `models/export_trt_engine.py`.
Engines are GPU- and TRT-version-specific, built once per machine. A model that can't
load raises a clear error (no silent fallback).

---

## 4. Conventions (how this repo is maintained)

- **Branching:** never commit straight to `main`. Kevin develops on `kev` and PRs
  `kev → main` for the maintainer's review. (GitHub's merge commits show "Unverified" —
  expected, don't rewrite.)
- **Per change:** bump `d20app/__init__.py` `__version__`, add a `CHANGELOG.md` entry,
  run the full suite, then commit. Update `README.md` / `ROADMAP.md` / `CLAUDE.md` test
  counts when behaviour or counts change.
- **Optional deps degrade gracefully** with a clear message (onvif, gTTS, playsound3,
  moondream, onnxruntime-gpu, openvino, tensorrt/cuda-python) — the core install stays lean.
- **Honesty bar:** be explicit about what's verified vs. reviewed-but-not-run-on-hardware
  (see §7). Say plainly when something is uncertain or untested — verify against the
  actual current code/runtime before asserting.

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
| 0.51.0 | PR #114 | **Two memory-leak fixes** — capture released on every reconnect (`_release_cap`); MJPEG stream can't spin forever (heartbeat + no-frame timeout). Merged, **not yet runtime-confirmed** against the real 30-min crash (§8). |

---

## 7. Verified vs. NAS-only (the honesty split)

**CI-verified:** all detection/crop/mapping/ladder-decision logic, config coercion,
endpoint behaviour, GUI wiring (headless Playwright render checks + behavioral audits).

**Run live, finding recorded:** track fusion has been exercised on real cameras. It
*works*, but is **near-useless for this setup** (Kevin's cats score ~0.93 clean and
rarely need weak-hit recovery) and may be a **net FP source** (issue 31) — the
`fusion_debug` logging in issue 31 is partly to decide whether to keep fusion at all.
Not an open verification item; an open *keep-or-cut* question.

**NOT yet run on real hardware** (the "NAS validation queue" in `DESIGN_RATIONALE.md`
§277, and much of the Windows + local-USB/audio path). Do not claim these work:
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

## 8. Open work — status

The GitHub #91–#106 set (versions 0.44.0–0.50.0) all shipped; the maintainer merges PRs
without closing issues, so that "open" list overstates outstanding work. The real open
work now lives in three docs, **not** as GitHub issues:

**a) Audit findings — documented, NOT yet fixed** (`docs/reviews/2026-07-09-full-code-audit.md`;
index in `2026-07-09-audit-fixes-handoff.md`). No CRITICAL issues. Recommended slice, in
order — each a small one-function fix + test:

| # | Sev | Finding | Location |
|---|-----|---------|----------|
| H1 | HIGH | `_coerce` raises on blank/`None` numeric → **HTTP 500 on routine auto-saves**. Proven at runtime. | `config.py:284-298` |
| H2 | HIGH | `last_scan()` iterates `_scan_last` while workers insert keys → **intermittent 500** on the 1.2 s `/api/cats` poll. | `loop.py:375` vs `:743` |
| H3 | HIGH | One failed request during `init()` **bricks the UI**. | `app.js:1860-1874` |
| M1 | MED | **Inline URL credentials leak** via `GET /api/config` + `/api/cameras/saved`. | `webapp.py:937,946` |
| M2–M8 | MED | SSRF via `camera_url`; frontend save races (M3/M4); NMS (M5, see §b); `config.example.yaml` missing 17 fields (M6); list/dict coercion (M7); `stop()` join-timeout (M8). | see audit doc |

Plus 14 LOW items. **Fastest-ROI slice: H1 → H2 → M1.**

**b) The NMS question is RESOLVED** (was the M5-vs-issue-31 contradiction). Confirmed by
reading the current code: `yolo.py` `merge_nms` (cross-tile) is **per-class** (issue 31
is right); the class-agnostic NMS M5 flags is a *different* site, `detect_boxes:666`
(single-pass). Both findings are correct about their own function — they were never the
same code; `LIVE_TESTING_CONTEXT.md`'s "same code ~line 666" framing was the error. Any
fix must respect that they pull opposite ways (M5 wants less cross-class suppression;
issue 31 wants opt-in dog+cat merge before fusion) — don't unify them.

**c) Kevin's live-testing findings 30–34** (`LIVE_TESTING_CONTEXT.md`) — surfaced by
running on 7 real cameras, invisible to static review: provisioning regenerates valid
models instead of verify-and-adopt (30); track-fusion class conflation + `fusion_debug`
(31); find-my-cat leaks settings into the live detector + mis-tags boost (32); "last
known" box shouldn't fade at 30 min (33); Follow mode + second live feed (34). Plus the
**motion-verdict weakness**: `contourArea` after a 5×5 MORPH_OPEN under-triggers on
thin/distant cat motion.

**Watch for the recurring pattern:** items get "marked done that didn't take." Verify
fixes against the actual tree/runtime before claiming done — don't trust a prior comment.

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

- **Who's who:** Kevin (GitHub `OmarTheHippo`) is the developer and has the **only**
  live-camera environment — runtime-confirmation tasks are his. The original maintainer
  now reviews PRs. Verify against the actual current code/runtime before asserting; Kevin
  reliably catches over-eager hypotheses. Give specific, falsifiable claims.
- **Branching / push:** develop on `kev`, `git push -u origin kev`, open a PR `kev → main`
  for review. Don't commit straight to `main`. Don't open a PR unless asked.
- **Commit trailer** ends with `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`;
  **PR bodies** end with the "🤖 Generated with [Claude Code](https://claude.com/claude-code)" footer.
- **GitHub:** repo is `vcons002-ship-it/Kevin-s-Cat-App`; use the `gh` CLI. Be frugal
  with issue/PR comments.
- **No hardcoding usage from descriptive labels** (e.g. "workhorse"/"light"/"heavy") —
  expose neutral, configurable mechanisms. Hardcoding-from-labels has caused real bugs here.
- **`moondream_api_key`** / `camera_password` are never logged or returned by any endpoint
  (except the M1 inline-URL leak still to be fixed — §8).
- **Never disable TLS verification** for outbound HTTPS (moondream cloud, discovery, etc.).

---

## 11. Suggested pick-up points

Nothing is blocking. Natural next steps, roughly in priority order:

1. **Runtime-confirm the 0.51.0 memory-leak fix** against the real 30-min crash (Kevin's —
   he has the cameras). Log RSS + open-FD + `len(threading.enumerate())` once a minute for
   30 min: RSS+FD climbing with a flat Python heap ⇒ Fix 1 (capture); thread count climbing
   ⇒ Fix 2 (stream). Merged but unconfirmed against the specific crash.
2. **Land the audit slice H1 → H2 → M1** (§8a) — three small one-function fixes + tests,
   overlaps Kevin's save-coherence observations.
3. **Kevin's live findings 30–34** and the motion-verdict weakness (§8c) — pick from
   `LIVE_TESTING_CONTEXT.md`. Provisioning verify-and-adopt (30) and the find-my-cat state
   leak (32) are the highest-value.
4. **NAS validation pass** — the VLM/trail/co-residency queue in §7 is the remaining
   hardware unknown (the GPU YOLO path itself is now live-established).
5. **`caption()` report cards, cat re-ID (5090 benchmark), BLE-collar identity** — noted
   future ideas in `ROADMAP.md`, none committed.
