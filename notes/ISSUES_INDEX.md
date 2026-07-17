# Open GitHub Issues — Index & Mapping

> Pulled from `vcons002-ship-it/Kevin-s-Cat-App` open issues on **2026-07-17**
> (37 open issues; PRs excluded). Source: GitHub REST issues API — `gh` isn't
> installed on this box, so the data came from the public API directly (same
> payload `gh issue list --json` / `gh issue view` return). All 37 were reported
> by `OmarTheHippo`.
>
> **Caveat on "open":** the maintainer merges PRs **without closing the issue**, so
> the open list overstates outstanding work — most #85–#106 shipped in 0.43–0.51.
> Status below is a best-effort call: **verified** where I could check it against the
> current tree (mostly `yolo.py`) or `HANDOFF.md` §6; **inferred / not re-verified**
> for the UI and VLM-tester items. Treat the inferred ones as leads, not facts —
> confirm against the tree before acting.

---

## Mapping at a glance

### GitHub ↔ `LIVE_TESTING_CONTEXT.md` issues 30–34 (direct 1:1)

The five live-testing findings were each filed as a GitHub issue:

| LIVE_TESTING | GitHub | Topic | Status |
|---|---|---|---|
| Issue 30 | **#109** | Provision regenerates valid models instead of verify-and-adopt; resets wipe local provenance | **OPEN** |
| Issue 31 | **#110** | dog-as-cat: overlapping strong-dog/weak-cat boxes not merged → false fusion confirm | **OPEN** |
| Issue 32 | **#111** | Find-my-cat leaks settings into the live detector; boost mis-tagged "still-scan" | **OPEN** |
| Issue 33 | **#112** | "Last known location" box shouldn't fade at 30 min | **OPEN** |
| Issue 34 | **#113** | "Follow" mode (feed auto-tracks the cat) + optional second live feed | **OPEN** |

### Audit findings H1–M8 ↔ GitHub

The full-code audit (`docs/reviews/2026-07-09-full-code-audit.md`) findings are **not
filed as GitHub issues** — the audit docs are their record. The one real overlap:

| Audit | GitHub | Relationship |
|---|---|---|
| **M5** — class-agnostic NMS at `yolo.py:666` (`detect_boxes`) drops a cat overlapping a person | **#110** (= LIVE 31) | **Two different NMS sites, same topic.** M5 = `detect_boxes:666` (class-agnostic single-pass). #110 = `merge_nms:675` (per-class cross-tile). Both real; they pull opposite ways (M5 wants less cross-class suppression; #110 wants opt-in dog+cat merge). See `LIVE_TESTING_CONTEXT.md` NMS section. |
| H1–H3, M1–M4, M6–M8 | — | No GitHub issue; audit-doc only. H1 (`_coerce` 500-on-save) is thematically near the save-coherence issues #91/#100/#102 but is a distinct, unfiled crash. |

**Audit findings not on GitHub (from `2026-07-09-audit-fixes-handoff.md`):** H1 `_coerce`
blank/None → HTTP 500; H2 `last_scan()`/`_scan_last` race → 500; H3 `init()` failure bricks
UI; M1 inline-URL creds leak; M2 SSRF via `camera_url`; M3/M4 frontend save races; M5 NMS
(above); M6 `config.example.yaml` missing 17 fields; M7 list/dict no coercion; M8 `stop()`
join-timeout. Recommended slice: **H1 → H2 → M1**.

---

## Genuinely open work (not yet shipped)

- **#109** — *Provision regenerates valid existing models instead of verify+adopt; git resets wipe local provenance.* `provision()` treats any non-`ok` file as "rebuild" (minutes) instead of hash + golden-head-verify + adopt into the manifest (seconds, proven). Manifest is committed, so `git reset` erases local provenance. Fix: verify-and-adopt + a gitignored `models_manifest.local.json`. **(= LIVE 30)**
- **#110** — *dog-as-cat: strong hit registers only as a fusion confirm.* Per-class `merge_nms` keeps a strong `dog` box and weak `cat` box on the same animal separate; with the toggle on they should merge into one strong **cat** *before* fusion. Fusion firing is the symptom. **(= LIVE 31; see audit M5 for the other NMS site.)**
- **#111** — *Find-my-cat leaks settings into the live detector; boost mis-tagged.* Find runs via `_run_test_detection()`, which mutates the shared live detector in place (tiling/confidence/classes) and never restores it → live feed runs scan settings for ~10 s. Boost sets `force_scan=True` (the still-scan flag) so boost detections log as "still-scan". Keep boost; fix the leak + the tag. **(= LIVE 32)**
- **#112** — *"Last known location" box shouldn't fade at 30 min* (`_LAST_KNOWN_TTL=1800`). Most useful for a long-still/asleep cat; age is already labelled, so staleness is transparent. Remove/extend TTL. **(= LIVE 33)**
- **#113** — *"Follow" mode + optional second live feed.* Follow toggle auto-switches the feed to the most-recent confirmed cat; optional 2nd feed shows the previous room. Core requirement: sticky, debounced, per-feed assignment (no flip-flop). Tuned live. **(= LIVE 34)**

---

## Shipped, but issue still open (verified against tree / HANDOFF §6)

- **#71** — Remove inferior models (11x/11m/26n) + verify golden exports. `yolo.py` has `DROPPED_MODELS` + the `(1,84,N)` golden-head guard in `detect_boxes`. **Shipped.**
- **#79** — Should dropped YOLO11m variants stay in the registry? Current `MODELS` no longer lists them (only `DROPPED_MODELS` raises a loud error). **Answered/shipped.**
- **#80** — Deployed 11n runs at 320 but labelled with 640 numbers. `yolo.py` now exports 11n at **640** (comment cites #80). **Fixed.**
- **#85** — TensorRT driver-guard misreads `CUDA UMD Version:` (610.x drivers). `_parse_cuda_version` regex now has optional `(?: UMD)?`. **Fixed.**
- **#90** — Collapse precision out of the model picker; report the *effective* accelerator on fallback. `resolve_variant` + `_fp16` hidden entries + `effective_accelerator`/`fallback_reason` annotation. **Shipped (0.45.0).**
- **#91** — Camera quick-toggles revert when saving another camera → uniform auto-save. **Shipped (0.50.0).**
- **#92** — "Show me the cat" active scan across cameras + selectable model/settings. **Shipped (0.47.0).**
- **#93** — Cat-sightings log + report card with source tags + in-page images. **Shipped (0.47.0).**
- **#94** — "Check for a still cat": dedicated model + own settings + last-run indicator. **Shipped (0.46.0).**
- **#95** — Move escalation into its own section (out of the VLM/API-key card). **Shipped (0.47.1)** — but #106 flags the *toggle* didn't move with it.
- **#97** — GUI consistency & layout pass (spacing/alignment/headers/arrows). **Shipped (0.47.1).**
- **#100** — Live setting changes need stop/start (config snapshotted at start) → hot-reload. **Shipped (0.48.0).**
- **#101** — Per-mode independent settings (live / still-scan / find); fix what "tiling"/"motion off" do. **Shipped (0.49.0).**
- **#102** — Move "last still scan" under Cat-cam "Check for still cat"; coherent save behavior. **Shipped (0.49.0 / 0.50.0).**
- **#106** — GUI layout pass #2 (incl. two items "claimed done that didn't take": escalation toggle still in VLM card; camera Edit arrow doesn't flip). HANDOFF marked verified complete against the tree. **Shipped.**
- **#86** — Setup should generate/refresh the settled model lineup (app silently runs stale/FP32/missing models). Provisioning exists; **partially shipped** — the verify-and-adopt gap is the still-open #109.

---

## Likely shipped — GUI/log polish, not re-verified this pass

Spot-check against the current tree before assuming done.

- **#88** — "Count dogs as the cat" / "Track fusion" checkbox labels overflow + misalign (CSS). Likely folded into the #97/#106 GUI passes.
- **#96** — Motion "custom" has no input fields; motion can't be turned off. Motion-off routing was touched in 0.48; the **custom-parameter fields** may still be missing. **Partial — verify.**
- **#98** — Never surface "dog" in the log; say "cat" when dog-as-cat is on. `loop.py` logs the raw `label`. **Verify** — related to #110.
- **#103** — Find-my-cat scans unwatched cameras + no "why it jumped" feedback + wrong live-feed camera after toggling watch. **Verify** — overlaps #111.
- **#104** — Workflow description + "still-scan" log tag inaccurate (don't reflect actual settings). The mis-tag half is tied to the still-open #111. **Verify.**
- **#105** — Recent-sightings thumbnails stretched; last-seen image causes layout flicker (need `object-fit` / reserved aspect box).

---

## VLM-tester items — status not verified this pass

These target the moondream tester / escalation VLM path; I didn't confirm them against the code.

- **#72** — VLM tester batch path ignores the user prompt (uses a baked-in one); also set validated default prompt **P6** (97% recall / 2% FP on moondream2).
- **#73** — Remove/raise the **UI batch upload cap** (e.g. 1000). Explicitly NOT moondream's `max_batch_size`(4)/`kv_cache_pages`(2048) VRAM limits — those must stay.
- **#74** — Warn that prompts are model-specific: the P6 prompt tuned for moondream2 causes ~100% FP on moondream3. Non-blocking warning on model switch.
- **#75** — Expose moondream `detect` mode (bounding boxes) in the tester to diagnose *where* a false-positive fires.
- **#76** — Add a `reasoning=True` toggle for moondream3 (currently called without it → runs in weaker non-reasoning mode). Default off; only affects M3.
- **#87** — Benchmark tool's Accelerator/Model dropdowns are hardcoded and out of sync (TensorRT missing from the benchmark `<select>`; stale/dropped models listed). Fix = generate from `ACCEL_OPTS`/`MODELS`, don't hand-maintain.

---

## Reference / design & benchmark docs (not implementation tickets)

Filed as issues but are thinking/results documents — the authoritative background the
architecture rests on.

- **#17** — Locator path findings: a sleeping cat downscaled to 640 is too small to detect (0.00 conf); the problem is **effective resolution**, not model/threshold. Proposed tiling + a GUI image tester. (Led to the tiled still-scan / locator path.)
- **#65** — "Find My Cat" design direction: scope decisions — **no** individual-cat ID, **no** reliable multi-cat counting, **no** house-graph transition modeling (fragile; priors-not-state). The design basis for #92/#109-113.
- **#69** — Super-resolution before detection (Real-ESRGAN) **conclusively rejected**: −6–8 recall points at every tiling config, ~40× slower; SR pushes images out of YOLO's training distribution. (Generative SR untested.)
- **#70** — Full cat-detection benchmark (199 cat + 43 null frames): the authoritative reference. 26x FP16 3×3/0.20 = **91% recall / 0% FP**; VLMs 37–42% FP on nulls (why "never record an unconfirmed claim" exists). NAS timings.
- **#71** — (Also an action item; see "Shipped" above.) Companion to #70: the lineup-settling ticket.

---

## Full numeric list (37)

| # | Title | Bucket |
|---|-------|--------|
| 17 | Locator path: still cats need higher effective resolution | Reference/design |
| 65 | "Find My Cat" — design direction & next steps | Reference/design |
| 69 | Super-Resolution before detection (Real-ESRGAN) — rejected | Reference |
| 70 | Cat Detection — full benchmark & deployment recommendation | Reference |
| 71 | Remove inferior models; verify golden exports | Shipped |
| 72 | Fix VLM tester — batch ignores prompt; set default prompt | VLM tester — unverified |
| 73 | Remove/raise batch file-upload limit | VLM tester — unverified |
| 74 | Warn prompts are model-specific (M2 prompt breaks M3) | VLM tester — unverified |
| 75 | Expose moondream `detect` mode in the tester | VLM tester — unverified |
| 76 | Add `reasoning=True` toggle for moondream3 | VLM tester — unverified |
| 79 | Should dropped YOLO11m variants stay in the registry? | Shipped/answered |
| 80 | Deployed 11n runs at 320 but advertised with 640 numbers | Fixed |
| 85 | TensorRT driver-guard misreads `CUDA UMD Version:` | Fixed |
| 86 | Setup should generate/refresh the settled model lineup | Partial (see #109) |
| 87 | Benchmark tool dropdowns hardcoded/out of sync | Unverified |
| 88 | Checkbox labels overflow + misalign (CSS) | Likely shipped — verify |
| 90 | Collapse precision out of picker; show effective accelerator | Shipped |
| 91 | Quick-toggles revert when saving another camera | Shipped |
| 92 | "Show me the cat" active scan + selectable model | Shipped |
| 93 | Cat sightings log + report card | Shipped |
| 94 | "Check for a still cat" — dedicated model + last-run | Shipped |
| 95 | Move escalation into its own section | Shipped (toggle flagged in #106) |
| 96 | Motion "custom" has no fields; motion can't be turned off | Partial — verify |
| 97 | GUI consistency & layout pass | Shipped |
| 98 | Never surface "dog" in the log (dog-as-cat) | Verify (rel. #110) |
| 100 | Live changes don't apply until stop/start (hot-reload) | Shipped |
| 101 | Per-mode independent settings; fix tiling/motion-off | Shipped |
| 102 | "Last still scan" placement + coherent save behavior | Shipped |
| 103 | Find-my-cat scans unwatched cams + no feedback + wrong feed | Verify (rel. #111) |
| 104 | Workflow text + "still-scan" tag inaccurate | Verify (rel. #111) |
| 105 | Thumbnails stretched; last-seen image flicker | Likely shipped — verify |
| 106 | GUI layout pass #2 (two items didn't take) | Shipped |
| 109 | Provision verify+adopt; resets wipe provenance | **OPEN** — LIVE 30 |
| 110 | dog+cat boxes not merged → false fusion confirm | **OPEN** — LIVE 31 / audit M5 |
| 111 | Find-my-cat leaks settings; boost mis-tagged | **OPEN** — LIVE 32 |
| 112 | "Last known location" box shouldn't fade at 30 min | **OPEN** — LIVE 33 |
| 113 | "Follow" mode + optional second live feed | **OPEN** — LIVE 34 |
