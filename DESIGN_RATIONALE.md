# Design rationale — for reviewers

This document exists so a reviewer (human, or a Claude asked to review this repo)
can understand *why* the code is shaped the way it is, not just *what* it does. It
covers the reasoning behind the smart-detection arc (v0.29.0–v0.32.0) and the
standing principles that every change in this repo is held to. If a change appears
to violate something here, that's either a bug or it needs an argued exception —
flag it.

Companion documents: [`CLAUDE.md`](CLAUDE.md) (working conventions),
[`ROADMAP.md`](ROADMAP.md) (feature history + parked ideas with verdicts),
[`CHANGELOG.md`](CHANGELOG.md) (per-version detail), issue #65 (the "Find My Cat"
design review that several decisions below trace back to).

---

## 1. What the app is, and the bar it's held to

A D20 "treat roller": a background loop watches cameras; a **person** triggers a
dice roll and, on a win, a chime — the cue that it's OK to give the cat a treat.
Cats never roll but are **tracked** (sightings log, "Show cat" button). CPU-first,
no Docker, no cloud accounts, one Flask GUI. Deployment target is a NAS with an
RTX 3070 (8 GB); an RTX 5090 exists for experiments only — **nothing shipped may
depend on it**.

The maintainer set an explicit standard: *be honest about what's verified*. Much
of this codebase is CI-tested logic whose real-hardware behaviour (cameras, GPU
co-residency, VLM quality) has **not** been validated. Every PR states its
CI-verified/NAS-only split. A reviewer should treat unmarked confidence as a
defect: if a claim isn't tested here and isn't flagged as NAS-pending, that's
wrong.

## 2. Standing principles (the invariants)

1. **The treat path is sacred.** The person→roll→chime loop is the product.
   Every detection experiment (escalation, trail, temporal) is on-demand or
   opt-in and must not add latency, locks, or failure modes to that loop.
2. **Cheap-first, always.** CPU before GPU, motion math before YOLO, YOLO before
   VLM, one VLM call before many. The escalation ladder's rung order is this
   principle made executable, and tests assert it (a rung-1 hit must never touch
   the VLM).
3. **Never record an unconfirmed claim as fact.** A bare VLM `detect()` region is
   never logged as a sighting (open-vocabulary detectors love cat-shaped decoys) —
   it must be confirmed by YOLO or a voted query first. The "probable location"
   tier is *never* written to the sightings log. Vote ratios are reported as the
   honest confidence, not converted into fake certainty.
4. **Priors, not state.** Issue #65 rejected house-graph tracking (room-to-room
   state machines) because one missed transition corrupts the belief forever.
   That rejection stands. History may *rank* where to look (time-of-day prior,
   last-sighting hints, trail endpoints) but the app never *believes* it knows
   where a cat is without looking.
5. **Opt-in for anything with a cost or a privacy surface.** Live-camera VLM
   features sit behind the off-by-default `vlm_escalation` toggle. Cloud VLM mode
   sends frames off-device — that's the user's explicit choice, never a default.
   The moondream API key is never logged and never echoed back to the browser.
6. **Graceful degradation.** Optional deps (onvif, gTTS, playsound3, moondream)
   fail with a clear message, never a crash; the core install stays lean. A model
   that can't load raises a clear error — no silent fallback (a lesson from the
   MobileNet-SSD removal, 0.25.0).
7. **The GUI is the whole interface.** Every feature is point-and-click; API
   endpoints exist to serve the page, not as a separate product.

## 3. The smart-detection arc — what was built and why

The arc answers one question: **what closes the remaining detection gaps —
small/distant cats, still cats, cats the single-frame detector just misses?**
Four ideas were proposed by the maintainer; each got a feasibility verdict, and
the viable ones shipped in cost order.

### 0.29.0 — the escalation ladder (`d20app/escalation.py`, PR #66, merged)

*Gap:* a curled-up cat five pixels tall fails the full-frame detector.
*Insight:* two capabilities were already lying unused — the moondream package
ships `detect()` (pointing, not just yes/no) which the app never called, and the
motion pre-filter computed blob bounding boxes and then **threw them away**,
returning only a bool.

The ladder zooms in like a human would, cheapest look first:

1. **zoom+yolo** — square, padded, full-resolution crops around the hints
   (motion blobs + last sighting + a predicted-next box), rerun the normal fast
   detector. In a close-up the cat is big. No VLM involved.
2. **vlm+yolo** — one `detect()` call proposes regions; YOLO confirms on a crop.
   The proposal alone never wins (principle 3).
3. **vlm query** — voted yes/no queries on the crops, last resort.

**Only YOLO confirms (tightened in 0.33.0, after #69's decoy data).** The
maintainer's benchmarks measured VLMs at 37–42% false positives on the decoy
set; majority voting reduces run-to-run *variance*, not that systematic *bias* —
three votes agree wrongly on a convincing decoy. So a votes-only VLM "yes"
(rung 2's query fallback, all of rung 3) can never return `found`: it comes back
as a `vlm_probable` lead, surfaces in the "probable" tier (orange, never
recorded), and on a live camera it *boosts detection* so real YOLO gets the next
frames — a real cat becomes an ordinary recorded sighting, a decoy dies quietly.
The trust asymmetry is the design: the VLM proposes and suspects; YOLO (0% FP on
the same benchmark) decides.

Every confirmed find is tagged with its `source` (`zoom+yolo` / `vlm+yolo`) so
the NAS run can measure which rung actually earns its keep.

**Why injected callables:** `escalate()` takes `run_yolo`/`vlm_detect`/`vlm_query`
as parameters. The ladder's decision logic is pure math, fully unit-testable with
counters and stubs — no GPU, no model files, no network in CI. This is the
repo-wide testability pattern; expect it wherever expensive dependencies meet
logic.

**The CPU "cat targeting" predictor** (`predict_hint_box`) was a maintainer
refinement: extrapolate the cat's next position from the last two timestamped
fixes (centroid velocity), pad the box by staleness. Rationale: take load off the
GPU and reduce VLM reliance — the LLM is the *failsafe* for when the track is too
thin (<2 fixes or stale → `None`), not the first resort. Honest limit, stated in
code and docs: a sleeping cat has no velocity; prediction only helps the moving
case.

**Scope decision:** on-demand only (a button + endpoint), no hook in the camera
worker. The live loop is treat-critical (principle 1); wiring the ladder into the
automatic still-cat scan is a deliberate ~10-line follow-up *after* NAS
validation, reusing the same `vlm_escalation` gate.

### 0.30.0 — the cat trail (`d20app/trail.py`, PR #67, merged — main's tip)

*Gap/opportunity:* "colour in pixel changes to show movement" — refined by the
maintainer into something better than frame-to-frame diffing.

**Why null-frame diff:** diffing consecutive frames lights up only the *edges* of
motion. Diffing against the last **null frame** (the scene when nothing was
moving) lights up the **whole cat silhouette** wherever it differs from the empty
room. The null frame refreshes after a few still seconds, which self-heals slow
lighting and auto-exposure drift — the classic failure mode of a fixed background.

Each silhouette pixel is stamped with *when* it was covered; rendering sweeps hue
from blue (oldest) to red (newest). One image = path + direction + timing.

**The known ghost quirk is documented, not hidden:** a cat who settles gets
absorbed into the refreshed null frame, so her later departure lights her *old*
spot in the trail's oldest colours. This is an accepted trade-off of
self-healing baselines; a reviewer should not "fix" it without understanding
that pinning the null frame reintroduces lighting drift.

**Trail-endpoint targeting + the "probable location" tier** (maintainer
refinement): the trail's red end is a *coordinate the app computes itself*. If
detection fails but the last movement ended **inside** the view — not at a frame
edge, and (since 0.31.0) not inside an exit zone — that endpoint becomes the
ladder's strongest hint for a deep scrub. If even that fails, the app reports
**"probable location"**: an orange box, an explicit "no confirmed detection; last
movement ended here" note, the trail image as evidence — and **no sightings-log
entry** (principle 3). Three honest outcome tiers: confirmed / probable /
not-found. Collapsing probable into confirmed would be the single worst
regression a change could introduce here.

### 0.31.0 — zones, heat maps, time-of-day prior (on Dev, PR #68, unmerged)

- **Semantic zones**: per-camera named rectangles ("the couch", doorways) drawn
  with the same GUI interaction as the ROI picker. Sightings inside a zone are
  recorded with its name — human-meaningful logs. Zones marked **exit** give the
  trail's "may have left the view" check a precise answer (v1 was the cheap
  frame-edge heuristic; the doorway polygon is the real one).
- **Heat maps**: the 500-cap sightings log already contained the data; this is
  pure aggregation + Gaussian blur + a colormap over the current frame. No new
  ML, no new state.
- **Time-of-day prior**: `by_hour()` / `likely_cameras(hour)` rank cameras by
  historical presence around the current hour. This is deliberately the *weakest
  possible* form of "intelligent tracking across a known map": a sort order for
  a search sweep, never a belief (principle 4). Wording in the GUI reflects that
  ("Usually around now: …").

**The coordinate-space rule (read this before touching any box code):** zones are
in **full-preview coordinates** (that's where the user draws); detection boxes
are in **adjusted ROI-crop coordinates** (that's what the detector sees).
`zone_for()` shifts detection boxes by the ROI origin before testing. Also, the
detector's `_live_frame` is already adjusted in the sync path but **raw** in
smooth-feed mode — endpoint code must apply `_adjust` conditionally. Most subtle
bugs in this repo are coordinate-space bugs; when reviewing box math, first ask
"which space is this in?"

### 0.32.0 — temporal VLM mosaic (on Dev, PR #68, unmerged)

*Gap:* a single frame can miss what eight seconds make obvious (a cat walking
through). moondream is image-only — no video input.

Three temporal rungs were designed; the cheapest shipped: each camera keeps a
small **frame ring buffer** (8 frames, ≥1 s apart, downscaled to ≤480 px — a few
MB per camera), and `POST /api/vlm/temporal` tiles them into **one numbered,
age-labelled grid** ("1 (-4s)" … "N (now)") and asks a single voted query: *did a
cat appear or move through, and in which frame(s)?* One VLM call covers ~8 s.

**Why not a video-LLM:** Qwen2.5-VL-class models are a heavyweight new dependency
stack and 5090-tier; committing to that before measuring the near-free mosaic
trick would violate cheap-first. The middle rung (per-frame voted queries over
the ring) is a parked follow-up.

**The load-bearing caveat, stated everywhere it matters:** whether moondream can
*genuinely reason over a grid of numbered frames* is unproven. CI proves the ring
buffer, the grid geometry, the gating, and that the model is shown one mosaic
with the right prompt — it cannot prove the model's skill. This is the first
thing the NAS must validate, and the main reason PR #68 is being held unmerged.
The endpoint returns the mosaic itself so a human can judge what the model saw.

Since 0.33.0 (after #69's decoy data), a temporal "yes" is explicitly a **hint,
never a verdict**: the response labels it unconfirmed, and on a live camera it
boosts detection so YOLO decides on the next frames. Nothing is ever recorded
from the mosaic alone.

## 4. Ideas assessed and rejected (so they aren't re-proposed blind)

- **House-graph / room-transition tracking** — rejected in #65, stands rejected.
  Fragile state; one missed transition corrupts it. History is a hint, never a
  belief.
- **Wi-Fi CSI sensing** ("extra detection") — honest no for the app: needs new
  hardware/firmware, is tuned to human-sized RF disturbance (cats are marginal
  targets), and is a separate integration project. Experiment-only.
- **Per-cat identity via vision** — the two cats are similar enough that people
  struggle; a re-ID embedding benchmark is parked for the 5090 before judging.
  The pragmatic certain answer, noted in the ROADMAP: a **BLE collar tag**
  (espresense-style), which gives identity for free.
- **Multi-cat counting as ground truth** — too flaky to rely on; a multi-cat hit
  is accepted opportunistically, never depended on.
- **Super-resolution before detection** — empirically rejected (issue #69, a
  maintainer-run benchmark on the 5090, 199 cat + 43 null images). Real-ESRGAN
  applied per-tile before an unmodified YOLO *reduced* recall 6–8 points at
  every config and cost ~40× the time. The mechanism matters more than the
  number: SR optimises for human perception — it denoises and synthesises clean
  edges — which pushes the image out of the detector's training distribution;
  the confidence drop was monotonic with how much SR changed the pixels. The
  precise verdict: *SR before an unmodified COCO detector hurts*; SR could help
  a detector fine-tuned on SR'd imagery, which is a much larger project and not
  planned. Generative SR (SUPIR/InvSR) is untested but predicted at least as
  bad, with hallucinated-texture false positives on top. Don't re-propose this
  without new evidence of that calibre.
- **Wiring the ladder into the live loop now** — deferred, not rejected: it's a
  small follow-up gated on NAS validation (principles 1 and 2).

## 5. Engineering conventions a reviewer should hold changes to

- **Branching:** develop on `Dev`, PR `Dev` → `main`, merge only when the
  maintainer asks. Never commit straight to `main`.
- **Per change:** bump `d20app/__init__.py.__version__`, add a CHANGELOG entry,
  keep the full suite green (279 tests at 0.32.0), update README/ROADMAP when
  behaviour or counts change. (Docs-only commits are the historical exception to
  the version bump.)
- **Config back-compat:** `config.update()` coerces values to dataclass field
  types; new fields (e.g. `cameras[].zones`, `vlm_escalation`) must default
  sanely so old `config.yaml` files keep working untouched.
- **Locking:** `_live_lock` guards frame/ring access; `_roll_lock`/`_status_lock`/
  `_cam_lock` in the loop; hint-box lists are rebound-never-mutated so readers
  are lock-free (documented where used).
- **Test patterns to reuse, not reinvent:**
  - Expensive deps are injected callables; ladder tests use stubs + call counters
    to assert *decisions* (e.g. cheap-first).
  - Live-path endpoint tests build a real `DetectionLoop` with a stub-alive
    thread and a real `PersonDetector`, then inject frames directly — no camera.
  - Config isolation monkeypatches `config_mod.load`/`update` with
    tmp-path-default lambdas. **Do not** monkeypatch `CONFIG_PATH` — the
    functions bind their default at def time, and a test once silently wrote a
    fake API key into the real gitignored `config.yaml` that way.
- **Secrets:** the moondream API key is accepted per-request or stored in config;
  it is never logged and never sent back to the browser.

## 6. Current state and the open question (as of 2026-07-02, evening)

- **`main`** is at **v0.34.0** (PR #68 — zones/heat maps/prior, temporal mosaic,
  the VLM demotion, frame averaging — merged at the maintainer's instruction).
- **`Dev`** carries the benchmark-response arc: **v0.35.0** (the #70/#71
  benchmark-settled model lineup, the golden-export guard, the `auto`-CUDA
  default, 3×3/0.35 scan defaults), **v0.36.0** (the VLM tester fixes #72–#76:
  batch-prompt bug, P6 default, per-model prompts, 1000-image queue, detect
  mode, M3 reasoning toggle), and **v0.37.0** (temporal score fusion /
  track-before-detect — weak YOLO hits that chain smoothly AND move confirm as
  one `source="track"` sighting; movement is the decoy guard).
- **The authoritative benchmark is issue #70** — model tiers, tiling/overlap
  optima, FP16, the golden-export recipe, VLM prompt results (P6: 97%/2%), and
  the complementarity finding (the VLM rescues 12 of 26x's 17 misses). Future
  detection decisions should start there.
- **NAS-only validation queue** (nothing below is provable in CI):
  0. Track fusion on real footage — hit rate on walking cats; compression
     shimmer must not fake smooth, travelling chains.
  1. The temporal-mosaic premise — can moondream reason over a frame grid at
     all? (Prompt/tile-size iteration expected.)
  2. moondream `detect()` real-world quality and coordinate orientation.
     **Context that raises the stakes** (issue #69, in passing): the
     maintainer's benchmarks measured VLMs at **37–42% false positives on the
     decoy set** — versus tuned yolo26x at 91% recall / 0% FP. Majority voting
     reduces run-to-run *variance*, not systematic *bias*: if the model
     reliably thinks a cat-shaped decoy is a cat, three votes agree wrongly.
     This vindicated the ladder's never-trust-a-bare-VLM-answer rule, and in
     **0.33.0** the remaining gap was closed: a votes-only "yes" is demoted
     from *confirmed* to *probable* everywhere (ladder rungs 2–3, the temporal
     mosaic), never recorded, and on a live camera it boosts detection so YOLO
     can confirm. The NAS decoy run can still measure the voted-moondream
     setup's own FP rate for the record.
  3. VRAM co-residency: moondream2 (~4.6 GB) + CUDA YOLO on the 8 GB 3070 (the
     ladder's confirm rung defaults to CPU YOLO for exactly this reason).
  4. Trail behaviour over hours (ghosting frequency, lighting drift).
  5. Zone-drawing UX on real previews (v1 uses `prompt()`/`confirm()` —
     functional, unpolished, known).
  6. Heat-map readability on real scenes; the prior needs days of sightings.
  7. Per-camera RSS with ring + trail buffers on all cameras.
- **A numbering wart, owned:** in-code comments and CHANGELOG entries for
  0.31.0/0.32.0 cite **#68**, which is the PR shipping them — the standalone
  issues originally planned under those numbers were never opened, and a
  follow-up commit on Dev corrected the dangling `#69` citations. One pushed
  commit title still says "#69"; history was left unrewritten on purpose.

If you are reviewing a future change against this document, the questions to ask
are: *does it protect the treat path? does it try the cheap thing first? does it
record only confirmed facts? does it state plainly what hardware validation it
still owes?* Those four cover most of what this codebase considers correctness.
