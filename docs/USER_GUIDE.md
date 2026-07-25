# Kevin's Cat App — User & Workflow Guide

This guide explains **what the app does automatically, what each button does, and
what every toggle changes** — for people using the GUI and for anyone (or any AI)
working on the code. The *why* behind the design lives in
[`DESIGN_RATIONALE.md`](../DESIGN_RATIONALE.md); the measurements behind the
defaults live in issue **#70** (the authoritative benchmark). This document is
served in the app at **`/guide`** (the ❓ link in the header).

---

## 1. The app in one minute

Cameras watch rooms. When a **person** appears, the app rolls a D20 — a winning
roll plays a chime on your Google Home / PC speakers: *time for a cat treat*.
**Cats never roll** — but every cat sighting is logged (when, which camera,
where), and a set of detection tools works to make sure a cat is *found*, even
when she's small, dark, asleep, or walking through the edge of a frame.

Two rules govern everything:

1. **The treat path is sacred.** The person→roll→chime loop is fast and simple;
   every clever cat feature runs beside it, never inside it.
2. **Only YOLO confirms.** The AI second-opinion (moondream, a vision-language
   model) can *propose* and *suspect*, but per our own benchmarks it false-fires
   on cat-shaped decoys — so nothing it says alone is ever recorded as a fact.

---

## 2. The default workflow (what runs the moment you press Start)

This is the **benchmark-proven pipeline** (#70, #71): every default below was
measured, not guessed. Per watched camera, for every frame:

1. **Motion gate** *(pure CPU, always on)* — the frame is diffed against a
   rolling baseline. No pixel change → the neural net doesn't run at all.
   Sensor noise, timestamp tickers, and thin decode artifacts are filtered out.
2. **YOLO** *(the only thing that can confirm anything)* — runs on motion, on the
   **`auto` accelerator**: CUDA when it genuinely binds (verified — never a
   silent slow fallback), else CPU. A person here feeds the dice roll
   (after `confirm_frames` consecutive frames — the anti-fluke streak); a cat
   here becomes a logged sighting.
3. **Track fusion** *(on by default)* — weak cat detections (too uncertain to
   count alone) are remembered across frames. If they **chain smoothly and
   actually travel** across the frame, they're confirmed as one sighting tagged
   `source: track`. This catches the walking cat no single frame can confirm;
   the movement requirement keeps cat-shaped decoys out.
4. **Cat trail** *(always on, CPU)* — every frame is compared to the last *still*
   frame, so the cat's whole silhouette is painted with a timestamp. The trail
   (blue = older → red = newest, with a route line and an on-image legend) shows
   where she went; its red end is where the last movement stopped. A global
   lighting/exposure change resets the baseline instead of flood-painting the
   room, and only animal-sized blobs are kept. View it as a still (🌈 Show
   trail) or live: the **🌈 Trail overlay** checkbox by the live feed paints it
   onto the stream, updating as the cat moves. Next to it, **📍 Last known**
   (on by default) keeps a purple, age-labelled box at the newest *confirmed*
   cat position — "where was she last?" stays answered even when nothing is
   detected right now. It hides itself while a live cat box is on screen (the
   live box is the answer), reappears the moment she vanishes at the exact
   spot of her last confirmed detection, and fades out after 30 min. Movement inside a detected
   **person** box is excluded (and scrubbed retroactively once the net names the
   mover), so your arm's cat-sized gestures don't paint the trail — it stays a
   *cat* trail.
5. **Still-cat scan** *(default: every 30 s per cat-tracking camera)* — a
   sleeping cat makes no motion, so the net is forced periodically with the
   heavy settings. The clock runs from the last **still-scan**, so activity in
   the room can't defer it (before 0.62.0 any net run reset it, and a person at a
   desk suppressed the scan entirely — in the very room where a cat sleeps
   unnoticed). Every run writes a line to the activity log, found or not, so
   "did it run on Basement?" is answerable from history. Settings: **3×3 tiling at
   0.35 overlap** (the benchmark's clean recall-per-ms winner). Note the still-scan
   has its **own** overlap, separate from Find's — set them together if you tune
   one. The scan can run its own
   **Scan model** (heavier than the live one — it gets one hard static look,
   where live detection gets many frames of a moving cat), and each camera
   card shows *"still scan: Ns ago — cat found / no cat"*. **Run the still scan**
   chooses *on a timer* (every interval, whatever else is happening) or *only when
   no cat was found* (skipped while one has already been detected there in the same
   window). Frame averaging is currently inert — see the CHANGELOG for 0.62.0.

Nothing else runs on its own. The VLM, the escalation ladder, the temporal
check — all **on-demand buttons**, and live-camera use of them sits behind an
off-by-default privacy toggle.

### The recommended upgrade (one-time, on the GPU box)

Out of the box the app uses the bundled **yolo11n at 640** (the benchmark
"floor": 75% recall / 0% FP — the 640 export is the one those numbers belong
to, #80). The **proven workhorse** is **yolo26x FP16 @ 3×3**
(91% / 0%, 167 ms warm on the deployment 3070). It's too big to bundle, so
export it once where your `.pt` weights live:

```
python scripts/export_yolo.py --model yolo26x --imgsz 640 --out yolo26x --half
```

…then pick **YOLO26x FP16** in the model dropdown (it appears once the file
exists) and keep the accelerator on **auto**. The app *refuses* a bad
(end2end) export with a clear error, so a wrong re-export can't silently
degrade recall. Middle option: **yolo26m** (bundled, 82–85%).

---

## 3. GUI map — every card, every button

The page reads top to bottom: *live things first, setup below the divider.*

### ▶ Status bar (top)

**Start watching / Stop** — the whole background loop. The line under it shows
the **active workflow** (model, tiling, accelerator, which assists are on) so
you always know what's actually running.

### 🐱 Cat cam

The star. **"Show me the cat!"** flashes whenever a cat is on camera *right
now* (moving or still); tapping it jumps to the live feed of the camera that
saw her, rotating if several rooms have cats. Below it: the latest sighting
(with zone names like "the couch" if you've drawn zones), and *"Usually around
now"* — which cameras historically see cats at this hour (a hint from history,
never a claim). **Check for a still cat** sets how often the heavy scan runs.

### Live detection

The annotated live view — boxes drawn as things are recognised. The camera
selector picks which feed. Network cameras always play at their own frame rate,
while the detector looks every `scan_fps` frames — so the picture stays fluid
whatever the detection rate.

**📸** saves the camera's current picture — no boxes, lossless PNG — to the app's
`screenshots/` folder, named with the time and camera so you can match it against
the camera's own recordings later. PNG rather than JPEG because these get fed back
into the **Test detection** tool and compared against what the live scan reported:
that comparison only means anything if the pixels are the ones the detector
actually judged. Those are kept indefinitely; the automatic
detection snapshots in the activity log are not. Each camera's chip in **Setup**
shows how far behind the live edge it is, if it ever falls behind.

### 🔬 Test detection

**A sandbox.** Upload a screenshot or clip and try detection settings against
it — model, tiling, confidence, gamma… **Nothing here touches the live loop**
until you save the settings to a camera or the defaults. Inside:

- **Run detection** — one pass with the current knobs; boxes + timing shown.
- **sweep options** — "Benchmark this image": every model × tiling combo on
  this frame, as a shareable HTML report.
- **batch several images** — the same sweep across a whole folder, with
  recall and false-positive rates reported separately (tick "cat present" per
  image so empty-room controls count correctly).

### 🌙 Cat-presence (VLM)

The AI second opinion — **moondream**, a small vision-language model. On-demand
only, and it **never confirms a sighting by itself** (YOLO decides; see rule 2).
Needs the optional moondream install + API key (set below in the card).

- **Ask "is there a cat?"** — the yes/no question on an uploaded frame, with
  the model's reasoning. *Passes* runs it N times and majority-votes (the vote
  ratio is the honest confidence).
- **📍 Where?** — moondream's *detect* mode draws boxes where it *thinks* a cat
  is. This is the **false-positive diagnostic**: run it on an empty frame that
  keeps false-firing to see what feature (a cushion, a shadow) it locks onto.
  Purely informational — nothing is recorded.
- **🎞 Temporal check** — (video uploads, or live cameras via the escalation
  row) tiles the last ~8 seconds of frames into one numbered grid and asks
  "did a cat pass through?". A **yes is a hint, not a verdict**: on a live
  camera it briefly boosts real detection so YOLO can confirm.
- **Prompt box** — the question the model is asked. The default is the
  bake-off winner (97% recall / 2% FP); **prompts are model-specific** — the
  GUI swaps defaults when you change models and warns if you carry a custom
  prompt across (the moondream2 prompt scores ~100% false positives on
  moondream3). Always end a custom prompt with "Answer Yes or No".
- **Reasoning (M3)** — lets moondream3 think before answering (no effect on
  moondream2).
- **batch** — the same voted question across many images, scored like the
  detection batch.
- **escalation: zoom in and look again** — the ladder (next section).

### 🔍 The escalation ladder (inside the VLM card)

For when the normal look misses: it does what you would — **zoom in where it
makes sense and look again**, cheapest step first, stopping at the first
*YOLO-confirmed* find:

1. Full-resolution close-ups around the best hints (recent motion, last
   sighting, where the trail ends, where a moving cat is *predicted* to be) →
   re-run YOLO. Most misses die here, no AI needed.
2. Ask moondream to *point* → YOLO checks each pointed spot.
3. Voted yes/no on the close-ups — which can only ever produce a **"probable"**
   (orange) lead, never a recorded sighting.

If nothing is confirmed but the trail ends *inside* the room (not at a door/
frame edge), you get the honest **"probable location"**: *no confirmed
detection, but the last movement ended here.* A VLM-flavoured lead also fires a
**targeted boost**: for the next ~10 s the camera's forced scans additionally
zoom a full-resolution crop around the lead's box and run the **heaviest model
on disk** there (26x > 26m > the camera's own — the benchmark's answer: a
heavier model helps, a bigger raw input doesn't; the crop is the resolution
lever that works). While it runs, the feed shows the spot as an orange
"checking (lead)" box. Run the ladder on an uploaded test frame
anytime, or against a **live camera** once **Live-camera escalation** is
enabled (see toggles). **🌈 Show trail** and **🔥 Heat map** on the same row
show the movement trail and the room's historical hot-spots.

### Activity log

Everything the app did, color-coded, with annotated snapshots — the fastest
way to understand a false positive or a missed roll.

### ⚙ Setup (below the divider)

**1. Cameras** — add/discover cameras; each has its own role (**rolls** and/or
**tracks cats**), model, confidences, motion sensitivity, and:
**📷 Set region** (detect only inside a box), **🚪 Add zone** (name spots like
"the couch" so sightings say so; mark doorways as *exit* zones so the app knows
when a cat may have left the room). **2. Speakers**, **3. Sound / spoken
message**, **4. Game rules** (dice, DC, cooldown — the odds line does the math),
**5. Quiet time**.

---

## 4. Every toggle, and what it changes

Detection assists (defaults = the proven workflow; each is safe to flip):

- **Track fusion** *(per camera; default ON)* — off: every frame is judged
  alone, and a walking cat that never scores above the confidence bar is never
  logged (pre-0.37 behaviour). On: strings of weak-but-moving hits become one
  honest `source: track` sighting. Turn off if a camera's compression shimmer
  ever fakes a moving track (none seen in testing; the movement + smooth-chain
  requirements are the guards).
- **Motion hold** *(global; default 2 s; Live detection ⚙)* — how long the
  detector keeps running after motion stops, refreshed by every further motion
  frame. At 0 the net only ever sees frames that *moved* — which are exactly the
  motion-blurred, mid-stride ones — so a cat that pauses mid-walk goes unwatched
  and the sharpest frames (just after it settles) are thrown away. Raise it on
  cameras where cats amble and stop; lower it if a camera with steady background
  movement (a curtain, a tree outside) keeps the net running for nothing. It
  gates like motion, so a treat-cooldown pause and round-robin still win over it.
  Distinct from **Cat check**, which covers a cat that never moves at all.
- **Scan frames** *(per camera; default 3)* — frames averaged per still-cat
  scan. 1 = off (single frame). Higher = cleaner dim-room scans at slightly
  longer scan time. Never affects the fast treat path.
- **Cat tiling / overlap** *(per camera; default 3×3 / 0.35)* — the still-scan's
  magnifier. `off` = fastest, misses small cats; 4×4 = ~1 more cat per hundred
  at 1.7× the cost (and high overlap starts buying recall with false
  positives). 3×3 is the measured sweet spot.
- **Accelerator** *(default auto)* — `auto` = CUDA when it genuinely binds,
  else CPU (it tells you). Force `cpu` to keep the GPU free; `onnx-cuda` to
  *require* CUDA (errors rather than run slow).
- **Check for a still cat** *(default 30 s)* — the heavy-scan cadence.
  *Always* = every frame (most CPU); *Off* = motion only (a sleeping cat is
  only found by the on-demand tools).
- **Count dogs as the cat** *(per camera)* — the model sometimes calls a cat a
  "dog", especially at distance; in a no-dog household, counting dogs raises
  cat recall for free.

VLM & privacy:

- **Live-camera escalation** *(default OFF)* — the gate for pointing any VLM
  tool at a **live camera** (the ladder's "Check camera now", the live temporal
  check). Off = VLM tools work only on frames you upload. Turning it on never
  makes anything automatic — it just enables the buttons.
- **Mode: local vs cloud** — local moondream2 is private and free (needs a
  GPU); cloud moondream3 **sends the frame off-device**. The GUI warns when a
  cloud mode is selected.
- **Passes** *(default 3)* — votes per VLM question. 1 is fine for casual use
  (and is what the benchmark deployed); more passes = steadier verdicts on hard
  frames, linearly slower.
- **Reasoning (M3)** *(default off)* — moondream3's thinking mode.

Performance & plumbing:

- **Round-robin** *(default off)* — with many cameras, only N detect at a time,
  rotating; resting cameras cost nothing but react slower. **👁 always-watch**
  exempts a camera (e.g. the treat cam).
- ~~**Smooth live feed**~~ — *removed in 0.59.0; it is now always on for network
  cameras and there is no toggle.* It was never really a comfort setting: an RTSP
  stream is a **queue**, so reading it slower than the camera sends doesn't give
  you fewer frames, it gives you ever-older ones — measured at 46 seconds of lag
  per minute, with detection acting on those same stale frames. Staying at the
  live edge means reading every frame, and every frame read has to be decoded, so
  displaying them costs almost nothing on top. To cut CPU, lower the **camera's**
  stream frame rate instead. (USB cameras are unaffected — they don't queue.)
- **Pause during cooldown** *(default on)* — after a treat, skip the net until
  the next roll window (nothing it sees could trigger anyway).
- **Keep speaker connection warm** *(default off)* — loops a silent clip so the
  Google Home doesn't play its "connecting" chime before each treat.

---

## 5. Recipes

- **"Where's my cat right now?"** → tap **Show me the cat!** If it's not
  flashing: open the VLM card → escalation → pick the camera → **Check camera
  now**. Look at the trail (🌈) — the red end is where movement stopped; an
  orange box is the app saying "probably here, unconfirmed".
- **A camera keeps missing the cat** → screenshot a miss → **Test detection** →
  try 26x/26m, 3×3 tiling, gamma up for dark rooms → **sweep options** to
  measure instead of eyeballing → save to that camera.
- **A camera false-fires on something cat-shaped** → upload the offending frame
  to the VLM card → **📍 Where?** shows what it locks onto → add an exclusion
  to the prompt, or mask the spot out with **📷 Set region**.
- **Night misses** → raise **Scan frames** to 5–6, try gamma ~1.5 in the test
  tool first; the real fix is more light or the fine-tuning guide in
  [`ROADMAP.md`](../ROADMAP.md).
- **Make sightings say "the couch"** → camera card → **🚪 Add zone** → drag a
  box, name it; mark doorways as **exit** zones to sharpen "did she leave the
  room?".
- **Compare models on your own frames** → Test detection → **batch several
  images** (tick "cat present" honestly, include empty-room controls) — recall
  and false positives are reported separately, like the real benchmark.

---

## 6. How a sighting earns its way into the log (trust ladder)

From most to least direct — every entry carries its `source`:

- `yolo` — the detector cleared the confidence bar on a frame. The everyday
  case.
- `track` — a string of weak detections that chained smoothly **and moved**
  (track fusion). Honest score = the mean of the weak hits.
- `zoom+yolo` / `vlm+yolo` — the escalation ladder found it in a close-up /
  moondream pointed and YOLO confirmed.
- **Never recorded**: a VLM yes by itself (shown as an orange *probable* at
  most), the temporal check's yes (it boosts real detection instead), anything
  from 📍 Where? (diagnostic only).

That ordering *is* the app's philosophy: cheap first, YOLO decides, history and
AI only ever say where to look harder.
