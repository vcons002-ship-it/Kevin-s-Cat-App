"use strict";

const $ = (id) => document.getElementById(id);
const api = async (path, opts) => {
  let res;
  try {
    res = await fetch(path, opts);
  } catch (_) {
    // Network error / server unreachable: return a handled shape instead of
    // rejecting, so a single blip can't abort init() before its pollers are
    // registered and brick the UI (H3). Callers already handle {ok:false}.
    return { ok: false, body: null, netError: true };
  }
  let body = null;
  try { body = await res.json(); } catch (_) { /* no body */ }
  return { ok: res.ok, body };
};
const postJSON = (obj) => ({
  method: "POST", headers: { "Content-Type": "application/json" },
  body: JSON.stringify(obj),
});
const esc = (s) => String(s == null ? "" : s)
  .replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

// Global (non per-camera) config fields that map 1:1 to inputs.
const FIELDS = ["dice_sides", "dc", "cooldown_seconds", "quiet_start", "quiet_end"];

// Motion-sensitivity presets → the three raw MotionPrefilter knobs.
const MOTION_PRESETS = {
  low:    { motion_min_area_frac: 0.006,  motion_diff_threshold: 30, motion_min_blob_px: 18 },
  medium: { motion_min_area_frac: 0.003,  motion_diff_threshold: 25, motion_min_blob_px: 14 },
  high:   { motion_min_area_frac: 0.0015, motion_diff_threshold: 18, motion_min_blob_px: 10 },
};

// Per-camera dropdown options.
// The model dropdown is the single resolution selector: YOLO sizes are baked into
// the export (320/640/960/1280).
// Fallback only — the real list comes from /api/models (the registry-backed source
// shared with the benchmark, so the live picker can never drift again). #50
// (#87: only the two BUNDLED models, with their registry labels — never a
// dropped model or a stale size, even for the pre-load flash.)
let MODEL_OPTS = [
  ["yolo11n", "YOLO11n (640) — floor, low CPU"],
  ["yolo26m", "YOLO26m (640) — lightweight"],
];

// Load the canonical model list and build every model dropdown from it (the Test card's
// #t_model here; the per-camera editor reads MODEL_OPTS when it renders). #50
async function loadModelOptions() {
  const { body } = await api("/api/models");
  if (body && Array.isArray(body.models) && body.models.length) {
    MODEL_OPTS = body.models.map((m) => [m.value, m.label]);
  }
  const sel = $("t_model");
  if (sel) {
    const prev = sel.value;
    sel.innerHTML = MODEL_OPTS.map(([v, l]) => `<option value="${esc(v)}">${esc(l)}</option>`).join("");
    if (MODEL_OPTS.some(([v]) => v === prev)) sel.value = prev;
  }
}
// ---- model provisioning (#86): audit table + generate button ----------------
let provisionPoll = null;

async function loadModelsAudit() {
  const { body } = await api("/api/models/audit");
  if (!body || !Array.isArray(body.items)) return;
  const badge = (st) => st === "ok" ? "✅ ok"
    : st === "missing" ? "⬜ missing"
    : `⚠ ${st}`;
  $("models-audit").innerHTML = `<table class="bench-tbl"><tr>
      <th>file</th><th>kind</th><th>precision</th><th>status</th><th></th></tr>
    ${body.items.map((r) => `<tr><td>${esc(r.file)}</td><td>${r.kind}${r.optional ? " (opt.)" : ""}</td>
      <td>${r.precision}</td><td>${badge(r.status)}</td><td class="hint">${esc(r.note || "")}</td></tr>`).join("")}
    </table>`;
  const btn = $("models-provision");
  btn.disabled = !body.can_provision || body.running;
  $("models-provision-note").textContent = body.running ? "generating…"
    : body.can_provision ? ""
    : "needs `pip install ultralytics` on this machine (build-time only)";
}

async function pollProvision() {
  const { body } = await api("/api/models/provision/status");
  if (!body) return;
  const log = $("models-provision-log");
  log.classList.remove("hidden");
  log.textContent = (body.log || []).join("\n") + (body.error ? `\nERROR: ${body.error}` : "");
  if (!body.running) {
    clearInterval(provisionPoll); provisionPoll = null;
    loadModelsAudit(); loadModelOptions();     // refresh dropdowns with new files
  }
}

const ACCEL_OPTS = [["auto", "Auto — CUDA if available (rec.)"], ["cpu", "CPU"], ["opencl", "OpenCL iGPU"], ["openvino-auto", "OpenVINO AUTO"], ["openvino-gpu", "OpenVINO GPU"], ["onnx-cuda", "NVIDIA CUDA (onnxruntime-gpu)"], ["tensorrt", "TensorRT engine — fastest (CUDA 13+ driver, prebuilt engine)"]];
const SENS_OPTS = [["low", "Low"], ["medium", "Medium"], ["high", "High"], ["custom", "Custom"], ["off", "Off — detect every frame"]];
const TILING_OPTS = [["off", "Off"], ["2x2", "2×2"], ["3x3", "3×3"], ["4x4", "4×4"]];
// (The "Cat input size" 960/1280 knob was removed in 0.40.0 — the benchmark (#70)
// measured >640 input as WORSE than 640; tiling is the resolution lever.)
const optsHTML = (list, val) => list.map(([v, l]) =>
  `<option value="${v}" ${String(v) === String(val) ? "selected" : ""}>${l}</option>`).join("");

// Region of interest [x, y, w, h] in original-frame pixels (the picker writes this).
let roi = null;
let roiEdit = null;     // {card} while the shared ROI picker is open

let speakers = [];
let cameras = [];        // full per-camera dicts from /api/cameras/saved
let activeCameras = [];  // names being watched
let camDefaults = {};    // defaults for a new camera (from global cfg)

// ---- speakers --------------------------------------------------------------
async function loadSpeakers(detect) {
  const sel = $("speaker-select");
  sel.innerHTML = `<option disabled>${detect ? "Detecting…" : "Loading…"}</option>`;
  const { body } = await api("/api/speakers");
  speakers = body || [];
  sel.innerHTML = "";
  const localOpt = document.createElement("option");
  localOpt.value = "__local__";
  localOpt.textContent = "This PC (local audio)";
  if (savedSpeakers.includes("__local__")) localOpt.selected = true;
  sel.appendChild(localOpt);
  for (const s of speakers) {
    const opt = document.createElement("option");
    opt.value = s.name;
    opt.textContent = s.is_group ? `${s.name}  (group)` : s.name;
    if (savedSpeakers.includes(s.name)) opt.selected = true;
    sel.appendChild(opt);
  }
  if (!speakers.length) {
    const none = document.createElement("option");
    none.value = ""; none.disabled = true;
    none.textContent = "(no Cast speakers found — check same WiFi)";
    sel.appendChild(none);
  }
  updateSpeakerWarning();
}

function selectedSpeakers() {
  return Array.from($("speaker-select").selectedOptions).map((o) => o.value).filter(Boolean);
}

// ---- camera manager (multi-camera, per-camera settings) -------------------
function cardByName(name) {
  return Array.from($("camera-list").querySelectorAll(".cam-card"))
    .find((c) => c.dataset.name === name);
}

function cameraCardHTML(cam) {
  const watched = activeCameras.includes(cam.name);
  const roiTxt = Array.isArray(cam.roi) && cam.roi.length === 4
    ? `region ${cam.roi[2]}×${cam.roi[3]}px` : "whole frame";
  return `<div class="cam-card" data-name="${esc(cam.name)}">
    <div class="cam-head">
      <strong>${esc(cam.name)}</strong>
      <span class="cam-chip muted" data-chip></span>
      <span class="status-spacer"></span>
      <label class="checkbox" title="watch this camera"><input type="checkbox" data-watch ${watched ? "checked" : ""}/> Watch</label>
      <label class="checkbox" title="a person here rolls for a treat"><input type="checkbox" data-f="roll" ${cam.roll ? "checked" : ""}/> 🎲</label>
      <label class="checkbox" title="log cat sightings from here"><input type="checkbox" data-f="track_cats" ${cam.track_cats ? "checked" : ""}/> 🐱</label>
      <label class="checkbox" title="never rest this camera under round-robin"><input type="checkbox" data-f="always_watch" ${cam.always_watch ? "checked" : ""}/> 👁</label>
      <button type="button" class="ghost" data-toggle>Edit ▾</button>
    </div>
    <div class="cam-body hidden">
      <label>Stream URL <input type="text" data-f="url" value="${esc(cam.url)}" placeholder="rtsp://… or usb:0"/></label>
      <div class="grid">
        <label>Username <input type="text" data-f="username" autocomplete="off" value="${esc(cam.username)}"/></label>
        <label>Password <input type="password" data-f="password" autocomplete="off" placeholder="${cam.has_password ? "(unchanged)" : ""}"/></label>
        <label>Model <select data-f="model">${optsHTML(MODEL_OPTS, cam.model)}</select></label>
        <label>Accelerator <select data-f="accelerator">${optsHTML(ACCEL_OPTS, cam.accelerator)}</select></label>
        <label title="LIVE detection tiling (#101), independent of the still-scan's — splits each live frame into an overlapping grid so a small/distant cat fills more of the net. Default off; on multiplies per-frame cost by the tile count (watch CPU/GPU at your scan rate).">Live tiling <select data-f="live_tiling">${optsHTML(TILING_OPTS, cam.live_tiling || "off")}</select></label>
        <label title="overlap for live tiling when on (benchmark: 0.35 for 3×3, 0.20 for 2×2)">Live overlap <input type="number" step="0.05" min="0" max="0.5" data-f="live_tile_overlap" value="${cam.live_tile_overlap === undefined ? 0.2 : cam.live_tile_overlap}"/></label>
        <!-- Still-scan settings moved to the Cat-cam section (global, #101/#102). -->
        <label>Person confidence <input type="number" step="0.05" min="0.1" max="1" data-f="person_confidence" value="${cam.person_confidence}"/></label>
        <label title="how confident a cat detection must be to count (the locator path; independent of person)">Cat confidence <input type="number" step="0.05" min="0.1" max="1" data-f="cat_confidence" value="${cam.cat_confidence}"/></label>
        <label>Confirm frames <input type="number" min="1" max="30" data-f="confirm_frames" value="${cam.confirm_frames}"/></label>
        <label>Scan rate (fps) <input type="number" min="1" max="30" data-f="scan_fps" value="${cam.scan_fps}"/></label>
        <label>Notify floor <input type="number" step="0.05" min="0.1" max="1" data-f="label_floor" value="${cam.label_floor}"/></label>
        <label class="checkbox" title="count a 'dog' as the cat — for a no-dog household where the model mislabels the cat"><input type="checkbox" data-dog ${(cam.locator_classes || []).includes("dog") ? "checked" : ""}/> Count dogs as the cat</label>
        <label>Motion sensitivity <select data-f="motion_sensitivity">${optsHTML(SENS_OPTS, cam.motion_sensitivity)}</select></label>
        <label class="mcustom" title="fraction of the frame that must change to count as motion (higher = less sensitive)" style="${cam.motion_sensitivity === "custom" ? "" : "display:none"}">Motion min area <input type="number" step="0.0005" min="0" max="0.05" data-f="motion_min_area_frac" value="${cam.motion_min_area_frac}"/></label>
        <label class="mcustom" title="per-pixel brightness change that counts as different (lower = more sensitive)" style="${cam.motion_sensitivity === "custom" ? "" : "display:none"}">Motion diff threshold <input type="number" min="1" max="80" data-f="motion_diff_threshold" value="${cam.motion_diff_threshold}"/></label>
        <label class="mcustom" title="reject change regions thinner than this many pixels (kills artifact lines)" style="${cam.motion_sensitivity === "custom" ? "" : "display:none"}">Motion min blob px <input type="number" min="1" max="80" data-f="motion_min_blob_px" value="${cam.motion_min_blob_px}"/></label>
        <label class="checkbox" title="temporal fusion: a string of weak cat detections that chain smoothly and actually move across the frame is confirmed as one sighting (source 'track') — catches a walking cat no single frame can confirm. The movement requirement keeps cat-shaped decoys out."><input type="checkbox" data-f="track_fusion" ${cam.track_fusion !== false ? "checked" : ""}/> Track fusion</label>
      </div>
      <div class="row">
        <button type="button" class="ghost" data-roi>📷 Set region…</button>
        <button type="button" class="ghost" data-zone title="named spots ('the couch') label sightings; mark doorways as exits to sharpen the trail's left-the-room check">🚪 Add zone…</button>
        <span class="muted" data-roinote>${roiTxt}</span>
      </div>
      <div class="row"><span data-zonelist>${zoneChipsHTML(cam.zones || [])}</span></div>
      <div class="row">
        <button type="button" class="primary" data-save>Save camera</button>
        <button type="button" class="stop" data-delete>Delete</button>
        <span class="muted" data-note></span>
      </div>
    </div>
  </div>`;
}

function zoneChipsHTML(zones) {
  if (!zones.length) return '<span class="muted">no zones</span>';
  return zones.map((z, i) =>
    `<span class="muted" style="margin-right:8px">${z.exit ? "🚪" : "📍"} ${esc(z.name || "?")}
     <a href="#" data-zone-del="${i}" title="remove zone">×</a></span>`).join("");
}

function readCard(card) {
  const stored = cameras.find((c) => c.name === card.dataset.name) || {};
  const cam = { name: card.dataset.name };
  card.querySelectorAll("[data-f]").forEach((el) => {
    const f = el.dataset.f;
    if (el.type === "checkbox") cam[f] = el.checked;
    else if (el.type === "number") cam[f] = Number(el.value);
    else cam[f] = el.value;
  });
  // Locator classes: "the cat" is always included; the dog checkbox adds "dog".
  cam.locator_classes = card.querySelector("[data-dog]").checked ? ["cat", "dog"] : ["cat"];
  // Region: the picker stashes the chosen box on the card; else keep stored.
  cam.roi = card._roi !== undefined ? card._roi : (stored.roi || null);
  // Zones (#68): same stash-on-card pattern as the ROI.
  cam.zones = card._zones !== undefined ? card._zones : (stored.zones || []);
  // Expand a motion preset into the three raw knobs the detector actually reads.
  if (MOTION_PRESETS[cam.motion_sensitivity]) Object.assign(cam, MOTION_PRESETS[cam.motion_sensitivity]);
  if (!cam.password) delete cam.password;   // blank keeps the stored password
  return cam;
}

function renderCameras() {
  const list = $("camera-list");
  if (!cameras.length) {
    list.innerHTML = '<p class="muted">No cameras yet — click “Add camera”, or detect one above.</p>';
  } else {
    list.innerHTML = cameras.map(cameraCardHTML).join("");
    list.querySelectorAll(".cam-card").forEach(wireCameraCard);
  }
  populateLiveCameras();
}

function wireCameraCard(card) {
  const name = card.dataset.name;
  card.querySelector("[data-toggle]").onclick = (e) => {
    const body = card.querySelector(".cam-body");
    body.classList.toggle("hidden");
    // #106: the arrow reflects expanded/collapsed state (▾ closed, ▴ open).
    e.currentTarget.textContent = body.classList.contains("hidden") ? "Edit ▾" : "Edit ▴";
  };
  card.querySelector("[data-watch]").onchange = saveActiveCameras;
  // #96: "custom" motion is only usable if its knobs are editable — reveal them.
  const sens = card.querySelector('[data-f="motion_sensitivity"]');
  if (sens) sens.onchange = () => {
    card.querySelectorAll(".mcustom").forEach((el) => {
      el.style.display = sens.value === "custom" ? "" : "none";
    });
  };
  // Quick-toggles on the collapsed row auto-save on click (#91): they have no
  // visible Save button, and unsaved DOM state was silently wiped whenever any
  // OTHER camera saved (the save re-renders every card from the server). Each
  // sends a minimal {name, url, field} — the server merges into the saved
  // entry, so nothing else about the camera (or any other camera) is touched.
  for (const f of ["roll", "track_cats", "always_watch"]) {
    const el = card.querySelector(`.cam-head [data-f="${f}"]`);
    if (el) el.onchange = async () => {
      const payload = { name, url: card.querySelector('[data-f="url"]').value };
      payload[f] = el.checked;
      const { ok, body } = await api("/api/cameras/saved", postJSON(payload));
      if (!ok) {
        el.checked = !el.checked;          // don't lie about a failed save
        alert((body && body.error) || "Couldn't save the toggle.");
        return;
      }
      const stored = cameras.find((c) => c.name === name);
      if (stored) stored[f] = el.checked;  // keep the in-memory copy honest too
      refreshStatus();                     // roles feed the status chips
    };
  }
  card.querySelector("[data-save]").onclick = async () => {
    const note = card.querySelector("[data-note]");
    note.textContent = "Saving…";
    const { ok, body } = await api("/api/cameras/saved", postJSON(readCard(card)));
    if (ok) { await loadCamerasList(); }
    else { note.textContent = (body && body.error) || "Failed"; }
  };
  card.querySelector("[data-delete]").onclick = async () => {
    await api("/api/cameras/saved/delete", postJSON({ name }));
    await loadCamerasList();
  };
  card.querySelector("[data-roi]").onclick = () => openRoiEditor(card);
  card.querySelector("[data-zone]").onclick = () => openZoneEditor(card);
  card.querySelectorAll("[data-zone-del]").forEach((el) => {
    el.onclick = (e) => {
      e.preventDefault();
      const stored = cameras.find((c) => c.name === name) || {};
      const zones = (card._zones !== undefined ? card._zones : (stored.zones || [])).slice();
      zones.splice(Number(el.dataset.zoneDel), 1);
      card._zones = zones;
      card.querySelector("[data-zonelist]").innerHTML = zoneChipsHTML(zones);
      card.querySelector("[data-save]").click();      // persist the removal
    };
  });
}

// Zone editor (#68): reuses the ROI picker's frame + drag; the drawn box becomes a
// named zone (optionally an exit/doorway) appended to the camera's zone list.
let zoneEdit = null;

function openZoneEditor(card) {
  zoneEdit = { card };
  openRoiEditor(card);
  roi = null; drawRoiBox();
  $("roi-note").textContent = "Drag a box over the spot (the couch, a doorway…)";
}

function addCamera(prefill = {}) {
  const name = (prefill.name || (prompt("Name this camera:", "") || "")).trim();
  if (!name) return;
  if (cameras.some((c) => c.name === name)) { alert("A camera with that name already exists."); return; }
  const cam = Object.assign(
    { roll: true, track_cats: true, url: "", username: "", has_password: false, roi: null },
    camDefaults, prefill, { name },
  );
  cameras.push(cam);
  renderCameras();
  const card = cardByName(name);
  if (card) card.querySelector(".cam-body").classList.remove("hidden");
}

async function saveActiveCameras() {
  const names = Array.from($("camera-list").querySelectorAll(".cam-card"))
    .filter((c) => c.querySelector("[data-watch]").checked)
    .map((c) => c.dataset.name);
  activeCameras = names;
  await api("/api/cameras/active", postJSON({ names }));
  populateLiveCameras();
}

async function loadCamerasList() {
  const { body } = await api("/api/cameras/saved");
  cameras = body || [];
  renderCameras();
  populateTestSaveTargets();
}

function populateLiveCameras() {
  const names = activeCameras.length ? activeCameras : cameras.map((c) => c.name);
  const opts = names.length
    ? names.map((n) => `<option value="${esc(n)}">${esc(n)}</option>`).join("")
    : `<option value="">no cameras yet</option>`;
  // Both pickers stay VISIBLE even with one camera (they used to hide themselves,
  // which is why the picker seemed missing) — the second one is disabled until
  // there's a second feed to steer, and both are disabled while Follow drives them.
  for (const id of ["live-camera", "live-camera2"]) {
    const sel = $(id);
    if (!sel) continue;
    const prev = sel.value;
    sel.innerHTML = opts;
    if (id === "live-camera2") {
      // Never default to (or keep) the main feed's camera — a <select> otherwise
      // lands on the first option, which is exactly what made both feeds show the
      // same room (#116).
      const primary = $("live-camera").value || "";
      sel.value = (names.includes(prev) && prev !== primary)
        ? prev
        : (names.find((n) => n !== primary) || "");
    } else if (names.includes(prev)) {
      sel.value = prev;
    }
  }
  syncFeedControls();
}

// Detected network / USB cameras feed the "add camera" flow.
async function loadCameras(detect) {
  const sel = $("camera-select");
  sel.innerHTML = `<option value="">${detect ? "scanning…" : "↻ network…"}</option>`;
  const { body } = await api("/api/cameras");
  for (const c of body || []) {
    const opt = document.createElement("option");
    opt.value = c.rtsp_url || "";
    opt.textContent = c.rtsp_url ? c.name : `${c.name} (needs login)`;
    opt.dataset.name = c.name;
    sel.appendChild(opt);
  }
}

async function loadLocalCameras(detect) {
  const sel = $("camera-local-select");
  sel.innerHTML = `<option value="">${detect ? "scanning…" : "↻ USB…"}</option>`;
  const { body } = await api("/api/cameras/local");
  for (const c of body || []) {
    const opt = document.createElement("option");
    opt.value = c.value; opt.textContent = c.label;
    sel.appendChild(opt);
  }
}

// ---- region-of-interest picker (shared; opened per camera) ----------------
function drawRoiBox() {
  const cv = $("roi-canvas"), img = $("roi-img");
  if (!cv || !cv.getContext) return;
  const ctx = cv.getContext("2d");
  ctx.clearRect(0, 0, cv.width, cv.height);
  if (!img.naturalWidth) return;
  const sx = cv.width / img.naturalWidth, sy = cv.height / img.naturalHeight;
  // Zone mode: sketch the camera's existing zones for context (#68).
  if (zoneEdit) {
    const stored = cameras.find((c) => c.name === zoneEdit.card.dataset.name) || {};
    const zones = zoneEdit.card._zones !== undefined ? zoneEdit.card._zones : (stored.zones || []);
    for (const z of zones) {
      if (!z.box) continue;
      ctx.strokeStyle = z.exit ? "#ff9f43" : "#3ddc84"; ctx.lineWidth = 1;
      ctx.strokeRect(z.box[0] * sx, z.box[1] * sy, z.box[2] * sx, z.box[3] * sy);
      ctx.fillStyle = ctx.strokeStyle; ctx.font = "12px sans-serif";
      ctx.fillText(z.name || "?", z.box[0] * sx + 3, z.box[1] * sy + 13);
    }
  }
  if (!roi) return;
  ctx.strokeStyle = "#7c5cff"; ctx.lineWidth = 2;
  ctx.strokeRect(roi[0] * sx, roi[1] * sy, roi[2] * sx, roi[3] * sy);
}

function openRoiEditor(card) {
  const name = card.dataset.name;
  const stored = cameras.find((c) => c.name === name) || {};
  const existing = card._roi !== undefined ? card._roi : stored.roi;
  roi = Array.isArray(existing) && existing.length === 4 ? existing.slice() : null;
  roiEdit = { card };
  $("roi-editor").classList.remove("hidden");
  $("roi-note").textContent = "Grabbing a frame…";
  const img = $("roi-img");
  img.onload = () => {
    const cv = $("roi-canvas");
    cv.width = img.clientWidth; cv.height = img.clientHeight;
    drawRoiBox();
    $("roi-note").textContent = roi
      ? `Region ${roi[2]}×${roi[3]}px — drag to change`
      : "Drag a box over the area to watch";
  };
  img.onerror = () => {
    $("roi-note").textContent = "Couldn't grab a frame — save the camera's URL first, then retry.";
  };
  img.src = `/api/preview?camera=${encodeURIComponent(name)}&ts=${Date.now()}`;
  $("roi-editor").scrollIntoView({ behavior: "smooth", block: "center" });
}

function wireRoiCanvas() {
  const cv = $("roi-canvas"), img = $("roi-img");
  let start = null;
  const toImg = (e) => {
    const r = cv.getBoundingClientRect();
    const sx = img.naturalWidth / cv.width, sy = img.naturalHeight / cv.height;
    return [Math.round(Math.max(0, e.clientX - r.left) * sx),
            Math.round(Math.max(0, e.clientY - r.top) * sy)];
  };
  cv.onmousedown = (e) => { start = toImg(e); };
  cv.onmousemove = (e) => {
    if (!start) return;
    const [x, y] = toImg(e);
    roi = [Math.min(start[0], x), Math.min(start[1], y), Math.abs(x - start[0]), Math.abs(y - start[1])];
    drawRoiBox();
  };
  cv.onmouseup = () => {
    start = null;
    if (roi && (roi[2] < 8 || roi[3] < 8)) roi = null;
    drawRoiBox();
    $("roi-note").textContent = roi ? `Region ${roi[2]}×${roi[3]}px` : "Whole frame";
  };
}

// ---- sounds ----------------------------------------------------------------
async function loadSounds() {
  const sel = $("sound-select");
  const { body } = await api("/api/sounds");
  sel.innerHTML = "";
  for (const f of body || []) {
    const opt = document.createElement("option");
    opt.value = f; opt.textContent = f;
    sel.appendChild(opt);
  }
  if (savedSound) sel.value = savedSound;
}

function updateSpeakerWarning() {
  const chosen = selectedSpeakers();
  const groups = speakers.filter((s) => s.is_group && chosen.includes(s.name));
  const warn = $("speaker-warn");
  if (groups.length) {
    warn.textContent = `⚠ ${groups.map((g) => g.name).join(", ")} `
      + `${groups.length > 1 ? "are groups" : "is a group"} — plays on every speaker in it.`;
    warn.classList.remove("hidden");
  } else {
    warn.classList.add("hidden");
  }
}

function updateOdds() {
  const sides = Number($("dice_sides").value);
  const dc = Number($("dc").value);
  if (sides > 0 && dc >= 1) {
    const winning = Math.max(0, Math.min(sides, sides - dc + 1));
    const pct = ((winning / sides) * 100).toFixed(1);
    $("odds").textContent = `${winning} in ${sides} (${pct}%)`;
    $("odds-msg").textContent = `For those who are mathematically challenged: ${pct}%`;
  } else {
    $("odds").textContent = "—";
    $("odds-msg").textContent = "";
  }
}

// ---- config load/save (GLOBAL settings only) -------------------------------
let savedSpeakers = [], savedSound = "";

function applySpeechVisibility() {
  $("speech-row").classList.toggle("hidden", !$("use_speech").checked);
}

// The workflow line (status bar): says what actually runs when the loop is on,
// so "what is trying to do what" has a one-glance answer. Re-rendered whenever
// config or the VLM gate changes; every default it shows is benchmark-settled.
// The "Workflow: …" summary line was removed (with `lastCfg`, which only existed to
// feed it): it read the GLOBAL config while model, accelerator and track_fusion are
// all per-camera settings, and it printed the *requested* accelerator rather than the
// effective one — so it could claim "onnx-cuda" while a camera ran TensorRT or had
// silently fallen back to CPU. The honest, per-camera version of this is already the
// camera chips' `ran_on` / `fallback` (#90).

async function loadConfig() {
  const { body: cfg } = await api("/api/config");
  if (!cfg) return false;   // couldn't reach the app — signal it so init() can show the retry (H3)
  for (const f of FIELDS) {
    if ($(f) && cfg[f] !== undefined && cfg[f] !== null) $(f).value = cfg[f];
  }
  $("dont_interrupt_playback").checked = !!cfg.dont_interrupt_playback;
  $("keep_speakers_warm").checked = !!cfg.keep_speakers_warm;
  $("pause_during_cooldown").checked = cfg.pause_during_cooldown !== false;
  $("round_robin").checked = !!cfg.round_robin;
  $("round_robin_size").value = cfg.round_robin_size;
  $("round_robin_interval").value = cfg.round_robin_interval;
  $("smooth_feed").checked = !!cfg.smooth_live_feed;
  if ($("cat_scan_interval") && cfg.cat_scan_interval !== undefined && cfg.cat_scan_interval !== null)
    $("cat_scan_interval").value = String(cfg.cat_scan_interval);
  $("use_speech").checked = !!cfg.use_speech;
  if (cfg.speech_text) $("speech_text").value = cfg.speech_text;
  applySpeechVisibility();
  savedSpeakers = Array.isArray(cfg.speaker_names) && cfg.speaker_names.length
    ? cfg.speaker_names.slice()
    : (cfg.speaker_name ? [cfg.speaker_name] : []);
  savedSound = cfg.sound_file || "";
  // Global still-scan + find settings (#101/#102): populate dropdowns + seed values.
  const modelWithOwn = (val) => optsHTML([["", "each camera's own"]].concat(MODEL_OPTS), val || "");
  if ($("cat_scan_model")) $("cat_scan_model").innerHTML = modelWithOwn(cfg.cat_scan_model);
  if ($("scan_tiling")) $("scan_tiling").innerHTML = optsHTML(TILING_OPTS, cfg.cat_scan_tiling || "3x3");
  if ($("scan_tile_overlap")) $("scan_tile_overlap").value = cfg.cat_scan_tile_overlap ?? 0.35;
  if ($("scan_frames")) $("scan_frames").value = cfg.cat_scan_frames ?? 3;
  if ($("scan_confidence")) $("scan_confidence").value = cfg.cat_scan_confidence ?? 0;
  if ($("find_scan")) $("find_scan").checked = !!cfg.find_scan;
  if ($("find_model")) $("find_model").innerHTML = modelWithOwn(cfg.find_model);
  if ($("find_tiling")) $("find_tiling").innerHTML = optsHTML(TILING_OPTS, cfg.find_tiling || "3x3");
  if ($("find_tile_overlap")) $("find_tile_overlap").value = cfg.find_tile_overlap ?? 0.35;
  if ($("find_confidence")) $("find_confidence").value = cfg.find_confidence ?? 0;
  // Gear-popup settings (#117).
  if ($("fusion_debug")) $("fusion_debug").checked = !!cfg.fusion_debug;
  if ($("swap_confirm_count")) $("swap_confirm_count").value = cfg.swap_confirm_count ?? 0;
  if ($("camera_reuse_cooldown_seconds"))
    $("camera_reuse_cooldown_seconds").value = cfg.camera_reuse_cooldown_seconds ?? 0;
  activeCameras = Array.isArray(cfg.active_cameras) ? cfg.active_cameras.slice() : [];
  // Defaults for a brand-new camera = the current global detection settings.
  camDefaults = {
    model: cfg.detector_model, accelerator: cfg.accelerator,
    person_confidence: cfg.person_confidence, confirm_frames: cfg.confirm_frames,
    scan_fps: cfg.scan_fps, label_floor: cfg.label_floor,
    motion_sensitivity: cfg.motion_sensitivity,
    gamma: cfg.gamma, brightness: cfg.brightness,
    contrast: cfg.contrast, saturation: cfg.saturation,
    cat_confidence: cfg.cat_confidence, locator_classes: cfg.locator_classes,
  };
  if (!testTouched) setTestControls(camDefaults);   // seed the test tool from saved settings
  updateOdds();
  return true;
}

function gatherConfig() {
  const values = {
    speaker_names: selectedSpeakers(),
    sound_file: $("sound-select").value,
    use_speech: $("use_speech").checked,
    speech_text: $("speech_text").value.trim() || "Give the cat a treat!",
    dice_sides: Number($("dice_sides").value),
    dc: Number($("dc").value),
    cooldown_seconds: Number($("cooldown_seconds").value),
    pause_during_cooldown: $("pause_during_cooldown").checked,
    round_robin: $("round_robin").checked,
    round_robin_size: Number($("round_robin_size").value),
    round_robin_interval: Number($("round_robin_interval").value),
    cat_scan_interval: Number($("cat_scan_interval").value),
    quiet_start: $("quiet_start").value,
    quiet_end: $("quiet_end").value,
    dont_interrupt_playback: $("dont_interrupt_playback").checked,
    keep_speakers_warm: $("keep_speakers_warm").checked,
    // Global still-scan + find settings (#101/#102).
    cat_scan_model: $("cat_scan_model").value,
    cat_scan_tiling: $("scan_tiling").value,
    cat_scan_tile_overlap: Number($("scan_tile_overlap").value),
    cat_scan_frames: Number($("scan_frames").value),
    cat_scan_confidence: Number($("scan_confidence").value),
    find_scan: $("find_scan").checked,
    find_model: $("find_model").value,
    find_tiling: $("find_tiling").value,
    find_tile_overlap: Number($("find_tile_overlap").value),
    find_confidence: Number($("find_confidence").value),
    // Gear-popup setting (#117), hot: it rides the running detectors' reload.
    fusion_debug: $("fusion_debug").checked,
    swap_confirm_count: Number($("swap_confirm_count").value),
    camera_reuse_cooldown_seconds: Number($("camera_reuse_cooldown_seconds").value),
  };
  return values;
}

let saveInFlight = false, savePending = false;

async function saveConfig() {
  // Serialize and coalesce (audit M4). Every control posts the WHOLE form, so two
  // overlapping saves can land out of order and the earlier response silently
  // reverts the later edit — changing a few knobs quickly used to lose some of
  // them, with no error shown. Keep at most one request in flight; if changes
  // arrive while it's away, run exactly ONE more afterwards, re-gathering the form
  // so it carries every intervening edit rather than replaying a stale snapshot.
  if (saveInFlight) { savePending = true; return; }
  saveInFlight = true;
  const note = $("save-note");
  let ok = false;
  try {
    do {
      savePending = false;              // anything set after this point re-runs
      note.textContent = "Saving…";
      const gathered = gatherConfig();
      ({ ok } = await api("/api/config", postJSON(gathered)));
      // #104: keep the workflow line honest — reflect what was just saved, not the
      // config snapshot from page load (the "still-scan every 5s when off" bug).
    } while (savePending);
  } finally {
    saveInFlight = false;
  }
  note.textContent = ok ? "Saved ✓" : "Save failed";
  setTimeout(() => (note.textContent = ""), 2500);
}

// ---- control / status ------------------------------------------------------
let isRunning = false;

function renderCamChips(cams) {
  for (const c of cams || []) {
    const card = cardByName(c.name);
    if (!card) continue;
    const chip = card.querySelector("[data-chip]");
    if (!chip) continue;
    if (c.last_error) { chip.textContent = "⚠ failing"; chip.className = "cam-chip chip-bad"; }
    else if (c.resting) { chip.textContent = "💤 resting"; chip.className = "cam-chip muted"; }
    else if (c.connected && c.fallback) {
      // #90: a degraded accelerator must be visible, not a buried warning.
      chip.textContent = `● live (${c.ran_on}!)`; chip.className = "cam-chip chip-bad";
      chip.title = c.fallback;
    }
    else if (c.connected) {
      chip.textContent = c.ran_on ? `● live · ${c.ran_on}` : "● live";
      chip.className = "cam-chip chip-ok"; chip.title = "";
    }
    else { chip.textContent = "… connecting"; chip.className = "cam-chip muted"; }
  }
}

async function refreshStatus() {
  const { body } = await api("/api/status");
  if (!body) return;
  isRunning = !!body.running;
  const dot = $("status-dot"), text = $("status-text"), detail = $("status-detail");
  if (body.running) {
    dot.className = "dot running"; text.textContent = "Watching";
    $("start-btn").disabled = true; $("stop-btn").disabled = false;
  } else {
    dot.className = "dot stopped"; text.textContent = "Stopped";
    $("start-btn").disabled = false; $("stop-btn").disabled = true;
  }
  const parts = [];
  if (body.last_roll) parts.push(body.last_roll);
  if (body.rolls) parts.push(`${body.rolls} rolls, ${body.treats} treats`);
  if (body.last_error) parts.push(`⚠ ${body.last_error}`);
  detail.textContent = parts.join("  ·  ");
  detail.title = detail.textContent;   // clamped to 2 lines in CSS — hover for the rest
  renderCamChips(body.cameras);
  updateLiveView(body.running);
  updateEscalationCameraRow(body);
}

// ---- live detection feed ---------------------------------------------------
let liveOn = false, liveCam = null, liveTrail = false, liveLastKnown = true;

function updateLiveView(running) {
  const img = $("live-img"), note = $("live-note"), cam = $("live-camera").value || "";
  const want = running && $("live-enabled").checked;
  const trail = !!($("trail-overlay") && $("trail-overlay").checked);
  const lk = !($("lastknown-overlay") && !$("lastknown-overlay").checked); // default ON
  if (want && (!liveOn || cam !== liveCam || trail !== liveTrail || lk !== liveLastKnown)) {
    const q = (cam ? `camera=${encodeURIComponent(cam)}&` : "")
      + (trail ? "trail=1&" : "") + (lk ? "" : "last_known=0&");
    img.src = `/api/stream?${q}ts=${Date.now()}`;
    img.classList.remove("hidden");
    const parts = ["Live — green = person, orange = cat"];
    if (trail) parts.push("trail: blue = older → red = newest");
    if (lk) parts.push("purple = last known location");
    note.textContent = parts.join(" · ") + ".";
    liveOn = true; liveCam = cam; liveTrail = trail; liveLastKnown = lk;
  } else if (!want && liveOn) {
    img.src = ""; img.classList.add("hidden"); liveOn = false;
  }
  if (!want) {
    note.textContent = running
      ? "Live feed off — tick “Show live feed” to view."
      : "Start watching to see the live feed.";
  }
}

// ---- Follow mode + second feed (#113) --------------------------------------
// The sticky/debounced "which camera does each feed show" decision lives in the
// backend (d20app/feeds.py) so it's testable; this just renders the answer and,
// crucially, only re-points a stream when its camera actually CHANGES — setting
// img.src restarts the MJPEG connection, so re-pointing every poll would stutter.
let followOn = false, feed2On = false, feed2Cam = null;
// Feed locks are a runtime toggle owned by the browser (not saved): the poll sends
// which slots are pinned and the router pins whatever those slots currently hold.
const feedLocks = [false, false];

function renderFeedLocks() {
  feedLocks.forEach((locked, i) => {
    const btn = $("lock-" + i);
    if (!btn) return;
    // Locks only mean anything while Follow is assigning feeds; with Follow off the
    // pickers already decide, so the button would be a no-op.
    const usable = followOn && isRunning && (i === 0 || feed2On);
    btn.classList.toggle("hidden", !usable);
    btn.setAttribute("aria-pressed", String(locked));
    btn.textContent = locked ? "🔒" : "🔓";
  });
}

function toggleFeedLock(i) {
  feedLocks[i] = !feedLocks[i];
  renderFeedLocks();
  pollFeeds();
}

// Follow drives both pickers, so they're read-only while it's on (they still show
// where each feed is). The second picker is only steerable when there IS a second
// feed and Follow isn't already choosing for it.
function syncFeedControls() {
  const cam1 = $("live-camera"), cam2 = $("live-camera2");
  if (cam1) cam1.disabled = followOn;
  if (cam2) cam2.disabled = followOn || !feed2On;
}

async function pollFeeds() {
  renderFeedLocks();
  if (!isRunning || (!followOn && !feed2On)) { updateSecondFeed(null); return; }
  if (!followOn) { updateSecondFeed(null); return; }   // manual: the picker decides
  const slots = feed2On ? 2 : 1;
  const locked = feedLocks.slice(0, slots)
    .map((on, i) => (on ? i : null)).filter((i) => i !== null).join(",");
  const { body } = await api(`/api/feeds?slots=${slots}&locked=${locked}`);
  if (!body || !Array.isArray(body.slots)) return;
  const [primary, secondary] = body.slots;
  if (primary && primary.camera && watchableCam(primary.camera)
      && $("live-camera").value !== primary.camera) {
    $("live-camera").value = primary.camera;
    boostCam(primary.camera);      // keep the room we follow to actively detecting
    updateLiveView(isRunning);
  }
  // Keep the (disabled) second picker showing what that feed is actually on.
  if (secondary && secondary.camera && watchableCam(secondary.camera)) {
    $("live-camera2").value = secondary.camera;
  }
  updateSecondFeed(secondary);
}

// ---- per-section header popups (#117) --------------------------------------
// One pattern, reused by every section-header button: ⚙ opens that section's
// settings, ❓ opens its explanation. Anchored (not modal) so a knob can be
// twisted — or the help read — while watching the feed behind it. Keeping the
// long help behind a button is the point: it stops explanation text padding out
// the scroll for someone who already knows the app.
function closeGearPopups(keep) {
  for (const pop of document.querySelectorAll(".gear-popup")) {
    if (pop !== keep) pop.classList.add("hidden");
  }
  for (const btn of document.querySelectorAll(".popup-btn")) {
    btn.setAttribute("aria-expanded",
      String(!!keep && btn.dataset.popup === keep.id));
  }
}

function wireGearPopups() {
  for (const btn of document.querySelectorAll(".popup-btn")) {
    btn.setAttribute("aria-expanded", "false");
    btn.onclick = (e) => {
      e.stopPropagation();
      const pop = $(btn.dataset.popup);
      if (!pop) return;
      const opening = pop.classList.contains("hidden");
      closeGearPopups(opening ? pop : null);   // only one open at a time
      pop.classList.toggle("hidden", !opening);
    };
  }
  for (const btn of document.querySelectorAll(".gear-close")) {
    btn.onclick = () => closeGearPopups(null);
  }
  // Dismissible: click anywhere outside, or Escape.
  document.addEventListener("click", (e) => {
    if (!e.target.closest(".gear-popup") && !e.target.closest(".popup-btn")) {
      closeGearPopups(null);
    }
  });
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") closeGearPopups(null);
  });
}

const camNames = () =>
  Array.from($("live-camera").options).map((o) => o.value).filter(Boolean);

function updateSecondFeed(slot) {
  const wrap = $("feed2-wrap"), img = $("live-img2"), label = $("feed2-label");
  const primaryCam = $("live-camera").value || "";
  let pick = null;
  if (followOn) {
    pick = slot && slot.camera ? slot : null;
  } else {
    // Follow off: the user steers this feed from its own picker — which bypasses
    // the router, so the "two feeds are never the same camera" rule has to be
    // enforced here too (#116). Nudge the picker to a different room rather than
    // silently mirroring the main feed.
    let cam = $("live-camera2").value || "";
    if (cam && cam === primaryCam) {
      cam = camNames().find((n) => n !== primaryCam) || "";
      $("live-camera2").value = cam;
    }
    pick = cam ? { camera: cam, source: "manual" } : null;
  }
  // Belt and braces: whatever chose it, this feed never mirrors the main one.
  if (pick && pick.camera === primaryCam) pick = null;
  const show = feed2On && isRunning && $("live-enabled").checked
    && pick && pick.camera && watchableCam(pick.camera);
  if (!show) {
    if (feed2Cam !== null) { feed2Cam = null; }
    img.removeAttribute("src");
    // Say why there's nothing to show rather than leaving a blank slot.
    const oneCam = feed2On && isRunning && camNames().length < 2;
    if (oneCam) {
      wrap.classList.remove("hidden");
      label.textContent = "A second feed needs a second camera.";
    } else {
      wrap.classList.add("hidden");
    }
    return;
  }
  wrap.classList.remove("hidden");
  label.textContent = pick.source === "last-seen"
    ? `${pick.camera} — where she came from`
    : pick.source === "manual" ? pick.camera : `${pick.camera} — cat here now`;
  if (pick.camera !== feed2Cam) {          // only on a real change
    img.src = `/api/stream?camera=${encodeURIComponent(pick.camera)}&ts=${Date.now()}`;
    feed2Cam = pick.camera;
  }
}

// ---- cat sightings ("show cat") --------------------------------------------
// Rotation: when Show-cat is engaged and >1 camera is seeing a cat, cycle the
// live feed between those rooms. `catRotate` is armed by showCat() and disarmed
// when the user picks a camera manually.
let catRotate = false, catCams = [], catRotateIdx = 0;

const watchableCam = (name) =>
  Array.from($("live-camera").options).some((o) => o.value === name);

// Ask the loop to run that camera's detector continuously for a short window, so
// the live feed keeps boxing the cat (even a still one) while the user looks.
function boostCam(name) {
  if (name) api("/api/cats/boost", postJSON({ camera: name }));
}

async function loadCats() {
  const { body } = await api("/api/cats");
  if (!body) return;
  if (catRotate) catCams = (body.cameras || []).filter(watchableCam);
  const btn = $("show-cat"), label = $("show-cat-label");
  const n = (body.cameras || []).length;
  if (body.present) {
    btn.classList.add("detecting");
    label.textContent = n > 1 ? `Cats in ${n} rooms — show me!` : "Cat spotted — show me!";
  } else { btn.classList.remove("detecting"); label.textContent = "Show me the cat!"; }
  const box = $("cat-last");
  if (!body.last) {
    // Don't bail here: the rest of the panel still needs clearing. Returning early
    // was why "Clear log" appeared to only wipe the card above — the sightings list
    // kept rendering its stale rows because renderSightings() never ran.
    box.innerHTML = '<p class="muted">No cats seen yet.</p>';
    renderSightings([]);
    if ($("scan-last")) $("scan-last").textContent = "";
    return;
  }
  const s = body.last;
  const where = (s.zone || s.region) ? ` — ${esc(s.zone || s.region)}` : "";
  const cam = s.camera ? ` on <strong>${esc(s.camera)}</strong>` : "";
  const thumb = s.image
    ? `<img class="cat-thumb" src="/snapshots/${s.image}" alt="last cat sighting"
         title="click to enlarge" style="cursor:zoom-in" onclick="zoomImg(this.src)"/>`
    : "";
  const today = `${body.today} sighting${body.today === 1 ? "" : "s"} today`;
  // The time-of-day prior (#68): where the cats usually are around now — a hint
  // from history for "where should I look first", never a tracked state.
  const likely = (body.likely || []).slice(0, 3)
    .map((l) => `${esc(l.camera)} (${l.weight})`).join(" · ");
  const likelyLine = likely
    ? `<div class="muted" style="font-size:12px">Usually around now: ${likely}</div>` : "";
  box.innerHTML = `${thumb}<div>
      <div><strong>Last seen</strong> ${fmtTime(s.ts)}${cam}${where}
        <span class="muted">(score ${s.score})</span></div>
      <div class="muted">${today}</div>${likelyLine}</div>`;
  renderSightings(body.recent || []);
  // #102: a single glanceable still-scan heartbeat under "Check for still cat".
  const sl = $("scan-last");
  if (sl) {
    const ls = body.last_scan;
    sl.textContent = ls
      ? `Still scan: ${ls.ago_s < 90 ? Math.round(ls.ago_s) + "s" : Math.round(ls.ago_s / 60) + "m"} ago on ${ls.camera} — ${ls.found ? "cat found" : "no cat"}`
      : "";
  }
}

// How many sightings the log renders. The list is height-capped and scrolls, so
// this is "how deep the history goes", not "how many fit on screen" — raise it
// (and the server's /api/cats?limit=) to keep more.
const SIGHTINGS_SHOWN = 20;

// #93: the cats-only sightings log — every entry tagged with HOW it was found
// (motion / still-scan / track / find / zoom+yolo…), images open in the lightbox.
function renderSightings(recent) {
  const box = $("cat-sightings");
  if (!box) return;
  if (!recent.length) { box.innerHTML = '<p class="muted">No sightings yet.</p>'; return; }
  // Fixed columns, not an inline run of spans: every value lands in the same place
  // on every row, so the list scans vertically instead of looking ragged. Top line
  // is when + where-camera; bottom line is the detail (spot · how · confidence).
  box.innerHTML = recent.slice(0, SIGHTINGS_SHOWN).map((r) => {
    const spot = r.zone || r.region || "";
    const src = r.source || "yolo";
    const score = Number(r.score);
    const img = r.image
      ? `<img class="sighting-thumb" src="/snapshots/${r.image}" alt="sighting"
           title="click to enlarge" onclick="zoomImg(this.src)"/>`
      : `<span class="sighting-thumb sighting-thumb-empty" aria-hidden="true"></span>`;
    return `<div class="sighting">
      ${img}
      <div class="sighting-meta">
        <div class="sighting-when"><strong>Seen at</strong> ${fmtTime(r.ts)}${
          r.camera ? ` on <strong>${esc(r.camera)}</strong>` : ""}</div>
        <div class="sighting-sub">
          <span title="${esc(spot)}">${esc(spot)}</span>
          <span title="how this sighting was detected">${esc(src)}</span>
          <span>${Number.isFinite(score) ? score.toFixed(2) : esc(String(r.score ?? ""))}</span>
        </div>
      </div>
    </div>`;
  }).join("");
}

function findFeedback(msg) {
  // #103: persistent, readable feedback for why find-my-cat navigated — a
  // status line that stays put, not a label that flashes too fast to read.
  const el = $("find-feedback");
  if (el) el.textContent = msg || "";
}

async function showCat() {
  // #92: active scan first, when enabled — search, don't just jump.
  if ($("find_scan") && $("find_scan").checked) {
    $("show-cat-label").textContent = "Scanning…";
    findFeedback("Scanning watched cameras…");
    const { ok, body } = await api("/api/cats/find", postJSON({}));
    $("show-cat-label").textContent = "Show me the cat!";
    if (ok && body && body.best) {
      const hit = (body.results || []).find((r) => r.camera === body.best) || {};
      const where = hit.where ? ` (${hit.where})` : "";
      findFeedback(`✅ Found a cat on ${body.best}${where} — score ${(hit.score || 0).toFixed(2)}. Showing it.`);
      const sel = $("live-camera");
      if (watchableCam(body.best)) sel.value = body.best;
      await loadCats();
      $("live-enabled").checked = true;
      if (liveOn) liveOn = false;
      await refreshStatus();
      document.getElementById(isRunning ? "live-stage" : "cat-last")
        .scrollIntoView({ behavior: "smooth", block: "center" });
      return;
    }
    const scanned = ok && body ? (body.results || []).length : 0;
    findFeedback(ok
      ? `No cat found on ${scanned} watched camera${scanned === 1 ? "" : "s"} — showing the last sighting instead.`
      : `Scan failed: ${(body && body.error) || "unknown error"} — showing the last sighting.`);
    // fall through to the jump-to-last behavior below
  }
  return showCatJump();
}

async function showCatJump() {
  const { body } = await api("/api/cats");
  const sel = $("live-camera");
  // Prefer a camera seeing a cat right now; fall back to the last sighting's camera.
  const present = (body && body.cameras) || [];
  const last = body && body.last && body.last.camera;
  const pick = present.find(watchableCam)
    || (last && watchableCam(last) ? last : null);
  if (pick) sel.value = pick;
  // Arm rotation only if more than one room has a cat right now.
  catCams = present.filter(watchableCam);
  catRotate = catCams.length > 1;
  catRotateIdx = Math.max(0, catCams.indexOf(pick));
  boostCam(pick);   // keep the jumped-to camera actively detecting so the box shows
  await loadCats();
  $("live-enabled").checked = true;
  if (liveOn) liveOn = false;          // force the feed to re-point at the chosen camera
  await refreshStatus();
  const target = isRunning ? "live-stage" : "cat-last";
  $(target).scrollIntoView({ behavior: "smooth", block: "center" });
}

// Cycle the live feed between rooms that currently have a cat (when armed).
function rotateCatFeed() {
  // Follow mode owns the feed while it's on — the two must not fight (#113).
  if (followOn) return;
  if (!catRotate || !liveOn || catCams.length < 2) return;
  catRotateIdx = (catRotateIdx + 1) % catCams.length;
  const name = catCams[catRotateIdx];
  if (!watchableCam(name)) return;
  $("live-camera").value = name;
  boostCam(name);                      // keep the room we rotate to actively detecting
  liveOn = false;                      // force updateLiveView to re-point the stream
  updateLiveView(isRunning);
}

// ---- test detection (photo / video) ----------------------------------------
let testSession = null, testFrameIdx = 0, testTimer = null, testTouched = false;

function testSettings() {
  const n = (id) => Number($(id).value);
  return {
    model: $("t_model").value,
    person_confidence: n("t_person_confidence"),
    cat_confidence: n("t_cat_confidence"),
    locator_classes: $("t_dog").checked ? ["cat", "dog"] : ["cat"],
    label_floor: n("t_label_floor"),
    gamma: n("t_gamma"),
    brightness: n("t_brightness"),
    contrast: n("t_contrast"),
    saturation: n("t_saturation"),
    tiling: $("t_tiling").value,
    tile_overlap: n("t_tile_overlap"),
    accelerator: $("t_accelerator").value,
  };
}

function testRangeLabels() {
  const s = testSettings();
  $("t_person_confidence_v").textContent = s.person_confidence.toFixed(2);
  $("t_cat_confidence_v").textContent = s.cat_confidence.toFixed(2);
  $("t_label_floor_v").textContent = s.label_floor.toFixed(2);
  $("t_gamma_v").textContent = s.gamma.toFixed(1);
  $("t_brightness_v").textContent = (s.brightness > 0 ? "+" : "") + s.brightness;
  $("t_contrast_v").textContent = s.contrast.toFixed(2);
  $("t_saturation_v").textContent = s.saturation.toFixed(2);
  $("t_tile_overlap_v").textContent = s.tile_overlap.toFixed(2);
}

// Seed the controls from a settings object (camera defaults or global config).
function setTestControls(v) {
  v = v || {};
  if (v.model || v.detector_model) $("t_model").value = v.model || v.detector_model;
  const set = (id, val, dflt) => { $(id).value = (val === undefined || val === null) ? dflt : val; };
  set("t_person_confidence", v.person_confidence, 0.5);
  set("t_cat_confidence", v.cat_confidence, 0.5);
  $("t_dog").checked = Array.isArray(v.locator_classes) && v.locator_classes.includes("dog");
  set("t_label_floor", v.label_floor, 0.55);
  set("t_gamma", v.gamma, 1.0);
  set("t_brightness", v.brightness, 0);
  set("t_contrast", v.contrast, 1.0);
  set("t_saturation", v.saturation, 1.0);
  if (v.cat_scan_tiling) $("t_tiling").value = v.cat_scan_tiling;
  set("t_tile_overlap", v.cat_scan_tile_overlap, 0.2);
  if (v.accelerator) $("t_accelerator").value = v.accelerator;
  testRangeLabels();
}

function scheduleTestDetect() {
  testTouched = true;
  testRangeLabels();
  if (!testSession) return;
  clearTimeout(testTimer);
  testTimer = setTimeout(runTestDetection, 250);   // debounce slider drags
}

async function runTestDetection() {
  if (!testSession) return;
  $("test-note").textContent = "Detecting…";
  const { ok, body } = await api("/api/test/detect", postJSON({
    id: testSession.id, frame_index: testFrameIdx, settings: testSettings(),
  }));
  if (!ok || !body || body.error) {
    const msg = (body && body.error) || "Detection failed.";
    $("test-note").textContent = msg;
    if (/expired/i.test(msg)) testSession = null;
    return;
  }
  const img = $("test-img");
  img.src = body.annotated;
  img.classList.remove("hidden");
  $("test-placeholder").classList.add("hidden");
  renderTestDetections(body.detections);
  $("test-infer").textContent = body.inference_ms != null
    ? `Inference: ${body.inference_ms} ms` : "";
  $("test-note").textContent = "";
}

function renderTestDetections(list) {
  const box = $("test-detections");
  if (!list || !list.length) {
    box.innerHTML = '<span class="muted">No person or cat detected — try a lower confidence, a bigger input size, or lift the gamma.</span>';
    return;
  }
  box.innerHTML = list.map((d) => {
    const cls = d.label === "person" ? "det-person" : (d.label === "cat" ? "det-cat" : "det-other");
    return `<span class="det-chip ${cls}">${esc(d.label)} ${Number(d.score).toFixed(2)}</span>`;
  }).join("");
}

function renderFilmstrip() {
  const film = $("test-film");
  if (!testSession || testSession.count <= 1) { film.classList.add("hidden"); film.innerHTML = ""; return; }
  film.classList.remove("hidden");
  film.innerHTML = testSession.thumbs.map((t, i) =>
    `<img class="film-thumb ${i === testFrameIdx ? "sel" : ""}" data-i="${i}" src="${t}" alt="frame ${i + 1}" />`).join("");
  film.querySelectorAll(".film-thumb").forEach((el) => {
    el.onclick = () => { testFrameIdx = Number(el.dataset.i); renderFilmstrip(); runTestDetection(); };
  });
}

let lastTestFile = null;     // keep the uploaded file so a video can be re-extracted at N frames

async function uploadTest(file, frames) {
  if (!file) return;
  lastTestFile = file;
  $("test-note").textContent = "Uploading…";
  const fd = new FormData(); fd.append("file", file);
  if (frames) fd.append("frames", String(frames));
  const { ok, body } = await api("/api/test/upload", { method: "POST", body: fd });
  if (!ok || !body || body.error) { $("test-note").textContent = (body && body.error) || "Upload failed."; return; }
  testSession = body; testSession.name = file.name; testFrameIdx = 0;
  $("test-note").textContent = body.kind === "video" ? `${body.count} frames extracted — pick one below, or benchmark them all.` : "";
  renderVideoTools();
  renderFilmstrip();
  runTestDetection();
}

// Controls shown only for a video: choose how many frames to extract (#38) and
// benchmark every extracted frame into the cross-image summary.
function renderVideoTools() {
  const tools = $("test-video-tools");
  if (!tools) return;
  const isVideo = testSession && testSession.kind === "video";
  tools.classList.toggle("hidden", !isVideo);
  if (!isVideo) return;
  $("test-frames").value = testSession.count;
  $("test-bench-all").textContent = `📊 Benchmark all ${testSession.count} frames`;
}

function populateTestSaveTargets() {
  const sel = $("test-save-target");
  if (!sel) return;
  const prev = sel.value;
  const opts = ['<option value="__global__">Global defaults (new cameras)</option>'];
  for (const c of cameras) opts.push(`<option value="${esc(c.name)}">${esc(c.name)}</option>`);
  sel.innerHTML = opts.join("");
  if (Array.from(sel.options).some((o) => o.value === prev)) sel.value = prev;
}

async function saveTestSettings() {
  const s = testSettings();
  const target = $("test-save-target").value;
  $("test-note").textContent = "Saving…";
  let ok;
  if (target === "__global__") {
    ({ ok } = await api("/api/config", postJSON({
      detector_model: s.model, accelerator: s.accelerator,
      person_confidence: s.person_confidence, cat_confidence: s.cat_confidence,
      locator_classes: s.locator_classes, label_floor: s.label_floor,
      gamma: s.gamma, brightness: s.brightness, contrast: s.contrast, saturation: s.saturation,
      cat_scan_tiling: s.tiling, cat_scan_tile_overlap: s.tile_overlap,
    })));
    await loadConfig();
  } else {
    const cam = cameras.find((c) => c.name === target);
    if (!cam) { $("test-note").textContent = "Pick a camera to save to."; return; }
    ({ ok } = await api("/api/cameras/saved", postJSON({
      name: cam.name, url: cam.url, model: s.model, accelerator: s.accelerator,
      person_confidence: s.person_confidence, cat_confidence: s.cat_confidence,
      locator_classes: s.locator_classes, label_floor: s.label_floor,
      gamma: s.gamma, brightness: s.brightness,
      contrast: s.contrast, saturation: s.saturation,
      cat_scan_tiling: s.tiling, cat_scan_tile_overlap: s.tile_overlap,
    })));
    await loadCamerasList();
  }
  $("test-note").textContent = ok ? "Saved ✓" : "Save failed";
  populateTestSaveTargets();
  setTimeout(() => { if ($("test-note").textContent === "Saved ✓") $("test-note").textContent = ""; }, 2500);
}

// ---- benchmark (sweep models × tiling, shareable report) -------------------
async function loadBenchOptions() {
  const { body } = await api("/api/test/benchmark/models");
  if (!body) return;
  const box = (id, items) => $(id).innerHTML = items.map((v) =>
    `<label class="checkbox"><input type="checkbox" value="${esc(v)}" checked/> ${esc(v)}</label>`).join("");
  box("bench-models", body.models);
  box("bench-tilings", body.tilings);
}

const benchChecked = (id) =>
  Array.from($(id).querySelectorAll("input:checked")).map((el) => el.value);

// The tester controls the benchmark now applies uniformly to every run (#44).
function benchControls() {
  const num = (id) => Number($(id).value);
  return {
    cat_threshold: num("t_cat_confidence"), accelerator: $("t_accelerator").value,
    tile_overlap: num("t_tile_overlap"), person_confidence: num("t_person_confidence"),
    gamma: num("t_gamma"), brightness: num("t_brightness"),
    contrast: num("t_contrast"), saturation: num("t_saturation"),
  };
}

async function runBenchmark() {
  if (!testSession) { $("bench-note").textContent = "Upload a photo first."; return; }
  const models = benchChecked("bench-models"), tilings = benchChecked("bench-tilings");
  if (!models.length || !tilings.length) { $("bench-note").textContent = "Pick a model and a tiling."; return; }
  const n = models.length * tilings.length;
  $("bench-run").disabled = true;
  $("bench-note").textContent = `Benchmarking ${n} run${n === 1 ? "" : "s"}… (4×4 / large models can take a while)`;
  const { ok, body } = await api("/api/test/benchmark", postJSON({
    id: testSession.id, frame_index: testFrameIdx, models, tilings,
    name: testSession.name || "uploaded image", ...benchControls(),
  }));
  $("bench-run").disabled = false;
  if (!ok || !body || body.error) { $("bench-note").textContent = (body && body.error) || "Benchmark failed."; return; }
  $("bench-note").textContent = "";
  $("bench-results").classList.remove("hidden");
  $("bench-html").href = body.html_url;
  const xl = $("bench-xlsx"), xlNote = $("bench-xlsx-note");
  if (body.xlsx_url) {
    xl.href = body.xlsx_url; xl.classList.remove("disabled"); xl.title = "";
    xlNote.classList.add("hidden"); xlNote.textContent = "";
  } else {
    // Don't leave a dead control: visibly disable it and say why + how to fix.
    xl.removeAttribute("href"); xl.classList.add("disabled");
    const why = body.xlsx_error || "XLSX export needs the 'openpyxl' package — pip install openpyxl";
    xl.title = why;
    xlNote.textContent = why; xlNote.classList.remove("hidden");
  }
  renderBenchTable(body.runs, body.meta);
}

// Model name for display. (Models are all YOLO variants whose size is their export,
// so the name stands on its own; the legacy "@size" split is a harmless no-op now.)
function displayModel(model) { return String(model).split("@")[0]; }

// Enlarge a thumbnail in-page. Navigating a new tab to a data: URL is blocked by
// browsers (lands on about:blank), so we use a lightbox overlay instead. #30
function zoomImg(src) {
  const lb = $("img-lightbox");
  if (!lb) { window.open(src, "_blank"); return; }
  $("img-lightbox-img").src = src;
  lb.classList.remove("hidden");
}

function renderBenchTable(runs, meta) {
  const rows = runs.map((r) => `<tr>
    <td><img class="bench-thumb" src="${r.thumb}" alt="" title="click to enlarge" onclick="zoomImg(this.src)"/></td>
    <td>${esc(displayModel(r.model))}</td><td>${r.size}</td><td>${esc(r.tiling)}</td>
    <td><strong>${r.combined_score.toFixed(2)}</strong></td>
    <td>${r.cat_score.toFixed(2)}</td><td>${r.dog_score.toFixed(2)}</td>
    <td>${r.detected ? "✓" : "·"}</td><td>${Math.round(r.inference_ms)} ms</td></tr>`).join("");
  // #90: a fallback must be visible in the report — never fallback numbers
  // silently labelled as the requested accelerator.
  const fb = runs.find((r) => r.fallback);
  const ranLine = fb
    ? ` · <span class="warn">ran on ${esc(fb.ran_on)} (${esc(fb.fallback)})</span>`
    : (runs[0] && runs[0].ran_on && runs[0].ran_on !== meta.accelerator
       ? ` · ran on ${esc(runs[0].ran_on)}` : "");
  $("bench-table").innerHTML =
    `<p class="muted">Best cat-or-dog score first · cat threshold ${meta.cat_threshold.toFixed(2)} · ${esc(meta.accelerator)}${ranLine}</p>
     <table class="bench-tbl"><tr><th>frame</th><th>model</th><th>size</th><th>tiling</th>
     <th>cat-or-dog</th><th>cat</th><th>dog</th><th>found?</th><th>time</th></tr>${rows}</table>`;
}

// ---- batch benchmark (several images → cross-image summary) ----------------
let batchItems = [];   // [{id, name, frame_index, has_cat}]

async function addBatchFiles(files) {
  if (!files || !files.length) return;
  $("bench-batch-note").textContent = `Uploading ${files.length}…`;
  for (const f of files) {
    const fd = new FormData(); fd.append("file", f);
    const { ok, body } = await api("/api/test/upload", { method: "POST", body: fd });
    if (ok && body && body.id) {
      batchItems.push({ id: body.id, name: f.name, frame_index: 0, has_cat: true });
    }
  }
  $("bench-batch-note").textContent = "";
  renderBatchList();
}

function renderBatchList() {
  const box = $("bench-batch-list");
  $("bench-batch-run").disabled = batchItems.length === 0;
  if (!batchItems.length) { box.innerHTML = ""; return; }
  box.innerHTML = batchItems.map((it, i) => `<div class="batch-row">
    <span class="grow">${esc(it.name)}</span>
    <label class="inline" title="Untick for an empty-room control frame (false-positive check)">
      <input type="checkbox" data-i="${i}" class="batch-cat" ${it.has_cat ? "checked" : ""}/> cat present</label>
    <button type="button" class="link batch-rm" data-i="${i}">remove</button>
  </div>`).join("");
  box.querySelectorAll(".batch-cat").forEach((el) => {
    el.onchange = () => { batchItems[Number(el.dataset.i)].has_cat = el.checked; };
  });
  box.querySelectorAll(".batch-rm").forEach((el) => {
    el.onclick = () => { batchItems.splice(Number(el.dataset.i), 1); renderBatchList(); };
  });
}

const BENCH_RECOMMENDED_IMAGES = 12;     // soft cap — warn past this, don't block (#37)
let activeBatchId = null;                // the running batch's id, for abort (#37)

function newBatchId() {
  try { return crypto.randomUUID(); }
  catch (e) { return "b-" + Date.now() + "-" + Math.random().toString(16).slice(2); }
}

// Shared runner for the multi-image batch AND the video "benchmark all frames" path.
async function runBatchOf(items, noteEl) {
  if (!items.length) { noteEl.textContent = "Add at least one image."; return; }
  const models = benchChecked("bench-models"), tilings = benchChecked("bench-tilings");
  if (!models.length || !tilings.length) { noteEl.textContent = "Pick a model and a tiling in sweep options above."; return; }
  const runs = items.length * models.length * tilings.length;
  if (items.length > BENCH_RECOMMENDED_IMAGES) {       // soft cap: show cost, confirm
    const estMin = Math.max(1, Math.round(runs * 0.6 / 60));
    if (!confirm(`${items.length} images × ${models.length * tilings.length} = ${runs} runs (rough est. ~${estMin} min on CPU). Run anyway?`)) return;
  }
  const batchId = newBatchId(); activeBatchId = batchId;
  $("bench-batch-run").disabled = true;
  $("bench-batch-abort").classList.remove("hidden");
  noteEl.textContent = `Benchmarking ${items.length} images (~${runs} runs)… you can abort.`;
  const vlmEl = $("bench-vlm");
  const { ok, body } = await api("/api/test/benchmark/batch", postJSON({
    items, models, tilings, batch_id: batchId, ...benchControls(),
    run_vlm: !!(vlmEl && vlmEl.checked),
    vlm_model: vlmCurrent().model, vlm_mode: vlmCurrent().mode, vlm_passes: vlmPasses(),
  }));
  activeBatchId = null;
  $("bench-batch-run").disabled = false;
  $("bench-batch-abort").classList.add("hidden");
  if (!ok || !body || body.error) { noteEl.textContent = (body && body.error) || "Batch failed."; return; }
  // Surface a partial run prominently — never let it look like a complete one (#40).
  if (body.cancelled) {
    noteEl.textContent = `Stopped early — summarised ${body.ran} of ${body.requested} images.`;
  } else if (body.skipped > 0) {
    noteEl.textContent = `⚠ Ran ${body.ran} of ${body.requested} — ${body.skipped} upload(s) were no longer available. Re-add them and run again.`;
  } else {
    noteEl.textContent = "";
  }
  $("bench-batch-results").classList.remove("hidden");
  $("bench-summary-html").href = body.summary_url;
  const zip = $("bench-summary-zip"); if (zip) zip.href = body.zip_url;
  renderBatchSummary(body.configs, body.meta, body.images, body.vlm);
  $("bench-batch-results").scrollIntoView({ behavior: "smooth", block: "nearest" });
}

function runBatch() { return runBatchOf(batchItems, $("bench-batch-note")); }

async function abortBatch() {
  if (!activeBatchId) return;
  $("bench-batch-note").textContent = "Stopping after the current image…";
  await api("/api/test/benchmark/cancel", postJSON({ batch_id: activeBatchId }));
}

// Video → "benchmark all frames": every extracted frame becomes a batch entry,
// reusing the same cross-image summary path (#38).
function benchmarkAllFrames() {
  if (!testSession || testSession.count < 1) return;
  const items = [];
  for (let i = 0; i < testSession.count; i++) {
    items.push({ id: testSession.id, frame_index: i,
                 name: `${testSession.name || "clip"} #${i + 1}`, has_cat: true });
  }
  $("bench-batch").open = true;
  runBatchOf(items, $("bench-batch-note"));
}

// ---- Cat-presence (VLM / moondream) tester (#48) ---------------------------
let vlmSession = null;

let VLM_CHOICES = [];   // [{value, mode, model, label, off_device}] — from /api/vlm/status

// The selected (mode, model) pair from the choice dropdown.
function vlmCurrent() {
  const sel = $("vlm-choice");
  const c = VLM_CHOICES.find((x) => x.value === (sel && sel.value)) || VLM_CHOICES[0] || {};
  return { mode: c.mode || "local", model: c.model || "moondream2", off_device: !!c.off_device };
}

// Passes per image for the majority vote (#60); clamped to the input's 1–9 range.
function vlmPasses() {
  const n = Number($("vlm-passes") && $("vlm-passes").value) || 3;
  return Math.max(1, Math.min(9, Math.round(n)));
}

// Show/hide the "cloud sends images off-device" warning for the current choice.
function updateVlmOffDevice() {
  const w = $("vlm-offdevice");
  if (w) w.classList.toggle("hidden", !vlmCurrent().off_device);
}

// Prompts are MODEL-specific (#74): the moondream2 default (P6) produced ~100% FP on
// moondream3. On a model switch: an untouched (or other-model-default) prompt swaps to
// the new model's default; a custom prompt is kept but flagged with a warning.
let VLM_MODEL_PROMPTS = {};
function onVlmChoiceChange() {
  updateVlmOffDevice();
  const p = $("vlm-prompt"), warn = $("vlm-prompt-warn");
  if (!p) return;
  const model = vlmCurrent().model;
  const mine = VLM_MODEL_PROMPTS[model] || "";
  const others = Object.entries(VLM_MODEL_PROMPTS)
    .filter(([m]) => m !== model).map(([, v]) => v);
  const cur = p.value.trim();
  if (!cur || others.includes(cur)) {
    p.value = mine;
    if (warn) warn.classList.add("hidden");
  } else if (warn) {
    warn.classList.toggle("hidden", cur === mine);
  }
}

async function loadVlmStatus() {
  const { body } = await api("/api/vlm/status");
  if (!body) return;
  VLM_MODEL_PROMPTS = body.model_prompts || {};
  if ($("vlm-prompt") && !$("vlm-prompt").value) $("vlm-prompt").value = body.default_prompt || "";
  VLM_CHOICES = Array.isArray(body.choices) ? body.choices : [];
  const sel = $("vlm-choice");
  if (sel && VLM_CHOICES.length) {
    sel.innerHTML = VLM_CHOICES.map((c) => `<option value="${esc(c.value)}">${esc(c.label)}</option>`).join("");
    if (body.default_choice) sel.value = body.default_choice;
    sel.onchange = onVlmChoiceChange;      // off-device + model-specific prompt (#74)
  }
  updateVlmOffDevice();
  const pin = $("vlm-passes");
  if (pin) {
    if (body.max_passes) pin.max = body.max_passes;
    if (body.default_passes && !pin.dataset.touched) pin.value = body.default_passes;
    pin.onchange = () => { pin.dataset.touched = "1"; };
  }
  // API-key status: a missing key is the field's own documentation, never a cryptic 401.
  const ks = $("vlm-key-status");
  if (ks) ks.textContent = body.has_api_key ? "✅ key set" : "⚠ no key set — paste yours to enable the VLM";
  // Escalation (#66): remember the live-camera gate + reflect it in the settings toggle.
  VLM_ESCALATION_ENABLED = !!(body.escalation && body.escalation.enabled);
  const et = $("vlm-escalation-toggle");
  if (et) et.checked = VLM_ESCALATION_ENABLED;
  if (!body.available) {
    const note = $("vlm-unavailable");
    note.textContent = "moondream isn’t installed — the tester is ready, but local Run needs: pip install moondream, a supported GPU (CUDA/Ampere or Apple Silicon), and an API key. Cloud mode needs only the package + key. It’ll work once that’s in place.";
    note.classList.remove("hidden");
  }
}

async function saveVlmKey() {
  const inp = $("vlm-key");
  const key = (inp && inp.value || "").trim();
  if (!key) { $("vlm-key-status").textContent = "Paste a key first."; return; }
  const { ok, body } = await api("/api/config", postJSON({ moondream_api_key: key }));
  if (ok && body) {
    inp.value = "";
    $("vlm-key-status").textContent = "✅ key saved";
  } else {
    $("vlm-key-status").textContent = "Save failed.";
  }
}

async function uploadVlm(file) {
  if (!file) return;
  $("vlm-note").textContent = "Uploading…";
  const fd = new FormData(); fd.append("file", file);
  const { ok, body } = await api("/api/test/upload", { method: "POST", body: fd });
  if (!ok || !body || body.error) { $("vlm-note").textContent = (body && body.error) || "Upload failed."; return; }
  vlmSession = body; vlmSession.name = file.name;
  $("vlm-note").textContent = body.kind === "video" ? `${body.count} frames — pick one below.` : "";
  const pick = $("vlm-frame-pick");
  pick.classList.toggle("hidden", body.kind !== "video");
  if (body.kind === "video") {
    $("vlm-frame").max = body.count - 1; $("vlm-frame").value = 0;
    $("vlm-frame-count").textContent = `of ${body.count}`;
  }
  $("vlm-run").disabled = false;
  const er = $("esc-run"); if (er) er.disabled = false;
  const vl = $("vlm-locate"); if (vl) vl.disabled = false;   // detect-mode diagnostic (#75)
  // Temporal check needs several frames — a sampled video, not a single photo (#68).
  const vt = $("vlm-temporal");
  if (vt) vt.disabled = body.kind !== "video";
}

// moondream detect mode (#75): draw boxes where the model THINKS a cat is — the
// diagnostic for false-positive frames (what feature is it locking onto?). Purely
// informational; nothing is recorded (#69: a bare detect region is never trusted).
async function runLocate() {
  if (!vlmSession) { $("vlm-note").textContent = "Upload a photo or video first."; return; }
  const cur = vlmCurrent();
  $("vlm-locate").disabled = true;
  $("vlm-note").textContent = "Asking moondream where…";
  const { ok, body } = await api("/api/vlm/locate", postJSON({
    id: vlmSession.id, frame_index: Number($("vlm-frame").value) || 0,
    model: cur.model, mode: cur.mode,
  }));
  $("vlm-locate").disabled = false;
  if (!ok || !body || body.error) { $("vlm-note").textContent = (body && body.error) || "Locate failed."; return; }
  $("vlm-note").textContent = body.n
    ? `${body.n} region(s) in ${Math.round(body.detect_ms || 0)} ms — diagnostic only, nothing recorded.`
    : (body.note || "");
  $("vlm-result").classList.remove("hidden");
  const img = $("vlm-locate-img");
  if (img) {
    img.classList.remove("hidden");
    img.src = body.annotated;
    img.onclick = () => zoomImg(img.src);
  }
}

// Temporal mosaic (#68): the last few frames tiled into one numbered grid, one
// voted question over it. camera=null → the uploaded video session's frames.
async function runTemporal(camera) {
  const note = $("esc-note") && camera ? $("esc-note") : $("vlm-note");
  const cur = vlmCurrent();
  const payload = { mode: cur.mode, model: cur.model, passes: vlmPasses() };
  if (camera) payload.camera = camera;
  else if (vlmSession) payload.id = vlmSession.id;
  else { note.textContent = "Upload a video first."; return; }
  note.textContent = "Building the mosaic + asking moondream…";
  const { ok, body } = await api("/api/vlm/temporal", postJSON(payload));
  if (!ok || !body || body.error) { note.textContent = (body && body.error) || "Temporal check failed."; return; }
  note.textContent = `${body.n_frames} frames${body.span_s ? ` over ~${body.span_s}s` : ""}` +
    (body.hint_note ? ` — ${body.hint_note}` : "");
  renderVlm(body);                       // verdict + ratio + reasoning, as usual
  const mi = $("vlm-mosaic");
  if (mi) {
    mi.classList.remove("hidden");
    mi.src = body.mosaic; mi.onclick = () => zoomImg(mi.src);
  }
}

// ---- escalation ladder (#66): zoom crops -> YOLO -> VLM detect/confirm ------
async function runEscalation(camera) {
  const note = $("esc-note");
  const payload = {
    use_vlm: !!($("esc-use-vlm") && $("esc-use-vlm").checked),
    settings: {},                                    // defaults; CPU YOLO confirm rung
  };
  if (payload.use_vlm) {
    const cur = vlmCurrent();
    payload.mode = cur.mode; payload.model = cur.model; payload.passes = vlmPasses();
  }
  if (camera) {
    payload.camera = camera;
  } else {
    if (!vlmSession) { note.textContent = "Upload a photo or video first."; return; }
    payload.id = vlmSession.id;
    payload.frame_index = Number($("vlm-frame").value) || 0;
  }
  note.textContent = "Climbing the ladder…";
  $("esc-run").disabled = true;
  const { ok, body } = await api("/api/vlm/escalate", postJSON(payload));
  $("esc-run").disabled = !vlmSession;
  if (!ok || !body || body.error) { note.textContent = (body && body.error) || "Escalation failed."; return; }
  note.textContent = body.note || "";
  renderEscalation(body);
}

function renderEscalation(b) {
  $("esc-results").classList.remove("hidden");
  const badge = $("esc-verdict");
  if (b.found) {
    badge.textContent = `CAT found — ${b.source}${b.ratio ? ` (vote ${b.ratio})` : ""}`;
    badge.className = "vlm-badge vlm-yes";
  } else if (b.probable) {
    // The inference tier: an unconfirmed VLM lead (#69 demotion) or the trail's
    // interior endpoint — never a confirmed sighting, always labelled "probable"
    // with its evidence (server-written note; trail image below).
    badge.textContent = `probably here — ${b.probable.note}`;
    badge.className = "vlm-badge vlm-unparsed";
  } else {
    badge.textContent = "no cat confirmed";
    badge.className = "vlm-badge vlm-no";
  }
  const totalMs = (b.rungs || []).reduce((s, r) => s + (r.ms || 0), 0);
  $("esc-summary").textContent =
    `${(b.hints_used || []).length} hint(s) · ${Math.round(totalMs)} ms total` +
    (b.camera ? ` · camera: ${b.camera}` : "");
  $("esc-rungs").innerHTML =
    "<tr><th>rung</th><th>ran</th><th>crops</th><th>ms</th><th>result</th></tr>" +
    (b.rungs || []).map((r) => `<tr><td style="text-align:left">${esc(r.name)}</td>
      <td>${r.ran ? "✓" : "—"}</td><td>${r.crops}</td><td>${Math.round(r.ms)}</td>
      <td style="text-align:left">${r.found ? "🐱 found" : esc(r.note || (r.ran ? "nothing" : ""))}</td></tr>`).join("");
  $("esc-crops").innerHTML = (b.crops || []).filter((c) => c.image).map((c) =>
    `<img src="${c.image}" title="${esc(c.source_rung)} crop — click to enlarge"
          style="max-width:140px;border-radius:4px;cursor:zoom-in" onclick="zoomImg(this.src)"/>`).join("");
  const ann = $("esc-annotated");
  ann.src = b.annotated || "";
  ann.onclick = () => zoomImg(ann.src);
  const tr = $("esc-trail");
  if (tr) {
    tr.classList.toggle("hidden", !b.trail);
    if (b.trail) { tr.src = b.trail; tr.onclick = () => zoomImg(tr.src); }
  }
}

// Fetch a per-camera diagnostic image (the cat trail or the sighting heat map)
// and open it in the lightbox. Cache-busted: fresh render each call.
async function showCameraImage(path) {
  const cam = $("esc-camera").value;
  const note = $("esc-note");
  const resp = await fetch(`${path}?camera=${encodeURIComponent(cam)}&t=${Date.now()}`);
  if (!resp.ok) {
    const body = await resp.json().catch(() => null);
    note.textContent = (body && body.error) || "Not available.";
    return;
  }
  note.textContent = "";
  const blob = await resp.blob();
  zoomImg(URL.createObjectURL(blob));
}
const showTrail = () => showCameraImage("/api/trail");
const showHeatmap = () => showCameraImage("/api/cats/heatmap");

// Camera picker for live-camera escalation: shown only when the config toggle is
// on AND the loop is running (mirrors the backend's 403/409 gates).
function updateEscalationCameraRow(statusBody) {
  const row = $("esc-camera-row");
  if (!row) return;
  const enabled = !!(VLM_ESCALATION_ENABLED && statusBody && statusBody.running);
  row.hidden = !enabled;
  if (enabled) {
    const sel = $("esc-camera");
    const cams = (statusBody.cameras || []).map((c) => c.name || c).filter(Boolean);
    const prev = sel.value;
    sel.innerHTML = cams.map((n) => `<option value="${esc(n)}">${esc(n)}</option>`).join("");
    if (cams.includes(prev)) sel.value = prev;
  }
}
let VLM_ESCALATION_ENABLED = false;

async function runVlm() {
  if (!vlmSession) { $("vlm-note").textContent = "Upload a photo or video first."; return; }
  $("vlm-run").disabled = true;
  $("vlm-note").textContent = "Asking moondream…";
  const cur = vlmCurrent();
  const { ok, body } = await api("/api/vlm/query", postJSON({
    id: vlmSession.id, frame_index: Number($("vlm-frame").value) || 0,
    prompt: $("vlm-prompt").value, model: cur.model, mode: cur.mode, passes: vlmPasses(),
    reasoning: !!($("vlm-reasoning") && $("vlm-reasoning").checked),   // M3 thinking (#76)
  }));
  $("vlm-run").disabled = false;
  if (!ok || !body || body.error) { $("vlm-note").textContent = (body && body.error) || "Query failed."; return; }
  $("vlm-note").textContent = "";
  renderVlm(body);
}

function renderVlm(b) {
  $("vlm-result").classList.remove("hidden");
  const mi = $("vlm-mosaic");            // stale temporal mosaic hides on a plain query
  if (mi && !b.mosaic) mi.classList.add("hidden");
  const li = $("vlm-locate-img");        // stale locate overlay hides too (#75)
  if (li) li.classList.add("hidden");
  const badge = $("vlm-answer");
  if (!b.parsed || b.answer == null) {
    badge.textContent = "unparsed"; badge.className = "vlm-badge vlm-unparsed";
  } else {
    badge.textContent = b.answer === "yes" ? "CAT: yes" : "CAT: no";
    badge.className = "vlm-badge " + (b.answer === "yes" ? "vlm-yes" : "vlm-no");
  }
  // With multiple passes, append the vote ratio (the honest confidence) + a borderline flag.
  if (b.passes > 1 && b.ratio) {
    const vote = ` · vote ${b.ratio}${b.borderline ? " ⚠ borderline" : " (unanimous)"}`;
    badge.textContent += vote;
  }
  const load = b.load_ms ? ` · model load ${Math.round(b.load_ms)} ms` : "";
  const passInfo = b.passes > 1 ? ` over ${b.passes} passes` : "";
  $("vlm-latency").textContent = `query ${Math.round(b.query_ms)} ms${passInfo}${load}`;
  $("vlm-raw").textContent = (b.raw || "(empty response)") +
    (b.reasoning ? `\n\n[reasoning] ${b.reasoning}` : "");             // M3 thinking (#76)
  $("vlm-meta").textContent = `model: ${b.model} · mode: ${b.mode || "local"} · device: ${b.device}`;
}

function vlmSummaryLine(v) {
  if (!v) return "";
  if (v.error) return `<p class="muted">VLM skipped — ${esc(v.error)}</p>`;
  const s = v.summary;
  const pct = (r, n, d) => r == null ? "—" : `${Math.round(r * 100)}% (${n}/${d})`;
  const dis = (v.disagreements || []).map((d) =>
    `<a href="/api/test/benchmark/${d.slug}.html" target="_blank" rel="noopener">${esc(d.name)}</a> (VLM ${d.vlm}${d.ratio ? " " + esc(d.ratio) : ""} / YOLO ${d.yolo})`).join(" · ");
  const bl = (v.borderline || []).map((b) =>
    `<a href="/api/test/benchmark/${b.slug}.html" target="_blank" rel="noopener">${esc(b.name)}</a> (${esc(b.answer || "tie")} ${esc(b.ratio || "")})`).join(" · ");
  const passNote = (v.passes || 1) > 1 ? ` <span style="font-size:12px">· ${v.passes} passes, majority vote</span>` : "";
  return `<p class="muted"><strong>VLM (${esc(v.model)}):</strong> recall ${pct(s.recall, s.found, s.n_cat)} · false-positive ${pct(s.fp_rate, s.fp, s.n_nocat)}${passNote}</p>`
    + (dis ? `<p class="muted" style="font-size:12px">Disagreements vs best YOLO: ${dis}</p>`
           : (v.summary && (v.disagreements || []).length === 0 ? `<p class="muted" style="font-size:12px">VLM and YOLO agree on every frame.</p>` : ""))
    + (bl ? `<p class="muted" style="font-size:12px">Borderline (split vote — review): ${bl}</p>` : "");
}

function renderBatchSummary(configs, meta, images, vlm) {
  const rows = configs.map((c) => {
    const found = `${c.found}/${c.total}`;
    const fp = meta.n_nocat ? `${c.fp}/${c.fp_total}` : "—";
    return `<tr><td style="text-align:left">${esc(displayModel(c.model))} ${c.size} ${esc(c.tiling)}</td>
      <td>${found}${c.found === c.total && c.total ? " ✓" : ""}</td>
      <td>${c.avg_score.toFixed(2)}</td><td>${Math.round(c.avg_ms)} ms</td><td>${fp}</td></tr>`;
  }).join("");
  const links = (images || []).map((im) =>
    `<a href="${im.report_url}" target="_blank" rel="noopener">${esc(im.name)}${im.has_cat ? "" : " ∅"}</a>`).join(" · ");
  $("bench-summary-table").innerHTML =
    `<p class="muted">${meta.n_images} images (${meta.n_cat} with a cat) · sorted by detection rate · cat threshold ${meta.cat_threshold.toFixed(2)} · ${esc(meta.accelerator)}. Open the summary report for the miss detail and the config×image grid.</p>
     ${vlmSummaryLine(vlm)}
     <table class="bench-tbl"><tr><th>config</th><th>found</th><th>avg conf</th><th>avg ms</th><th>false +</th></tr>${rows}</table>
     <p class="muted" style="font-size:12px">Per-image reports: ${links}</p>`;
}

// ---- standalone batch VLM tester (#54) -------------------------------------
let vlmBatchItems = [];
let vlmBatchId = null;

async function addVlmBatchFiles(files) {
  if (!files || !files.length) return;
  $("vlm-batch-note").textContent = `Uploading ${files.length}…`;
  for (const f of files) {
    const fd = new FormData(); fd.append("file", f);
    const { ok, body } = await api("/api/test/upload", { method: "POST", body: fd });
    if (ok && body && body.id) vlmBatchItems.push({ id: body.id, name: f.name, frame_index: 0, has_cat: true });
  }
  $("vlm-batch-note").textContent = "";
  renderVlmBatchList();
}

function renderVlmBatchList() {
  const box = $("vlm-batch-list");
  $("vlm-batch-run").disabled = vlmBatchItems.length === 0;
  box.innerHTML = vlmBatchItems.map((it, i) => `<div class="batch-row">
    <span class="grow">${esc(it.name)}</span>
    <label class="inline"><input type="checkbox" data-i="${i}" class="vbatch-cat" ${it.has_cat ? "checked" : ""}/> cat present</label>
    <button type="button" class="link vbatch-rm" data-i="${i}">remove</button></div>`).join("");
  box.querySelectorAll(".vbatch-cat").forEach((el) => {
    el.onchange = () => { vlmBatchItems[Number(el.dataset.i)].has_cat = el.checked; };
  });
  box.querySelectorAll(".vbatch-rm").forEach((el) => {
    el.onclick = () => { vlmBatchItems.splice(Number(el.dataset.i), 1); renderVlmBatchList(); };
  });
}

async function runVlmBatch() {
  if (!vlmBatchItems.length) return;
  vlmBatchId = newBatchId();
  $("vlm-batch-run").disabled = true;
  $("vlm-batch-abort").classList.remove("hidden");
  $("vlm-batch-note").textContent = `Running the VLM on ${vlmBatchItems.length} images… you can abort.`;
  const cur = vlmCurrent();
  const { ok, body } = await api("/api/vlm/batch", postJSON({
    items: vlmBatchItems, batch_id: vlmBatchId, model: cur.model, mode: cur.mode, passes: vlmPasses(),
    prompt: $("vlm-prompt").value,                                     // the batch honours the prompt (#72)
    reasoning: !!($("vlm-reasoning") && $("vlm-reasoning").checked),
  }));
  vlmBatchId = null;
  $("vlm-batch-run").disabled = false;
  $("vlm-batch-abort").classList.add("hidden");
  if (!ok || !body || body.error) { $("vlm-batch-note").textContent = (body && body.error) || "VLM batch failed."; return; }
  $("vlm-batch-note").textContent = body.cancelled
    ? `Stopped early — ${body.ran} of ${body.requested} done.`
    : (body.skipped ? `⚠ Ran ${body.ran} of ${body.requested} — ${body.skipped} upload(s) expired.` : "");
  renderVlmBatch(body);
}

async function abortVlmBatch() {
  if (!vlmBatchId) return;
  $("vlm-batch-note").textContent = "Stopping after the current image…";
  await api("/api/test/benchmark/cancel", postJSON({ batch_id: vlmBatchId }));
}

function renderVlmBatch(b) {
  const s = b.summary;
  const pct = (r, n, d) => r == null ? "—" : `${Math.round(r * 100)}% (${n}/${d})`;
  const multi = (b.passes || 1) > 1;
  const rows = (b.verdicts || []).map((v) => {
    const a = v.error ? `<span class="vlm-badge vlm-unparsed">error</span>`
      : (v.answer === "yes" ? `<span class="vlm-badge vlm-yes">yes</span>`
        : v.answer === "no" ? `<span class="vlm-badge vlm-no">no</span>`
          : `<span class="vlm-badge vlm-unparsed">?</span>`);
    const voteCell = multi
      ? `<td>${v.ratio ? esc(v.ratio) + (v.borderline ? " ⚠" : "") : "—"}</td>` : "";
    return `<tr><td style="text-align:left">${esc(v.name)}${v.has_cat ? "" : " ∅"}</td>
      <td>${a}</td>${voteCell}<td style="text-align:left">${esc(v.error || v.reason || "")}</td>
      <td>${Math.round(v.query_ms)} ms</td></tr>`;
  }).join("");
  const voteHdr = multi ? "<th>vote</th>" : "";
  const passNote = multi ? ` · ${b.passes} passes/image (majority vote; ⚠ = split)` : "";
  $("vlm-batch-results").innerHTML =
    `<p class="muted"><strong>${esc(b.model)}:</strong> recall ${pct(s.recall, s.found, s.n_cat)} · false-positive ${pct(s.fp_rate, s.fp, s.n_nocat)}${s.errors ? ` · ${s.errors} error(s)` : ""}${passNote}</p>
     <table class="bench-tbl"><tr><th>image</th><th>cat?</th>${voteHdr}<th>reason</th><th>latency</th></tr>${rows}</table>`;
}

// ---- activity log ----------------------------------------------------------
let lastLogKey = "";

function fmtTime(ts) {
  if (!ts) return "";
  const d = new Date(ts * 1000);
  const time = d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
  const sameDay = d.toDateString() === new Date().toDateString();
  return sameDay ? time : `${d.toLocaleDateString([], { month: "short", day: "numeric" })} ${time}`;
}

async function loadLog() {
  const { body } = await api("/api/log?limit=200");
  const entries = body || [];
  const key = entries.length ? `${entries.length}:${entries[0].ts}` : "0";
  if (key === lastLogKey) return;
  lastLogKey = key;
  const list = $("log-list");
  if (!entries.length) { list.innerHTML = `<p class="muted">No activity yet.</p>`; return; }
  list.innerHTML = "";
  for (const e of entries) {
    const row = document.createElement("div");
    row.className = `log-row log-${e.kind || "info"}`;
    const t = document.createElement("span");
    t.className = "log-time"; t.textContent = fmtTime(e.ts);
    const m = document.createElement("span");
    m.className = "log-msg"; m.textContent = e.message;
    row.append(t, m);
    if (e.image) {
      const a = document.createElement("a");
      a.href = "javascript:void(0)"; a.title = "Click to enlarge";
      a.onclick = () => zoomImg(`/snapshots/${e.image}`);
      const img = document.createElement("img");
      img.className = "log-thumb"; img.src = `/snapshots/${e.image}`; img.alt = "detection snapshot";
      a.appendChild(img); row.appendChild(a);
    }
    list.appendChild(row);
  }
}

// ---- wiring ----------------------------------------------------------------
function wire() {
  const retry = $("init-retry");
  if (retry) retry.onclick = retryInitialLoad;
  // Follow mode / second feed (#113).
  $("follow-cat").onchange = () => {
    followOn = $("follow-cat").checked;
    if (followOn) catRotate = false;           // don't fight the Show-cat rotation
    syncFeedControls();
    renderFeedLocks();
    pollFeeds();
  };
  $("second-feed").onchange = () => {
    feed2On = $("second-feed").checked;
    syncFeedControls();
    renderFeedLocks();
    if (!feed2On) updateSecondFeed(null);
    pollFeeds();
  };
  $("live-camera2").onchange = () => updateSecondFeed(null);
  for (const i of [0, 1]) {
    const btn = $("lock-" + i);
    if (btn) btn.onclick = () => toggleFeedLock(i);
  }
  $("speaker-refresh").onclick = () => loadSpeakers(true);
  $("speaker-select").onchange = updateSpeakerWarning;
  $("use_speech").onchange = applySpeechVisibility;
  $("dice_sides").oninput = updateOdds;
  $("dc").oninput = updateOdds;
  $("quiet-clear").onclick = () => { $("quiet_start").value = ""; $("quiet_end").value = ""; };

  // Camera manager.
  $("camera-add").onclick = () => addCamera();
  $("camera-refresh").onclick = () => loadCameras(true);
  $("camera-local-refresh").onclick = () => loadLocalCameras(true);
  $("camera-select").onchange = (e) => {
    const opt = e.target.selectedOptions[0];
    if (opt && opt.value) addCamera({ name: opt.dataset.name || opt.textContent, url: opt.value });
    e.target.value = "";
  };
  $("camera-local-select").onchange = (e) => {
    const opt = e.target.selectedOptions[0];
    if (opt && opt.value) addCamera({ name: opt.textContent, url: opt.value });
    e.target.value = "";
  };
  // ROI picker.
  wireRoiCanvas();
  $("roi-use").onclick = () => {
    if (zoneEdit) {                          // zone mode (#68): the box becomes a zone
      if (!roi) { $("roi-note").textContent = "Drag a box first."; return; }
      const zname = (prompt("Name this zone (e.g. the couch, kitchen door):", "") || "").trim();
      if (!zname) { $("roi-note").textContent = "Zone needs a name — drag again or Cancel."; return; }
      const isExit = confirm("Is this an exit (a doorway the cat can leave through)?\nOK = exit · Cancel = a normal spot");
      const card = zoneEdit.card;
      const stored = cameras.find((c) => c.name === card.dataset.name) || {};
      const zones = (card._zones !== undefined ? card._zones : (stored.zones || [])).slice();
      zones.push({ name: zname, box: roi.slice(), exit: isExit });
      card._zones = zones;
      card.querySelector("[data-zonelist]").innerHTML = zoneChipsHTML(zones);
      $("roi-editor").classList.add("hidden");
      zoneEdit = null; roiEdit = null;
      // Persist; its loadCamerasList() re-render also wires the new chip's ×.
      card.querySelector("[data-save]").click();
      return;
    }
    if (!roiEdit) return;
    roiEdit.card._roi = roi;
    const note = roiEdit.card.querySelector("[data-roinote]");
    if (note) note.textContent = roi ? `region ${roi[2]}×${roi[3]}px` : "whole frame";
    $("roi-editor").classList.add("hidden");
    roiEdit.card.querySelector("[data-save]").click();   // persist the new region
    roiEdit = null;
  };
  $("roi-clear").onclick = () => { roi = null; drawRoiBox(); $("roi-note").textContent = zoneEdit ? "Drag a box over the spot" : "Whole frame"; };
  $("roi-cancel").onclick = () => { $("roi-editor").classList.add("hidden"); roiEdit = null; zoneEdit = null; };

  $("sound-upload").onchange = async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    const fd = new FormData(); fd.append("file", file);
    const { ok, body } = await api("/api/sounds", { method: "POST", body: fd });
    if (ok) { savedSound = body.saved; await loadSounds(); }
    else alert((body && body.error) || "Upload failed");
  };

  $("test-btn").onclick = async () => {
    await saveConfig();
    $("test-btn").textContent = "Playing…";
    const { ok, body } = await api("/api/test", { method: "POST" });
    $("test-btn").textContent = "▶ Test";
    if (!ok) alert((body && body.error) || "Could not play on the speaker(s).");
  };

  $("log-clear").onclick = async () => {
    await api("/api/log/clear", { method: "POST" });
    lastLogKey = ""; loadLog();
  };

  $("save-btn").onclick = saveConfig;
  $("start-btn").onclick = async () => {
    await saveConfig();
    await api("/api/start", { method: "POST" });
    refreshStatus(); loadLog();
  };
  $("stop-btn").onclick = async () => {
    await api("/api/stop", { method: "POST" });
    refreshStatus(); loadLog();
  };

  // The tester/benchmark Accelerator dropdown is GENERATED from ACCEL_OPTS
  // (#87) — it was a hardcoded HTML list that silently missed tensorrt.
  $("t_accelerator").innerHTML = optsHTML(ACCEL_OPTS, "cpu");

  $("models-provision").onclick = async () => {
    const { body } = await api("/api/models/provision", postJSON({}));
    if (body && body.error) { $("models-provision-note").textContent = body.error; return; }
    $("models-provision").disabled = true;
    $("models-provision-note").textContent = "generating… (minutes; downloads weights on first run)";
    provisionPoll = setInterval(pollProvision, 2000);
  };
  loadModelsAudit();

  $("live-enabled").onchange = () => refreshStatus();
  const tov = $("trail-overlay");
  if (tov) tov.onchange = () => refreshStatus();   // rebuilds the stream URL (0.39.0)
  const lkv = $("lastknown-overlay");
  if (lkv) lkv.onchange = () => refreshStatus();   // rebuilds the stream URL (0.42.0)
  $("live-camera").onchange = () => {
    catRotate = false;
    if (liveOn) { liveOn = false; }
    updateSecondFeed(null);   // #116: move the 2nd feed off a collision with this one
    refreshStatus();
  };
  $("cat_scan_interval").onchange = saveConfig;
  $("live-img").onerror = () => { if (liveOn) { liveOn = false; $("live-img").classList.add("hidden"); } };
  $("smooth_feed").onchange = async () => {
    await api("/api/live/smooth", postJSON({ on: $("smooth_feed").checked }));
    if (liveOn) { liveOn = false; refreshStatus(); }
  };

  // #102 save-behavior audit: every setting on the page auto-saves on change,
  // so the whole GUI behaves one way (change → saved) instead of the top block
  // waiting for the bottom button while the Cat-cam block saved live. Wired with
  // addEventListener so the existing odds/visibility/warning handlers still fire.
  // `change` (not `input`) fires on blur/Enter — no per-keystroke save spam.
  for (const id of ["speaker-select", "sound-select", "use_speech", "speech_text",
                    "dice_sides", "dc", "cooldown_seconds", "pause_during_cooldown",
                    "quiet_start", "quiet_end", "dont_interrupt_playback",
                    "keep_speakers_warm"]) {
    const el = $(id);
    if (el) el.addEventListener("change", saveConfig);
  }

  $("show-cat").onclick = showCat;
  // Global still-scan + find settings (#101/#102) auto-save on change.
  for (const id of ["cat_scan_model", "scan_tiling", "scan_tile_overlap",
                    "scan_frames", "scan_confidence", "find_scan", "find_model",
                    "find_tiling", "find_tile_overlap", "find_confidence",
                    // gear-popup settings (#117) — same auto-save, so they apply live
                    "fusion_debug", "swap_confirm_count",
                    "camera_reuse_cooldown_seconds"]) {
    const el = $(id);
    if (el) el.onchange = saveConfig;
  }
  wireGearPopups();
  $("cats-clear").onclick = async () => { await api("/api/cats/clear", { method: "POST" }); loadCats(); };

  // Test-detection tool
  $("test-file").onchange = (e) => uploadTest(e.target.files[0]);
  for (const f of ["model", "person_confidence", "cat_confidence", "label_floor",
                   "gamma", "brightness", "contrast", "saturation",
                   "tiling", "tile_overlap", "accelerator", "dog"]) {
    const el = $("t_" + f);
    if (el) el[el.type === "checkbox" ? "onchange" : "oninput"] = scheduleTestDetect;
  }
  for (const id of ["round_robin", "round_robin_size", "round_robin_interval"]) {
    const el = $(id);
    if (el) el.onchange = saveConfig;
  }
  $("test-reset").onclick = () => {
    $("t_gamma").value = 1.0; $("t_brightness").value = 0;
    $("t_contrast").value = 1.0; $("t_saturation").value = 1.0;
    scheduleTestDetect();
  };
  $("test-save").onclick = saveTestSettings;
  $("bench-run").onclick = runBenchmark;
  const bf = $("bench-batch-files");
  if (bf) bf.onchange = (e) => { addBatchFiles(Array.from(e.target.files)); e.target.value = ""; };
  const br = $("bench-batch-run");
  if (br) br.onclick = runBatch;
  const ab = $("bench-batch-abort");
  if (ab) ab.onclick = abortBatch;
  const rx = $("test-reextract");
  if (rx) rx.onclick = () => {
    if (lastTestFile) uploadTest(lastTestFile, Number($("test-frames").value) || undefined);
  };
  const ba = $("test-bench-all");
  if (ba) ba.onclick = benchmarkAllFrames;
  const vf = $("vlm-file");
  if (vf) vf.onchange = (e) => uploadVlm(e.target.files[0]);
  const vks = $("vlm-key-save");
  if (vks) vks.onclick = saveVlmKey;
  const vr = $("vlm-run");
  if (vr) vr.onclick = runVlm;
  const vloc = $("vlm-locate");
  if (vloc) vloc.onclick = runLocate;
  const er = $("esc-run");
  if (er) er.onclick = () => runEscalation(null);
  const erc = $("esc-run-camera");
  if (erc) erc.onclick = () => runEscalation($("esc-camera").value);
  const etb = $("esc-trail-btn");
  if (etb) etb.onclick = showTrail;
  const ehb = $("esc-heatmap-btn");
  if (ehb) ehb.onclick = showHeatmap;
  const vtb = $("vlm-temporal");
  if (vtb) vtb.onclick = () => runTemporal(null);
  const etp = $("esc-temporal-btn");
  if (etp) etp.onclick = () => runTemporal($("esc-camera").value);
  const et = $("vlm-escalation-toggle");
  if (et) et.onchange = async () => {
    await api("/api/config", postJSON({ vlm_escalation: et.checked }));
    await loadVlmStatus();          // re-read the gate; refreshStatus shows/hides the row
  };
  const vbf = $("vlm-batch-files");
  if (vbf) vbf.onchange = (e) => { addVlmBatchFiles(Array.from(e.target.files)); e.target.value = ""; };
  const vbr = $("vlm-batch-run");
  if (vbr) vbr.onclick = runVlmBatch;
  const vba = $("vlm-batch-abort");
  if (vba) vba.onclick = abortVlmBatch;
}

async function loadVersion() {
  const { body } = await api("/api/version");
  if (body && body.version) $("app-version").textContent = "v" + body.version;
}

function showInitError() {
  const el = $("init-error");
  if (el) el.hidden = false;
}
function clearInitError() {
  const el = $("init-error");
  if (el) el.hidden = true;
}

// The one-time data load that seeds the whole UI. Kept separate from init() so a
// failure can be retried without re-wiring handlers or stacking pollers (H3).
// Returns whether the backend was reachable (the /api/config bootstrap loaded).
async function loadInitialData() {
  await loadModelOptions();      // before config seeds #t_model and before cameras render (#50)
  const reached = await loadConfig();
  await Promise.all([loadSpeakers(false), loadSounds(), loadCamerasList(), loadBenchOptions(), loadVlmStatus()]);
  return reached;
}

function applyLoadResult(reached) {
  if (reached) clearInitError(); else showInitError();
}

function retryInitialLoad() {
  loadInitialData().then(applyLoadResult).catch(() => showInitError());
}

// Start the recurring polls. Runs unconditionally (even if the first load failed),
// so the UI recovers on its own once the backend is reachable again (H3).
function startPolling() {
  refreshStatus();
  loadLog();
  loadCats();
  loadVersion();
  setInterval(() => { refreshStatus(); loadLog(); }, 3000);
  setInterval(() => { loadCats(); pollFeeds(); }, 1200);
  setInterval(rotateCatFeed, 4000);   // rotate among rooms with a cat when armed
}

async function init() {
  wire();
  try {
    applyLoadResult(await loadInitialData());
  } catch (err) {
    // A failed first load must not brick the UI (H3): show a visible retry and
    // still start the pollers below, which recover on their own.
    console.error("initial load failed", err);
    showInitError();
  }
  startPolling();
}

init();
