"use strict";

const $ = (id) => document.getElementById(id);
const api = async (path, opts) => {
  const res = await fetch(path, opts);
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
const MODEL_OPTS = [["yolo11n", "YOLO11n — low light (rec.)"], ["yolo11m", "YOLO11m — heavier"], ["mobilenet_ssd", "MobileNet — lightest"]];
const ACCEL_OPTS = [["cpu", "CPU"], ["opencl", "OpenCL iGPU"], ["openvino-auto", "OpenVINO AUTO"], ["openvino-gpu", "OpenVINO GPU"]];
const SIZE_OPTS = [["300", "Standard"], ["512", "High"], ["768", "Max"]];
const SENS_OPTS = [["low", "Low"], ["medium", "Medium"], ["high", "High"], ["custom", "Custom"]];
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
      <button type="button" class="ghost" data-toggle>Edit ▾</button>
    </div>
    <div class="cam-body hidden">
      <label>Stream URL <input type="text" data-f="url" value="${esc(cam.url)}" placeholder="rtsp://… or usb:0"/></label>
      <div class="grid">
        <label>Username <input type="text" data-f="username" autocomplete="off" value="${esc(cam.username)}"/></label>
        <label>Password <input type="password" data-f="password" autocomplete="off" placeholder="${cam.has_password ? "(unchanged)" : ""}"/></label>
        <label>Model <select data-f="model">${optsHTML(MODEL_OPTS, cam.model)}</select></label>
        <label>Accelerator <select data-f="accelerator">${optsHTML(ACCEL_OPTS, cam.accelerator)}</select></label>
        <label>Person confidence <input type="number" step="0.05" min="0.1" max="1" data-f="person_confidence" value="${cam.person_confidence}"/></label>
        <label>Confirm frames <input type="number" min="1" max="30" data-f="confirm_frames" value="${cam.confirm_frames}"/></label>
        <label>Detection detail <select data-f="detect_size">${optsHTML(SIZE_OPTS, cam.detect_size)}</select></label>
        <label>Scan rate (fps) <input type="number" min="1" max="30" data-f="scan_fps" value="${cam.scan_fps}"/></label>
        <label>Notify floor <input type="number" step="0.05" min="0.1" max="1" data-f="label_floor" value="${cam.label_floor}"/></label>
        <label>Motion sensitivity <select data-f="motion_sensitivity">${optsHTML(SENS_OPTS, cam.motion_sensitivity)}</select></label>
      </div>
      <div class="row">
        <button type="button" class="ghost" data-roi>📷 Set region…</button>
        <span class="muted" data-roinote>${roiTxt}</span>
      </div>
      <div class="row">
        <button type="button" class="primary" data-save>Save camera</button>
        <button type="button" class="stop" data-delete>Delete</button>
        <span class="muted" data-note></span>
      </div>
    </div>
  </div>`;
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
  // Region: the picker stashes the chosen box on the card; else keep stored.
  cam.roi = card._roi !== undefined ? card._roi : (stored.roi || null);
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
  card.querySelector("[data-toggle]").onclick = () =>
    card.querySelector(".cam-body").classList.toggle("hidden");
  card.querySelector("[data-watch]").onchange = saveActiveCameras;
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
}

function populateLiveCameras() {
  const sel = $("live-camera");
  if (!sel) return;
  const prev = sel.value;
  const names = activeCameras.length ? activeCameras : cameras.map((c) => c.name);
  sel.innerHTML = names.map((n) => `<option value="${esc(n)}">${esc(n)}</option>`).join("");
  sel.style.display = names.length > 1 ? "" : "none";   // only show when there's a choice
  if (names.includes(prev)) sel.value = prev;
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
  if (!roi || !img.naturalWidth) return;
  const sx = cv.width / img.naturalWidth, sy = cv.height / img.naturalHeight;
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

async function loadConfig() {
  const { body: cfg } = await api("/api/config");
  if (!cfg) return;
  for (const f of FIELDS) {
    if ($(f) && cfg[f] !== undefined && cfg[f] !== null) $(f).value = cfg[f];
  }
  $("dont_interrupt_playback").checked = !!cfg.dont_interrupt_playback;
  $("keep_speakers_warm").checked = !!cfg.keep_speakers_warm;
  $("pause_during_cooldown").checked = cfg.pause_during_cooldown !== false;
  $("smooth_feed").checked = !!cfg.smooth_live_feed;
  $("use_speech").checked = !!cfg.use_speech;
  if (cfg.speech_text) $("speech_text").value = cfg.speech_text;
  applySpeechVisibility();
  savedSpeakers = Array.isArray(cfg.speaker_names) && cfg.speaker_names.length
    ? cfg.speaker_names.slice()
    : (cfg.speaker_name ? [cfg.speaker_name] : []);
  savedSound = cfg.sound_file || "";
  activeCameras = Array.isArray(cfg.active_cameras) ? cfg.active_cameras.slice() : [];
  // Defaults for a brand-new camera = the current global detection settings.
  camDefaults = {
    model: cfg.detector_model, accelerator: cfg.accelerator,
    person_confidence: cfg.person_confidence, confirm_frames: cfg.confirm_frames,
    detect_size: cfg.detect_size, scan_fps: cfg.scan_fps, label_floor: cfg.label_floor,
    motion_sensitivity: cfg.motion_sensitivity,
  };
  updateOdds();
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
    quiet_start: $("quiet_start").value,
    quiet_end: $("quiet_end").value,
    dont_interrupt_playback: $("dont_interrupt_playback").checked,
    keep_speakers_warm: $("keep_speakers_warm").checked,
  };
  return values;
}

async function saveConfig() {
  const note = $("save-note");
  note.textContent = "Saving…";
  const { ok } = await api("/api/config", postJSON(gatherConfig()));
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
    else if (c.connected) { chip.textContent = "● live"; chip.className = "cam-chip chip-ok"; }
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
  renderCamChips(body.cameras);
  updateLiveView(body.running);
}

// ---- live detection feed ---------------------------------------------------
let liveOn = false, liveCam = null;

function updateLiveView(running) {
  const img = $("live-img"), note = $("live-note"), cam = $("live-camera").value || "";
  const want = running && $("live-enabled").checked;
  if (want && (!liveOn || cam !== liveCam)) {
    const q = cam ? `camera=${encodeURIComponent(cam)}&` : "";
    img.src = `/api/stream?${q}ts=${Date.now()}`;
    img.classList.remove("hidden");
    note.textContent = "Live — green = person, orange = cat.";
    liveOn = true; liveCam = cam;
  } else if (!want && liveOn) {
    img.src = ""; img.classList.add("hidden"); liveOn = false;
  }
  if (!want) {
    note.textContent = running
      ? "Live feed off — tick “Show live feed” to view."
      : "Start watching to see the live feed.";
  }
}

// ---- cat sightings ("show cat") --------------------------------------------
async function loadCats() {
  const { body } = await api("/api/cats");
  if (!body) return;
  const btn = $("show-cat"), label = $("show-cat-label");
  if (body.present) { btn.classList.add("detecting"); label.textContent = "Cat spotted — show me!"; }
  else { btn.classList.remove("detecting"); label.textContent = "Show me the cat!"; }
  const box = $("cat-last");
  if (!body.last) { box.innerHTML = '<p class="muted">No cats seen yet.</p>'; return; }
  const s = body.last;
  const where = s.region ? ` — ${s.region}` : "";
  const cam = s.camera ? ` on <strong>${esc(s.camera)}</strong>` : "";
  const thumb = s.image
    ? `<a href="/snapshots/${s.image}" target="_blank">
         <img class="cat-thumb" src="/snapshots/${s.image}" alt="last cat sighting" /></a>`
    : "";
  const today = `${body.today} sighting${body.today === 1 ? "" : "s"} today`;
  box.innerHTML = `${thumb}<div>
      <div><strong>Last seen</strong> ${fmtTime(s.ts)}${cam}${where}
        <span class="muted">(score ${s.score})</span></div>
      <div class="muted">${today}</div></div>`;
}

async function showCat() {
  const { body } = await api("/api/cats");
  // Point the live feed at the camera that saw the cat, if it's being watched.
  const cam = body && body.last && body.last.camera;
  const sel = $("live-camera");
  if (cam && Array.from(sel.options).some((o) => o.value === cam)) sel.value = cam;
  await loadCats();
  $("live-enabled").checked = true;
  await refreshStatus();
  const target = isRunning ? "live-stage" : "cat-last";
  $(target).scrollIntoView({ behavior: "smooth", block: "center" });
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
      a.href = `/snapshots/${e.image}`; a.target = "_blank"; a.title = "Open full snapshot";
      const img = document.createElement("img");
      img.className = "log-thumb"; img.src = `/snapshots/${e.image}`; img.alt = "detection snapshot";
      a.appendChild(img); row.appendChild(a);
    }
    list.appendChild(row);
  }
}

// ---- wiring ----------------------------------------------------------------
function wire() {
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
    if (!roiEdit) return;
    roiEdit.card._roi = roi;
    const note = roiEdit.card.querySelector("[data-roinote]");
    if (note) note.textContent = roi ? `region ${roi[2]}×${roi[3]}px` : "whole frame";
    $("roi-editor").classList.add("hidden");
    roiEdit.card.querySelector("[data-save]").click();   // persist the new region
    roiEdit = null;
  };
  $("roi-clear").onclick = () => { roi = null; drawRoiBox(); $("roi-note").textContent = "Whole frame"; };
  $("roi-cancel").onclick = () => { $("roi-editor").classList.add("hidden"); roiEdit = null; };

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

  $("live-enabled").onchange = () => refreshStatus();
  $("live-camera").onchange = () => { if (liveOn) { liveOn = false; } refreshStatus(); };
  $("live-img").onerror = () => { if (liveOn) { liveOn = false; $("live-img").classList.add("hidden"); } };
  $("smooth_feed").onchange = async () => {
    await api("/api/live/smooth", postJSON({ on: $("smooth_feed").checked }));
    if (liveOn) { liveOn = false; refreshStatus(); }
  };

  $("show-cat").onclick = showCat;
  $("cats-clear").onclick = async () => { await api("/api/cats/clear", { method: "POST" }); loadCats(); };
}

async function loadVersion() {
  const { body } = await api("/api/version");
  if (body && body.version) $("app-version").textContent = "v" + body.version;
}

async function init() {
  wire();
  await loadConfig();
  await Promise.all([loadSpeakers(false), loadSounds(), loadCamerasList()]);
  refreshStatus();
  loadLog();
  loadCats();
  loadVersion();
  setInterval(() => { refreshStatus(); loadLog(); }, 3000);
  setInterval(loadCats, 1200);
}

init();
