"use strict";

const $ = (id) => document.getElementById(id);
const api = async (path, opts) => {
  const res = await fetch(path, opts);
  let body = null;
  try { body = await res.json(); } catch (_) { /* no body */ }
  return { ok: res.ok, body };
};

// Fields that map 1:1 to config keys (populated into inputs on load).
const FIELDS = [
  "camera_name", "camera_url", "camera_username", "camera_password",
  "dice_sides", "dc", "cooldown_seconds", "person_confidence",
  "confirm_frames", "detect_size", "scan_fps", "quiet_start", "quiet_end",
  "label_floor", "motion_min_area_frac", "motion_diff_threshold", "motion_min_blob_px",
];

// Motion-sensitivity presets → the three raw MotionPrefilter knobs.
const MOTION_PRESETS = {
  low:    { motion_min_area_frac: 0.006,  motion_diff_threshold: 30, motion_min_blob_px: 18 },
  medium: { motion_min_area_frac: 0.003,  motion_diff_threshold: 25, motion_min_blob_px: 14 },
  high:   { motion_min_area_frac: 0.0015, motion_diff_threshold: 18, motion_min_blob_px: 10 },
};

function applyMotionPreset(name) {
  const p = MOTION_PRESETS[name];
  if (!p) return;   // "custom" — leave the advanced fields as the user set them
  for (const [k, v] of Object.entries(p)) $(k).value = v;
}

// Region of interest [x, y, w, h] in original-frame pixels (null = whole frame).
let roi = null;

let speakers = [];

// ---- populate dropdowns ----------------------------------------------------
async function loadSpeakers(detect) {
  const sel = $("speaker-select");
  sel.innerHTML = `<option disabled>${detect ? "Detecting…" : "Loading…"}</option>`;
  const { body } = await api("/api/speakers");
  speakers = body || [];
  sel.innerHTML = "";
  // Local PC audio is always available, listed first.
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

async function loadCameras(detect) {
  const sel = $("camera-select");
  sel.innerHTML = `<option>${detect ? "Detecting…" : "Loading…"}</option>`;
  const { body } = await api("/api/cameras");
  const cams = body || [];
  sel.innerHTML = `<option value="">— choose or enter manually below —</option>`;
  for (const c of cams) {
    const opt = document.createElement("option");
    opt.value = c.rtsp_url || "";
    opt.textContent = c.rtsp_url ? c.name : `${c.name} (needs login — enter below)`;
    opt.dataset.name = c.name;
    sel.appendChild(opt);
  }
  if (savedCameraUrl) sel.value = savedCameraUrl;
}

let savedCameras = [];

async function loadSavedCameras() {
  const sel = $("camera-saved-select");
  const { body } = await api("/api/cameras/saved");
  savedCameras = body || [];
  sel.innerHTML = savedCameras.length
    ? `<option value="">— pick a saved camera —</option>`
    : `<option value="">— none saved yet —</option>`;
  for (const c of savedCameras) {
    const opt = document.createElement("option");
    opt.value = c.name;
    opt.textContent = c.url ? c.name : `${c.name} (no URL)`;
    if (c.url && c.url === savedCameraUrl) opt.selected = true;
    sel.appendChild(opt);
  }
}

async function useSavedCamera() {
  const name = $("camera-saved-select").value;
  if (!name) return;
  const { ok } = await api("/api/cameras/saved/select", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name }),
  });
  if (ok) { await loadConfig(); await loadCameras(false); }
}

async function deleteSavedCamera() {
  const name = $("camera-saved-select").value;
  if (!name) return;
  await api("/api/cameras/saved/delete", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name }),
  });
  await loadSavedCameras();
}

async function saveCamera() {
  const note = $("camera-save-note");
  const payload = {
    name: $("camera_name").value.trim(),
    url: $("camera_url").value.trim() || $("camera-select").value,
    username: $("camera_username").value.trim(),
  };
  const pw = $("camera_password").value;
  if (pw) payload.password = pw;
  if (!payload.name || !payload.url) {
    note.textContent = "Need a name and a stream URL"; return;
  }
  const { ok, body } = await api("/api/cameras/saved", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  note.textContent = ok ? "Saved ✓" : ((body && body.error) || "Failed");
  if (ok) await loadSavedCameras();
  setTimeout(() => (note.textContent = ""), 2500);
}

async function loadLocalCameras(detect) {
  const sel = $("camera-local-select");
  sel.innerHTML = `<option value="">${detect ? "Detecting…" : "—"}</option>`;
  const { body } = await api("/api/cameras/local");
  const cams = body || [];
  sel.innerHTML = cams.length
    ? `<option value="">— pick a USB camera —</option>`
    : `<option value="">— none found on this PC —</option>`;
  for (const c of cams) {
    const opt = document.createElement("option");
    opt.value = c.value; opt.textContent = c.label;
    if (savedCameraUrl === c.value) opt.selected = true;
    sel.appendChild(opt);
  }
}

function useLocalCamera() {
  const opt = $("camera-local-select").selectedOptions[0];
  const v = opt ? opt.value : "";
  if (!v) return;
  $("camera_url").value = v;            // e.g. "usb:0" — becomes the active camera
  $("camera_name").value = opt.textContent;
  $("camera_username").value = "";
  $("camera_password").value = "";
  $("camera-select").value = "";        // it's a local cam, not a network one
}

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

// ---- config load/save ------------------------------------------------------
let savedSpeakers = [], savedCameraUrl = "", savedSound = "";

function applySpeechVisibility() {
  $("speech-row").classList.toggle("hidden", !$("use_speech").checked);
}

async function loadConfig() {
  const { body: cfg } = await api("/api/config");
  if (!cfg) return;
  for (const f of FIELDS) {
    if (f === "camera_password") continue; // never populated from server
    if ($(f) && cfg[f] !== undefined && cfg[f] !== null) $(f).value = cfg[f];
  }
  $("dont_interrupt_playback").checked = !!cfg.dont_interrupt_playback;
  $("keep_speakers_warm").checked = !!cfg.keep_speakers_warm;
  $("pause_during_cooldown").checked = cfg.pause_during_cooldown !== false;
  $("smooth_feed").checked = !!cfg.smooth_live_feed;
  if (cfg.detector_model) $("detector_model").value = cfg.detector_model;
  if (cfg.accelerator) $("accelerator").value = cfg.accelerator;
  if (cfg.motion_sensitivity) $("motion_sensitivity").value = cfg.motion_sensitivity;
  $("use_speech").checked = !!cfg.use_speech;
  if (cfg.speech_text) $("speech_text").value = cfg.speech_text;
  applySpeechVisibility();
  roi = Array.isArray(cfg.roi) && cfg.roi.length === 4 ? cfg.roi.slice() : null;
  $("roi-note").textContent = roi ? `Region set: ${roi[2]}×${roi[3]}px` : "";
  savedSpeakers = Array.isArray(cfg.speaker_names) && cfg.speaker_names.length
    ? cfg.speaker_names.slice()
    : (cfg.speaker_name ? [cfg.speaker_name] : []);
  savedCameraUrl = cfg.camera_url || "";
  savedSound = cfg.sound_file || "";
  updateOdds();
}

// ---- region-of-interest picker --------------------------------------------
function drawRoiBox() {
  const cv = $("roi-canvas"), img = $("roi-img");
  if (!cv.getContext) return;
  const ctx = cv.getContext("2d");
  ctx.clearRect(0, 0, cv.width, cv.height);
  if (!roi || !img.naturalWidth) return;
  const sx = cv.width / img.naturalWidth, sy = cv.height / img.naturalHeight;
  ctx.strokeStyle = "#7c5cff";
  ctx.lineWidth = 2;
  ctx.strokeRect(roi[0] * sx, roi[1] * sy, roi[2] * sx, roi[3] * sy);
}

async function grabPreview() {
  const note = $("roi-note");
  note.textContent = "Grabbing a frame…";
  await saveConfig();                 // make sure the camera URL/login are saved
  const img = $("roi-img");
  img.onload = () => {
    $("roi-stage").classList.remove("hidden");
    const cv = $("roi-canvas");
    cv.width = img.clientWidth;
    cv.height = img.clientHeight;
    drawRoiBox();
    note.textContent = roi
      ? `Region set: ${roi[2]}×${roi[3]}px — drag to change`
      : "Drag a box over the area to watch";
  };
  img.onerror = () => {
    note.textContent = "Couldn't grab a frame — is the camera reachable?";
  };
  img.src = "/api/preview?ts=" + Date.now();
}

function wireRoiCanvas() {
  const cv = $("roi-canvas"), img = $("roi-img");
  let start = null;
  const toImg = (e) => {
    const r = cv.getBoundingClientRect();
    const sx = img.naturalWidth / cv.width, sy = img.naturalHeight / cv.height;
    return [
      Math.round(Math.max(0, e.clientX - r.left) * sx),
      Math.round(Math.max(0, e.clientY - r.top) * sy),
    ];
  };
  cv.onmousedown = (e) => { start = toImg(e); };
  cv.onmousemove = (e) => {
    if (!start) return;
    const [x, y] = toImg(e);
    roi = [Math.min(start[0], x), Math.min(start[1], y),
           Math.abs(x - start[0]), Math.abs(y - start[1])];
    drawRoiBox();
  };
  cv.onmouseup = () => {
    start = null;
    if (roi && (roi[2] < 8 || roi[3] < 8)) roi = null;   // ignore stray clicks
    drawRoiBox();
    $("roi-note").textContent = roi ? `Region set: ${roi[2]}×${roi[3]}px` : "Cleared";
  };
}

function gatherConfig() {
  const camSel = $("camera-select");
  // Prefer an explicit manual URL; otherwise the selected camera.
  const cameraUrl = $("camera_url").value.trim() || camSel.value;
  const camName = $("camera_name").value.trim()
    || camSel.selectedOptions[0]?.dataset.name || "";
  const values = {
    camera_url: cameraUrl,
    camera_name: camName,
    camera_username: $("camera_username").value.trim(),
    speaker_names: selectedSpeakers(),
    sound_file: $("sound-select").value,
    use_speech: $("use_speech").checked,
    speech_text: $("speech_text").value.trim() || "Give the cat a treat!",
    dice_sides: Number($("dice_sides").value),
    dc: Number($("dc").value),
    cooldown_seconds: Number($("cooldown_seconds").value),
    person_confidence: Number($("person_confidence").value),
    confirm_frames: Number($("confirm_frames").value),
    detect_size: Number($("detect_size").value),
    detector_model: $("detector_model").value,
    accelerator: $("accelerator").value,
    smooth_live_feed: $("smooth_feed").checked,
    scan_fps: Number($("scan_fps").value),
    label_floor: Number($("label_floor").value),
    motion_sensitivity: $("motion_sensitivity").value,
    motion_min_area_frac: Number($("motion_min_area_frac").value),
    motion_diff_threshold: Number($("motion_diff_threshold").value),
    motion_min_blob_px: Number($("motion_min_blob_px").value),
    pause_during_cooldown: $("pause_during_cooldown").checked,
    quiet_start: $("quiet_start").value,
    quiet_end: $("quiet_end").value,
    roi: roi,
    dont_interrupt_playback: $("dont_interrupt_playback").checked,
    keep_speakers_warm: $("keep_speakers_warm").checked,
  };
  const pw = $("camera_password").value;
  if (pw) values.camera_password = pw; // only send if user typed one
  return values;
}

async function saveConfig() {
  const note = $("save-note");
  note.textContent = "Saving…";
  const { ok } = await api("/api/config", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(gatherConfig()),
  });
  note.textContent = ok ? "Saved ✓" : "Save failed";
  setTimeout(() => (note.textContent = ""), 2500);
}

// ---- control ---------------------------------------------------------------
let isRunning = false;

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
  updateLiveView(body.running);
}

// ---- live detection feed ---------------------------------------------------
let liveOn = false;

function updateLiveView(running) {
  const img = $("live-img"), note = $("live-note");
  const want = running && $("live-enabled").checked;
  if (want && !liveOn) {
    // Point the <img> at the MJPEG stream; the browser renders it live.
    img.src = "/api/stream?ts=" + Date.now();
    img.classList.remove("hidden");
    note.textContent = "Live — green = person, orange = cat.";
    liveOn = true;
  } else if (!want && liveOn) {
    img.src = "";                       // close the streaming connection
    img.classList.add("hidden");
    liveOn = false;
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
  // Flash the big button while a cat is actually on camera right now.
  const btn = $("show-cat"), label = $("show-cat-label");
  if (body.present) {
    btn.classList.add("detecting");
    label.textContent = "Cat spotted — show me!";
  } else {
    btn.classList.remove("detecting");
    label.textContent = "Show me the cat!";
  }
  const box = $("cat-last");
  if (!body.last) {
    box.innerHTML = '<p class="muted">No cats seen yet.</p>';
    return;
  }
  const s = body.last;
  const where = s.region ? ` — ${s.region}` : "";
  const cam = s.camera ? ` on <strong>${s.camera}</strong>` : "";
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
  await loadCats();
  // "Pull up the feed": make sure the live view is on, then scroll to it while
  // watching (so you can catch the cat live); otherwise show the last snapshot.
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
  // Skip the re-render (and preserve scroll position) when nothing changed.
  const key = entries.length ? `${entries.length}:${entries[0].ts}` : "0";
  if (key === lastLogKey) return;
  lastLogKey = key;

  const list = $("log-list");
  if (!entries.length) {
    list.innerHTML = `<p class="muted">No activity yet.</p>`;
    return;
  }
  list.innerHTML = "";
  for (const e of entries) {
    const row = document.createElement("div");
    row.className = `log-row log-${e.kind || "info"}`;
    const t = document.createElement("span");
    t.className = "log-time";
    t.textContent = fmtTime(e.ts);
    const m = document.createElement("span");
    m.className = "log-msg";
    m.textContent = e.message;
    row.append(t, m);
    if (e.image) {
      const a = document.createElement("a");
      a.href = `/snapshots/${e.image}`;
      a.target = "_blank";
      a.title = "Open full snapshot";
      const img = document.createElement("img");
      img.className = "log-thumb";
      img.src = `/snapshots/${e.image}`;
      img.alt = "detection snapshot";
      a.appendChild(img);
      row.appendChild(a);
    }
    list.appendChild(row);
  }
}

// ---- wiring ----------------------------------------------------------------
function wire() {
  $("speaker-refresh").onclick = () => loadSpeakers(true);
  $("camera-refresh").onclick = () => loadCameras(true);
  $("camera-local-refresh").onclick = () => loadLocalCameras(true);
  $("camera-local-select").onchange = useLocalCamera;
  $("camera-saved-use").onclick = useSavedCamera;
  $("camera-saved-delete").onclick = deleteSavedCamera;
  $("camera-save").onclick = saveCamera;
  $("speaker-select").onchange = updateSpeakerWarning;
  $("use_speech").onchange = applySpeechVisibility;
  $("motion_sensitivity").onchange = () => applyMotionPreset($("motion_sensitivity").value);
  for (const id of ["motion_min_area_frac", "motion_diff_threshold", "motion_min_blob_px"]) {
    $(id).oninput = () => { $("motion_sensitivity").value = "custom"; };
  }
  $("dice_sides").oninput = updateOdds;
  $("dc").oninput = updateOdds;
  $("quiet-clear").onclick = () => { $("quiet_start").value = ""; $("quiet_end").value = ""; };
  $("roi-grab").onclick = grabPreview;
  $("roi-clear").onclick = () => { roi = null; drawRoiBox(); $("roi-note").textContent = "Region cleared"; };
  wireRoiCanvas();

  $("sound-upload").onchange = async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    const fd = new FormData();
    fd.append("file", file);
    const { ok, body } = await api("/api/sounds", { method: "POST", body: fd });
    if (ok) { savedSound = body.saved; await loadSounds(); }
    else alert((body && body.error) || "Upload failed");
  };

  $("test-btn").onclick = async () => {
    await saveConfig(); // ensure chosen speakers/sound/message are persisted first
    $("test-btn").textContent = "Playing…";
    const { ok, body } = await api("/api/test", { method: "POST" });
    $("test-btn").textContent = "▶ Test";
    if (!ok) alert((body && body.error) || "Could not play on the speaker(s).");
  };

  $("log-clear").onclick = async () => {
    await api("/api/log/clear", { method: "POST" });
    lastLogKey = "";
    loadLog();
  };

  $("save-btn").onclick = saveConfig;
  $("start-btn").onclick = async () => {
    await saveConfig();
    await api("/api/start", { method: "POST" });
    refreshStatus();
    loadLog();
  };
  $("stop-btn").onclick = async () => {
    await api("/api/stop", { method: "POST" });
    refreshStatus();
    loadLog();
  };

  // Live feed: react to the toggle immediately, and recover from a dropped
  // stream (e.g. a brief 409 right after Start) by letting the next poll retry.
  $("live-enabled").onchange = () => refreshStatus();
  $("live-img").onerror = () => {
    if (liveOn) { liveOn = false; $("live-img").classList.add("hidden"); }
  };

  // Smooth feed: persist + apply live (the loop reconciles it). Re-point the
  // <img> so the browser reconnects to the now-faster stream.
  $("smooth_feed").onchange = async () => {
    await api("/api/live/smooth", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ on: $("smooth_feed").checked }),
    });
    if (liveOn) { liveOn = false; refreshStatus(); }   // reconnect the stream
  };

  $("show-cat").onclick = showCat;
  $("cats-clear").onclick = async () => {
    await api("/api/cats/clear", { method: "POST" });
    loadCats();
  };
}

async function loadVersion() {
  const { body } = await api("/api/version");
  if (body && body.version) $("app-version").textContent = "v" + body.version;
}

async function init() {
  wire();
  await loadConfig();
  await Promise.all([
    loadSpeakers(false), loadCameras(false), loadSounds(), loadSavedCameras(),
    loadLocalCameras(false),
  ]);
  refreshStatus();
  loadLog();
  loadCats();
  loadVersion();
  setInterval(() => { refreshStatus(); loadLog(); }, 3000);
  setInterval(loadCats, 1200);   // faster, so the Show-cat button flashes near real-time
}

init();
