"use strict";

const $ = (id) => document.getElementById(id);
const api = async (path, opts) => {
  const res = await fetch(path, opts);
  let body = null;
  try { body = await res.json(); } catch (_) { /* no body */ }
  return { ok: res.ok, body };
};

// Fields that map 1:1 to config keys.
const FIELDS = [
  "camera_url", "camera_username", "camera_password",
  "dice_sides", "dc", "cooldown_seconds", "person_confidence",
  "confirm_frames", "detect_size", "scan_fps", "quiet_start", "quiet_end",
];

// Region of interest [x, y, w, h] in original-frame pixels (null = whole frame).
let roi = null;

let speakers = [];

// ---- populate dropdowns ----------------------------------------------------
async function loadSpeakers(detect) {
  const sel = $("speaker-select");
  sel.innerHTML = `<option>${detect ? "Detecting…" : "Loading…"}</option>`;
  const { body } = await api("/api/speakers");
  speakers = body || [];
  sel.innerHTML = "";
  if (!speakers.length) {
    sel.innerHTML = `<option value="">No speakers found — check same WiFi</option>`;
  }
  for (const s of speakers) {
    const opt = document.createElement("option");
    opt.value = s.name;
    opt.textContent = s.is_group ? `${s.name}  (group)` : s.name;
    sel.appendChild(opt);
  }
  if (savedSpeaker) sel.value = savedSpeaker;
  updateSpeakerWarning();
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
  const sel = $("speaker-select");
  const warn = $("speaker-warn");
  const chosen = speakers.find((s) => s.name === sel.value);
  if (chosen && chosen.is_group) {
    warn.textContent = "⚠ This is a speaker group — the chime will play on every speaker in it.";
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
  } else {
    $("odds").textContent = "—";
  }
}

// ---- config load/save ------------------------------------------------------
let savedSpeaker = "", savedCameraUrl = "", savedSound = "";

async function loadConfig() {
  const { body: cfg } = await api("/api/config");
  if (!cfg) return;
  for (const f of FIELDS) {
    if (f === "camera_password") continue; // never populated from server
    if ($(f) && cfg[f] !== undefined && cfg[f] !== null) $(f).value = cfg[f];
  }
  $("dont_interrupt_playback").checked = !!cfg.dont_interrupt_playback;
  roi = Array.isArray(cfg.roi) && cfg.roi.length === 4 ? cfg.roi.slice() : null;
  $("roi-note").textContent = roi ? `Region set: ${roi[2]}×${roi[3]}px` : "";
  savedSpeaker = cfg.speaker_name || "";
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
  const camName = camSel.selectedOptions[0]?.dataset.name || "";
  const values = {
    camera_url: cameraUrl,
    camera_name: camName,
    camera_username: $("camera_username").value.trim(),
    speaker_name: $("speaker-select").value,
    sound_file: $("sound-select").value,
    dice_sides: Number($("dice_sides").value),
    dc: Number($("dc").value),
    cooldown_seconds: Number($("cooldown_seconds").value),
    person_confidence: Number($("person_confidence").value),
    confirm_frames: Number($("confirm_frames").value),
    detect_size: Number($("detect_size").value),
    scan_fps: Number($("scan_fps").value),
    quiet_start: $("quiet_start").value,
    quiet_end: $("quiet_end").value,
    roi: roi,
    dont_interrupt_playback: $("dont_interrupt_playback").checked,
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
async function refreshStatus() {
  const { body } = await api("/api/status");
  if (!body) return;
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
  $("speaker-select").onchange = updateSpeakerWarning;
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
    await saveConfig(); // ensure chosen speaker/sound are persisted first
    $("test-btn").textContent = "Playing…";
    const { ok, body } = await api("/api/test", { method: "POST" });
    $("test-btn").textContent = "▶ Test sound";
    if (!ok) alert((body && body.error) || "Could not play on the speaker.");
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
}

async function init() {
  wire();
  await loadConfig();
  await Promise.all([loadSpeakers(false), loadCameras(false), loadSounds()]);
  refreshStatus();
  loadLog();
  setInterval(() => { refreshStatus(); loadLog(); }, 3000);
}

init();
