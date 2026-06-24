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
  "confirm_frames", "quiet_start", "quiet_end",
];

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
  savedSpeaker = cfg.speaker_name || "";
  savedCameraUrl = cfg.camera_url || "";
  savedSound = cfg.sound_file || "";
  updateOdds();
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
    quiet_start: $("quiet_start").value,
    quiet_end: $("quiet_end").value,
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
