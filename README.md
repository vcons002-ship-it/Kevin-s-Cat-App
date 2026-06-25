# 🎲🐱 Kevin's Cat App — D20 Treat Roller for Google Home

When a **person** walks into the kitchen, this app "rolls a die." On a good
enough roll it plays a celebratory chime on your **Google Home / Nest speaker** —
the cue that it's time to give the cat a treat. It watches an existing IP camera,
**ignores the cats** (it only triggers on people), and is configured entirely
from a simple web page.

- **No Docker, no Frigate, no cloud, no Google account.** Just Python.
- **One setup script**, then everything is point-and-click in a browser.
- **Plays on one or many speakers**, with an optional **spoken message**
  ("Give the cat a treat!") instead of a sound.
- **Live activity log** of every roll, treat, error, and **non-human motion**,
  each with an **annotated snapshot** (boxes around the detected person/cat) so
  you can see exactly what triggered it — right in the page.
- **Quiet time** to silence chimes overnight, and a **confirm-over-N-frames**
  setting to stamp out false positives.
- Runs happily on an **OpenMediaVault** NAS (or any computer on the same WiFi).

![The Kevin's Cat App web GUI](docs/gui.png)

<sub>The whole app is this one page (example shown with sample devices).</sub>

📋 **[Full feature list & roadmap →](ROADMAP.md)**

---

## Requirements

- **Python 3.11 or newer.** OpenMediaVault 7 (Debian 12) ships this already.
  (On older OMV 6 / Debian 11 you'd need a newer Python first — `setup.sh`
  checks the version and tells you exactly what to do if it's too old.)
- **`python3-venv` / `python3-pip`.** On a minimal Debian these may be missing;
  if so, `setup.sh` detects it and **offers to install them for you (just answer
  `Y`)**. If OpenCV ever complains about a missing library, run
  `sudo apt install libglib2.0-0`.
- This computer and your Google Home + camera must be on the **same WiFi/subnet**.

## Quick start

```bash
# 1. Get the code (clone or download the ZIP), then:
./setup.sh

# 2. Start it:
./venv/bin/python run.py
```

`run.py` prints a URL like `http://192.168.1.20:8080`. Open it in a browser on
the **same WiFi**, then:

1. **Camera** — pick your kitchen camera from the auto-detected list (or enter
   its stream URL manually — main or sub feed; see *Detection detail* below).
2. **Speaker(s)** — pick one or more Google Home / Nest devices from the
   auto-detected list, and press **Test** to hear it play.
3. **Sound or spoken message** — keep the default chime, upload your own, or
   tick *Speak a message instead* and type what to say.
4. **Game rules** — set the dice size, the DC (how high you must roll to win a
   treat), how often it's allowed to roll, and the detection knobs
   (*Detection detail* and *Scan rate* — quality vs CPU).
5. **Quiet time** *(optional)* — silence chimes overnight.
6. **Region of interest** *(optional)* — **Grab frame** and drag a box to watch
   just part of the view.
7. Press **Start watching**. Done!

> **Keep it running:** `run.py` stays running in your terminal until you press
> `Ctrl+C` (the camera-watching loop runs inside it; Start/Stop is in the GUI).
> If you launched it over SSH, closing the session stops it. To keep it running
> all the time and restart on boot, set up the
> [systemd service](#run-it-automatically-on-boot-openmediavault--systemd) below.

---

## How it works

```
 Kitchen IP camera ──▶ OpenCV reads the stream
                          │
                          ▼  motion? then run person-detection (MobileNet-SSD)
                  person detected (cats ignored)
                          │
                          ▼  roll a d{sides}; treat if roll ≥ DC; rate-limited
                     on a treat ──▶ Google Cast ──▶ 🔊 chime on your speaker
```

Everything is one Python process. It serves the web GUI **and** runs the
camera-watching loop in the background. Every event (start/stop, each roll, every
treat, and any camera/speaker error) is recorded in the **Activity log** at the
bottom of the page — it persists to `activity.log` so it survives restarts, and
there's a **Clear** button to wipe it.

### Person vs. cat

Detection uses a small **MobileNet-SSD** neural network (bundled in
`d20app/models/`, runs on CPU via OpenCV — no GPU, no extra services). It knows
`person` and `cat` as separate categories, so it triggers on people and
**ignores the cats**. A cheap motion check runs first so the network only fires
when something actually moves, keeping CPU low — and when that motion turns out
**not** to be a person, it's noted in the Activity log (e.g. "*cat moved*").

Measured on 170 real pedestrian images it detects a person **99.4%** of the time
at the default confidence, with **no** cats mistaken for people. (See
`tests/test_detection_accuracy.py`, which guards this with bundled sample
photos.)

**Detection detail (`detect_size`).** The net input size defaults to **300px**,
the model's native size — most reliable for **people** (measured 99–100% recall,
slightly better than 512) and lighter on CPU. A small cat across the room can
vanish at 300; raise the setting to **512** ("High") to recover across-the-room
cats, at more CPU and a small hit to some person poses. People in hats, helmets,
and headgear, and people with their back to the camera, all detect reliably.

### Snapshots & taming false positives

Every detection event in the Activity log carries an **annotated snapshot** — a
thumbnail of the exact frame with a labelled box around what was found
(green = person, orange = cat). Click it for the full image. This is the fastest
way to see *why* something triggered (a coat on a chair, a reflection, the cat).

**Nothing fires without real movement.** Detection only runs after a motion
check passes. That check **median-blurs** each frame and keeps only **solid,
compact** regions of change, so sensor grain, compression noise, a ticking
on-screen clock, and **thin bands of decode-corruption pixels** don't count as
motion — the usual causes of a "trigger with nothing moving". The first frame
and a static scene both report *no* motion. So a roll requires genuine movement
**and** a person seen across several frames in a row:

- **Confirm over N frames** (default 3): a person must be seen in that many
  frames *in a row* before it counts, so a single-frame fluke never fires a
  chime. Raise it if you still get false positives.
- **Person confidence** (default 0.5): raise toward 0.7+ if the snapshots show
  low-confidence boxes on non-people.

### Quiet time

Set a **From/To** window (e.g. 22:00 → 07:00, wraps past midnight) and the app
keeps watching and logging through the night but **won't play the chime**. Leave
both blank to disable.

### Camera feed, scan rate & region of interest

All tunable from the GUI (no config-file editing):

- **Camera feed (main vs sub):** paste whichever stream URL you want in the
  Camera box. The high-res **main** feed spots distant cats better; the **sub**
  feed is lighter on CPU.
- **Detection detail** (`detect_size`): how big a frame the neural net sees —
  *Fast (300)*, *Balanced (512, default)*, or *Detailed (768)*. Bigger sees
  smaller/farther subjects at more CPU.
- **Scan rate** (`scan_fps`): frames read per second (default 10). Lower it to
  reclaim CPU when using the main feed.
- **Region of interest:** click **Grab frame** to pull a still, then **drag a
  box** over the area to watch. Everything outside is ignored — this cuts false
  positives and, by zooming in, makes distant cats much easier to detect.

![Drawing a region of interest on a grabbed camera frame](docs/roi.png)

### Google Home integration

The app talks to your speakers over the local network using the **Google Cast**
protocol (the same thing the "Cast" button uses). There is **no account, no
cloud login, and no API key**. It finds them by the names shown in the Google
Home app. To play audio it briefly serves the file from your NAS and tells the
speaker(s) to play it.

- **Multiple speakers:** pick one or several in the Speaker(s) box (Ctrl/Cmd-click
  or drag) and the treat plays on **all** of them at once.
- **Spoken message:** instead of a sound, tick *Speak a message instead* and type
  what to say (e.g. "Give the cat a treat!"). It's synthesized with **gTTS** and
  spoken on the speakers. Synthesis needs internet the first time a given message
  is used; after that it's cached.

**Will it interfere with my devices?** It only affects the speakers you choose,
briefly interrupting whatever they're playing (it won't resume it). A **speaker
*group*** plays on every speaker in it (the GUI flags groups). There's also a
"don't interrupt if music is already playing" toggle.

---

## Finding your info (if auto-detect comes up empty)

Auto-detection needs the NAS and your devices to be on the **same WiFi/subnet**.
If something doesn't appear:

- **Speaker not listed?** Confirm the speaker and the NAS are on the same WiFi.
  The name must match what you see in the **Google Home app**. Some networks
  block mDNS between WiFi and Ethernet or across "guest" networks.
- **Camera not listed?** Auto-detect uses **ONVIF**. If your camera isn't ONVIF
  or needs a login, open the **"Camera not listed?"** section in the GUI and
  enter the **RTSP URL** plus username/password. Find the RTSP URL in your
  camera's app or web page (often `rtsp://<camera-ip>:554/...`); the camera's IP
  is in your **router's device list** or the camera app.

### Camera connects in VLC but not here

The app opens streams with **FFmpeg over RTSP/TCP**, the same way VLC does, and
**percent-encodes** any username/password you enter (so symbols like `@ : /` in
a password work). If a stream still won't open, the **Activity log** shows the
exact URL it tried with the **password masked**, e.g.:

> Camera problem: could not open the camera stream `rtsp://admin:***@192.168.1.50:554/stream` — check the URL, and the username/password if the camera needs a login

Use that to confirm the address and login are what you expect. The app fails a
dead camera after ~5 seconds and retries with a growing back-off (it no longer
floods the console). To watch the raw FFmpeg decoder output while debugging,
start it with `OPENCV_FFMPEG_LOGLEVEL=24 ./venv/bin/python run.py`.

### The login works but nothing happens

When the loop starts it logs **"▶ Started watching…"**, and as soon as frames
arrive it logs **"📷 Camera connected (W×H)…"**. If you *don't* see the first
line, the loop isn't running — press **Start watching** (not just **Test
sound**) and check the status bar says **Watching**. If you see "Started" but
never "Camera connected", frames aren't decoding.

Run the built-in checker to see exactly where it breaks:

```bash
./venv/bin/python check_camera.py
```

It opens your configured camera, reports whether frames decode (and at what
resolution), runs the person-detector on a frame, and saves `snapshot.jpg`.
The two common findings:

- **"decoded 0 frames"** — the stream opens but the codec won't decode. Most
  cameras default the **main** stream to **H.265/HEVC**; point the app at the
  camera's **H.264 sub-stream** URL instead.
- **"no person above your threshold"** — frames are fine but the model didn't
  see a person. Check `snapshot.jpg` for framing/lighting, stand clearly in
  view, or lower **Person confidence** in the GUI.

---

## Configuration

You normally never edit config by hand — the GUI writes `config.yaml` for you.
Every setting is documented in [`config.example.yaml`](config.example.yaml)
(dice size, DC, frequency interval, detection confidence, optional region of
interest, ports, etc.).

---

## Run it automatically on boot (OpenMediaVault / systemd)

Optional, but recommended for an always-on setup. Create
`/etc/systemd/system/kevins-cat-app.service` — **replace both
`/path/to/Kevin-s-Cat-App` with the real folder path** (run `pwd` inside it to
get it), and set `User=` to the account that owns that folder (the app doesn't
need root):

```ini
[Unit]
Description=Kevin's Cat App
After=network-online.target
Wants=network-online.target

[Service]
User=youruser
WorkingDirectory=/path/to/Kevin-s-Cat-App
ExecStart=/path/to/Kevin-s-Cat-App/venv/bin/python run.py
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

Then enable and start it:

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now kevins-cat-app
```

It now starts on boot and the GUI is always available at `http://<nas-ip>:8080`.
Check it with `systemctl status kevins-cat-app` or `journalctl -u kevins-cat-app -f`.

---

## Stopping it & troubleshooting

**To stop watching but keep the page open:** press **Stop** in the GUI. That
halts the camera loop only; the web server stays up so you can reconfigure and
Start again. To stop the whole program, use whichever matches how you started it:

| How you started it | How to stop it |
|---|---|
| In a terminal / over SSH (`./venv/bin/python run.py`) | Press **Ctrl+C** (closing the SSH session also stops it). |
| As a systemd service | `sudo systemctl stop kevins-cat-app` (add `sudo systemctl disable kevins-cat-app` to stop it starting on boot). |
| In the background / lost the terminal | `pkill -f run.py`, or find it with `ps aux \| grep run.py` and `kill <pid>`. Since it holds port 8080, `sudo lsof -i :8080` also finds it. |

It shuts down cleanly either way — the camera loop and sound server are
background (daemon) threads, so they stop with the main process and leave
nothing running. Your settings are already saved to `config.yaml`.

**"Address already in use" on start?** Something else on the NAS is using port
**8080** (the GUI) or **8081** (the sound server) — some OMV plugins and Docker
containers use 8080. Pick free ports by setting `web_port` / `file_server_port`
in `config.yaml`, then start again.

**A note on access:** the GUI and the sound server have no password, so anyone
already on your home WiFi can open the page. It's only reachable on your local
network (nothing is exposed to the internet), which is fine for a home setup —
just don't run it on an untrusted/shared network.

---

## Development

```bash
./venv/bin/python -m pip install pytest
./venv/bin/python -m pytest        # 59 tests: dice, detection accuracy, motion, etc.
```

- `d20app/config.py` — load/save `config.yaml`; all settings + defaults.
- `d20app/dice.py` — rolling, DC check, cooldown gate (pure, fully tested).
- `d20app/detector.py` — motion pre-filter + person/cat detection, snapshots.
- `d20app/caster.py` — Google Cast playback (multi-speaker), local sound
  server, and gTTS spoken messages.
- `d20app/discovery.py` — speaker (Cast) and camera (ONVIF) auto-detection.
- `d20app/loop.py` — the background watch→roll→cast loop.
- `d20app/activitylog.py` — the persistent, file-backed event log shown in the GUI.
- `d20app/snapshots.py` — saves the annotated detection images the log displays.
- `d20app/webapp.py` + `templates/` + `static/` — the web GUI.
- `check_camera.py` — standalone camera/detection diagnostic.

See [`ROADMAP.md`](ROADMAP.md) for the full feature list and planned ideas, and
[`CHANGELOG.md`](CHANGELOG.md) for the release history.
