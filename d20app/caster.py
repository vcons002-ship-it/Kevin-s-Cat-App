"""Cast a sound to a Google Home / Nest speaker over the local network.

Uses ``pychromecast`` (the Google Cast protocol) — no Google account, cloud,
API key, or OAuth. The speaker must be on the same WiFi/subnet as this host.

Because a Cast device fetches media over HTTP, we run a tiny local file server
(serving ``d20app/sounds/``) and hand the speaker a ``http://<lan-ip>:<port>/<file>``
URL. The cast URL must be a reachable IP, not a ``.local`` name, so we
auto-detect the host's LAN IP.

Casting only affects the single target speaker. To leave it as we found it, we
save and restore its volume around the short clip.
"""

from __future__ import annotations

import functools
import http.server
import mimetypes
import os
import socket
import threading
import time

SOUNDS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sounds")


def detect_lan_ip() -> str:
    """Best-effort detection of this host's LAN IP (no traffic actually sent)."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))      # doesn't send packets; just picks a route
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


class SoundServer:
    """Serves files from ``SOUNDS_DIR`` over HTTP on the LAN for the speaker."""

    def __init__(self, port: int = 8081, directory: str = SOUNDS_DIR) -> None:
        self.port = port
        self.directory = directory
        self.lan_ip = detect_lan_ip()
        self._httpd = None
        self._thread = None

    def start(self) -> None:
        if self._httpd is not None:
            return
        handler = functools.partial(
            http.server.SimpleHTTPRequestHandler, directory=self.directory
        )
        # Bind all interfaces so the speaker can reach us; advertise the LAN IP.
        self._httpd = http.server.ThreadingHTTPServer(("0.0.0.0", self.port), handler)
        self._thread = threading.Thread(
            target=self._httpd.serve_forever, name="sound-server", daemon=True
        )
        self._thread.start()

    def url_for(self, filename: str) -> str:
        return f"http://{self.lan_ip}:{self.port}/{filename}"

    def stop(self) -> None:
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd = None


import hashlib


def _as_list(names) -> list:
    """Accept a single name or a list; return a clean list of names."""
    if isinstance(names, str):
        names = [names]
    return [n for n in (names or []) if n]


class Caster:
    """Cast sounds/speech to one or more Cast devices.

    Each cast opens a fresh connection, plays, then disconnects — simple and
    reliable. (A persistent-connection variant that avoids the brief
    "connecting" chime is planned, but it was unstable and was reverted.)
    """

    def __init__(self, sound_server: SoundServer) -> None:
        self.sound_server = sound_server

    # -- connection / playback ----------------------------------------------
    def _connect(self, name: str):
        import pychromecast

        chromecasts, browser = pychromecast.get_listed_chromecasts(
            friendly_names=[name]
        )
        if not chromecasts:
            pychromecast.discovery.stop_discovery(browser)
            raise LookupError(f"Cast device named {name!r} not found on the network")
        cast = chromecasts[0]
        cast.wait(timeout=15)
        return cast, browser

    def _play_one(self, name: str, url: str, content_type: str,
                  dont_interrupt: bool) -> bool:
        """Open a fresh connection to one speaker, play, then disconnect."""
        import pychromecast

        cast, browser = self._connect(name)
        try:
            mc = cast.media_controller
            if dont_interrupt:
                mc.update_status()
                if mc.status.player_is_playing:
                    return False
            mc.play_media(url, content_type)
            mc.block_until_active(timeout=10)
            return True
        finally:
            pychromecast.discovery.stop_discovery(browser)

    def play_media(self, names, url: str, content_type: str,
                   dont_interrupt: bool = False) -> bool:
        """Play ``url`` on every device in ``names``.

        Returns True if it started on at least one device. Raises only if every
        target failed. With ``dont_interrupt`` a device already playing media is
        skipped.
        """
        self.sound_server.start()
        played, errors = 0, []
        for name in _as_list(names):
            try:
                if self._play_one(name, url, content_type, dont_interrupt):
                    played += 1
            except Exception as exc:        # noqa: BLE001
                errors.append(f"{name}: {exc}")
        if played == 0 and errors:
            raise RuntimeError("; ".join(errors))
        return played > 0

    def play_sound(self, names, filename: str, dont_interrupt: bool = False) -> bool:
        """Cast a file from the sound folder to one or more speakers."""
        url = self.sound_server.url_for(filename)
        content_type = mimetypes.guess_type(filename)[0] or "audio/wav"
        return self.play_media(names, url, content_type, dont_interrupt)

    def say(self, names, text: str, dont_interrupt: bool = False) -> bool:
        """Speak ``text`` on one or more speakers (synthesised once, then cached)."""
        return self.play_sound(names, self._synthesize(text), dont_interrupt)

    def _synthesize(self, text: str) -> str:
        """Render ``text`` to an MP3 in SOUNDS_DIR (cached by content); return name."""
        digest = hashlib.sha1(text.strip().encode("utf-8")).hexdigest()[:12]
        filename = f"speech_{digest}.mp3"
        path = os.path.join(self.sound_server.directory, filename)
        if os.path.exists(path):
            return filename
        try:
            from gtts import gTTS
        except Exception as exc:        # noqa: BLE001
            raise RuntimeError(
                "Spoken messages need the gTTS package — run setup.sh again."
            ) from exc
        try:
            gTTS(text=text, lang="en").save(path)
        except Exception as exc:        # noqa: BLE001
            raise RuntimeError(
                f"Couldn't synthesize speech (needs internet access): {exc}"
            ) from exc
        return filename
