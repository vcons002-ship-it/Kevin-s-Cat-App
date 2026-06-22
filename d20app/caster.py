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


class Caster:
    """Discover a speaker by name and cast sounds to it."""

    def __init__(self, sound_server: SoundServer) -> None:
        self.sound_server = sound_server

    def _get_speaker(self, name: str):
        import pychromecast

        chromecasts, browser = pychromecast.get_listed_chromecasts(
            friendly_names=[name]
        )
        if not chromecasts:
            pychromecast.discovery.stop_discovery(browser)
            raise LookupError(f"Cast device named {name!r} not found on the network")
        cast = chromecasts[0]
        cast.wait()
        return cast, browser

    def play_sound(
        self,
        speaker_name: str,
        filename: str,
        dont_interrupt: bool = False,
        clip_seconds: float = 3.0,
    ) -> bool:
        """Cast ``filename`` to ``speaker_name``.

        Returns True if the sound was cast, False if skipped because
        ``dont_interrupt`` is set and the speaker is actively playing media.
        Restores the speaker's prior volume afterwards.
        """
        import pychromecast  # noqa: F401  (ensures dependency present early)

        self.sound_server.start()
        cast, browser = self._get_speaker(speaker_name)
        try:
            mc = cast.media_controller
            if dont_interrupt:
                mc.update_status()
                if mc.status.player_is_playing:
                    return False

            prior_volume = cast.status.volume_level if cast.status else None

            url = self.sound_server.url_for(filename)
            content_type = mimetypes.guess_type(filename)[0] or "audio/wav"
            mc.play_media(url, content_type)
            mc.block_until_active(timeout=10)
            # Let the short clip play out, then restore the speaker's volume.
            time.sleep(clip_seconds)
            if prior_volume is not None:
                cast.set_volume(prior_volume)
            return True
        finally:
            import pychromecast

            pychromecast.discovery.stop_discovery(browser)

    def say(self, speaker_name: str, text: str) -> None:
        """Optional future hook: speak a message (e.g. "Give the cat a treat!").

        Disabled for now — Kevin chose a sound and will pick a spoken message
        later. Would synthesise TTS (e.g. via gTTS) into SOUNDS_DIR and cast it
        the same way as :meth:`play_sound`.
        """
        raise NotImplementedError("Spoken messages are not enabled yet.")
