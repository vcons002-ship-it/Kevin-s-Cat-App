"""Tests for multi-speaker targeting and cached speech synthesis (no network)."""

import hashlib

from d20app.caster import Caster, _as_list
from d20app.config import Config, speaker_targets


def test_as_list_normalizes():
    assert _as_list("Kitchen") == ["Kitchen"]
    assert _as_list(["A", "", "B"]) == ["A", "B"]
    assert _as_list(None) == []


def test_speaker_targets_prefers_list():
    assert speaker_targets(Config(speaker_names=["A", "B"])) == ["A", "B"]
    assert speaker_targets(Config(speaker_name="Solo")) == ["Solo"]   # legacy fallback
    assert speaker_targets(Config()) == []
    # New list wins over the legacy single field.
    assert speaker_targets(Config(speaker_names=["X"], speaker_name="Y")) == ["X"]


class _FakeServer:
    def __init__(self, directory):
        self.directory = directory

    def start(self):
        pass

    def url_for(self, filename):
        return f"http://host/{filename}"


def test_synthesize_uses_cache_without_network(tmp_path):
    text = "Give the cat a treat!"
    digest = hashlib.sha1(text.strip().encode("utf-8")).hexdigest()[:12]
    name = f"speech_{digest}.mp3"
    (tmp_path / name).write_bytes(b"ID3 fake mp3")        # pretend it's already synthesized
    caster = Caster(_FakeServer(str(tmp_path)))
    # File exists, so this must return it without importing gTTS or hitting the net.
    assert caster._synthesize(text) == name


# --- treat-path regression (the 'speakers_label is not defined' crash) --------

import types

from d20app.config import Config
from d20app.loop import DetectionLoop


class _RecordingActivity:
    def __init__(self):
        self.entries = []

    def add(self, kind, message, image=None):
        self.entries.append((kind, message))


class _RecordingCaster:
    def __init__(self, fail=False):
        self.fail = fail
        self.calls = []

    def play_sound(self, names, filename, dont_interrupt=False):
        self.calls.append(("play_sound", list(names)))
        if self.fail:
            raise RuntimeError("speaker offline")
        return True

    def say(self, names, text, dont_interrupt=False):
        self.calls.append(("say", list(names)))
        if self.fail:
            raise RuntimeError("speaker offline")
        return True


def _recording_loop():
    loop = DetectionLoop()
    loop.activity = _RecordingActivity()
    return loop


def test_treat_cast_path_casts_to_targets():
    """A won roll casts to the given targets and logs a treat — no NameError.

    Regression for the crash where the cast path referenced ``targets`` /
    ``speakers_label`` from ``_run``'s scope inside ``_loop_body``.
    """
    loop = _recording_loop()
    caster = _RecordingCaster()
    result = types.SimpleNamespace(value=18, treat=True)
    loop._cast_for_treat(Config(), caster, ["Kitchen", "Den"], "Kitchen, Den",
                         result, "rolled 18 on d20 (need ≥ 20)", None)
    assert ("play_sound", ["Kitchen", "Den"]) in caster.calls
    assert any(kind == "treat" for kind, _ in loop.activity.entries)


def test_treat_cast_error_names_the_speakers():
    """A cast failure logs a graceful error naming the speakers (no crash).

    Exercises the ``except`` branch that originally raised a second NameError on
    ``speakers_label``.
    """
    loop = _recording_loop()
    caster = _RecordingCaster(fail=True)
    result = types.SimpleNamespace(value=20, treat=True)
    loop._cast_for_treat(Config(), caster, ["Kitchen"], "Kitchen",
                         result, "rolled 20", None)
    errors = [m for k, m in loop.activity.entries if k == "error"]
    assert errors and "Kitchen" in errors[0]


# --- persistent-connection reuse ----------------------------------------------

class _FakeMediaController:
    def __init__(self):
        self.status = types.SimpleNamespace(player_is_playing=False)

    def update_status(self):
        pass

    def play_media(self, url, content_type):
        pass

    def block_until_active(self, timeout=10):
        pass


class _FakeCast:
    def __init__(self):
        self.media_controller = _FakeMediaController()
        self.socket_client = types.SimpleNamespace(is_connected=True)

    def disconnect(self, blocking=False):
        pass


def test_caster_holds_and_reuses_connection(monkeypatch, tmp_path):
    """The second cast to a speaker reuses the held connection; a stale one rebuilds."""
    caster = Caster(_FakeServer(str(tmp_path)))
    fake_cast = _FakeCast()
    connects = {"n": 0}

    def fake_connect(self, name):
        connects["n"] += 1
        return fake_cast, object()        # browser is unused by the fake path

    monkeypatch.setattr(Caster, "_connect", fake_connect)

    caster.play_media("Kitchen", "http://x/a.wav", "audio/wav")
    caster.play_media("Kitchen", "http://x/a.wav", "audio/wav")
    assert connects["n"] == 1             # reused — no reconnect, no "connecting" chime

    fake_cast.socket_client.is_connected = False   # connection went stale
    caster.play_media("Kitchen", "http://x/a.wav", "audio/wav")
    assert connects["n"] == 2             # rebuilt lazily

    caster.close()
    assert caster._cache == {}            # everything released
