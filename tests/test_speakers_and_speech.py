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
