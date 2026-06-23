"""Tests for RTSP credential injection/encoding and masking for logs."""

from types import SimpleNamespace

from d20app.detector import mask_credentials
from d20app.loop import _camera_source


def _cfg(**kw):
    base = {"camera_url": "", "camera_username": "", "camera_password": ""}
    base.update(kw)
    return SimpleNamespace(**base)


def test_injects_plain_credentials():
    cfg = _cfg(camera_url="rtsp://192.168.1.5:554/s", camera_username="admin",
               camera_password="pass")
    assert _camera_source(cfg) == "rtsp://admin:pass@192.168.1.5:554/s"


def test_percent_encodes_special_characters():
    cfg = _cfg(camera_url="rtsp://10.0.0.2/live", camera_username="a d",
               camera_password="p@ss:w/rd")
    assert _camera_source(cfg) == "rtsp://a%20d:p%40ss%3Aw%2Frd@10.0.0.2/live"


def test_embedded_credentials_left_untouched():
    cfg = _cfg(camera_url="rtsp://u:p@host/s", camera_username="x",
               camera_password="y")
    assert _camera_source(cfg) == "rtsp://u:p@host/s"


def test_no_username_returns_url_unchanged():
    cfg = _cfg(camera_url="rtsp://host/s")
    assert _camera_source(cfg) == "rtsp://host/s"


def test_username_only_no_password():
    cfg = _cfg(camera_url="rtsp://host/s", camera_username="admin")
    assert _camera_source(cfg) == "rtsp://admin@host/s"


def test_mask_hides_password():
    assert mask_credentials("rtsp://admin:secret@h:554/s") == "rtsp://admin:***@h:554/s"


def test_mask_noop_without_credentials():
    assert mask_credentials("rtsp://h:554/s") == "rtsp://h:554/s"
