"""Tests for the new config fields and the preview-frame grabber."""

from d20app import config as config_mod
from d20app.detector import grab_frame_jpeg


def test_grab_frame_returns_none_on_bad_source():
    # An obviously invalid source must fail gracefully (no exception, no hang).
    assert grab_frame_jpeg("not-a-real-source", skip=0) is None


def test_config_coerces_new_fields(tmp_path):
    path = str(tmp_path / "config.yaml")
    cfg = config_mod.update(
        {"detect_size": "768", "scan_fps": "5", "roi": [10, 20, 100, 80],
         "confirm_frames": "4"},
        path=path,
    )
    assert cfg.detect_size == 768 and isinstance(cfg.detect_size, int)
    assert cfg.scan_fps == 5.0 and isinstance(cfg.scan_fps, float)
    assert cfg.confirm_frames == 4
    assert cfg.roi == [10, 20, 100, 80]


def test_config_defaults_present():
    cfg = config_mod.Config()
    assert cfg.detect_size == 300        # reverted from 512 to protect person recall
    assert cfg.person_confidence == 0.4
    assert cfg.scan_fps == 10.0
    assert cfg.roi is None
