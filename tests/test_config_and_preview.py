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
    assert cfg.person_confidence == 0.5
    assert cfg.scan_fps == 10.0
    assert cfg.roi is None
    # New tuning defaults (v0.4.0).
    assert cfg.label_floor == 0.55
    assert cfg.pause_during_cooldown is True
    assert cfg.motion_sensitivity == "medium"
    assert cfg.motion_min_area_frac == 0.003
    assert cfg.motion_diff_threshold == 25
    assert cfg.motion_min_blob_px == 14
    assert cfg.cameras == []
    assert cfg.keep_speakers_warm is False


def test_config_coerces_motion_fields_and_round_trips_cameras(tmp_path):
    path = str(tmp_path / "config.yaml")
    cams = [{"name": "Kitchen", "url": "rtsp://1.2.3.4/s",
             "username": "admin", "password": "secret"}]
    cfg = config_mod.update(
        {"label_floor": "0.6", "motion_diff_threshold": "30",
         "motion_min_area_frac": "0.005", "pause_during_cooldown": "false",
         "cameras": cams},
        path=path,
    )
    assert cfg.label_floor == 0.6 and isinstance(cfg.label_floor, float)
    assert cfg.motion_diff_threshold == 30 and isinstance(cfg.motion_diff_threshold, int)
    assert cfg.motion_min_area_frac == 0.005
    assert cfg.pause_during_cooldown is False
    # Saved cameras (incl. passwords) round-trip through YAML unchanged.
    assert config_mod.load(path).cameras == cams
