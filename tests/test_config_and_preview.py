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
    assert cfg.detector_model == "yolo11n"


def test_coerce_blank_or_none_numeric_keeps_value_instead_of_raising():
    # H1: auto-saving controls send "" (cleared input) or null; a numeric field
    # must not raise (which 500'd the whole POST /api/config). With no ``current``
    # value it falls back to the default; the proven-at-runtime audit cases:
    assert config_mod._coerce("", 0) == 0
    assert config_mod._coerce(None, 0) == 0
    assert config_mod._coerce("   ", 7) == 7          # whitespace-only counts as blank
    assert config_mod._coerce("", 30.0) == 30.0
    assert config_mod._coerce(None, 30.0) == 30.0
    # Unparseable garbage also falls back rather than raising.
    assert config_mod._coerce("abc", 0) == 0
    # keep-current: when a current value is supplied, a blank/bad input keeps it
    # (not the default). 0 is a valid current value — not treated as "missing".
    assert config_mod._coerce("", 0, 42) == 42
    assert config_mod._coerce(None, 30.0, 12.5) == 12.5
    assert config_mod._coerce("abc", 0, 42) == 42
    assert config_mod._coerce("", 5, 0) == 0
    # Well-formed values still coerce exactly as before (type from the default).
    assert config_mod._coerce("5", 0) == 5 and isinstance(config_mod._coerce("5", 0), int)
    assert config_mod._coerce("4.9", 0) == 4           # int keeps its truncating behaviour
    assert config_mod._coerce("2.5", 0.0) == 2.5
    assert config_mod._coerce("9", 0, 42) == 9         # a valid value overrides current


def test_update_blank_numeric_keeps_current_saved_value(tmp_path):
    # keep-current: a cleared numeric control auto-saves "" (a client may send
    # null). update() must persist without raising (was HTTP 500) and preserve
    # the last saved value rather than reset it to the default — so a spurious
    # blank auto-save can't clobber a real setting (audit H1/M3).
    path = str(tmp_path / "config.yaml")
    assert config_mod.Config().cooldown_seconds != 45     # 45 is a non-default value
    config_mod.update({"cooldown_seconds": "45"}, path=path)
    cfg = config_mod.update(
        {"cooldown_seconds": "", "scan_fps": None, "confirm_frames": "3"},
        path=path,
    )
    assert cfg.cooldown_seconds == 45                     # kept, not reset to default
    assert cfg.scan_fps == config_mod.Config().scan_fps   # never set → current == default
    assert cfg.confirm_frames == 3                        # a well-formed field still lands
    # And it actually saved — reload from disk matches.
    assert config_mod.load(path).cooldown_seconds == 45


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
