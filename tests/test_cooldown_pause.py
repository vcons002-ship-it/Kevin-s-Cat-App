"""The detection-pause lead time: resume early enough to not miss the next window."""

from d20app.config import Config
from d20app.loop import _cooldown_resume_delay


def test_resume_delay_covers_the_confirm_streak_plus_margin():
    # 4 frames at 10 fps = 0.4s of streak; + 1s margin = 1.4s, but the 3s floor wins.
    cfg = Config(confirm_frames=4, scan_fps=10.0)
    assert _cooldown_resume_delay(cfg) == 3.0
    assert _cooldown_resume_delay(cfg) >= cfg.confirm_frames / cfg.scan_fps


def test_resume_delay_has_a_three_second_floor():
    cfg = Config(confirm_frames=1, scan_fps=30.0)
    assert _cooldown_resume_delay(cfg) == 3.0


def test_resume_delay_scales_up_for_slow_scan_and_long_streak():
    # 10 frames at 2 fps = 5s + 1s margin = 6s (above the 3s floor).
    cfg = Config(confirm_frames=10, scan_fps=2.0)
    assert _cooldown_resume_delay(cfg) == 6.0


def test_resume_leaves_warmup_before_the_window_reopens():
    # The pause must end strictly before the cooldown does, so the streak can rebuild.
    cfg = Config(confirm_frames=4, scan_fps=10.0, cooldown_seconds=600)
    assert cfg.cooldown_seconds - _cooldown_resume_delay(cfg) > 0
