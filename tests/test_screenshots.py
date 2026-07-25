"""Manual screenshots from the live feed.

Kevin's ask: rather than opening the camera vendor's app on a phone and hunting
through an SD card to find the moment he just watched, take the picture from the
app and collect it off the NAS. So: no boxes (it has to line up with the camera's
own recordings), and never auto-deleted (he asked for this one on purpose).
"""

import os
import pathlib

import numpy as np

import d20app
from d20app.detector import PersonDetector
from d20app.snapshots import ScreenshotStore, SnapshotStore
from d20app.webapp import create_app



def test_screenshot_filename_is_sortable_and_names_the_camera(tmp_path):
    store = ScreenshotStore(directory=str(tmp_path))
    name = store.save(b"jpegbytes", "Living Room")
    assert name and name.endswith(".jpg")
    # YYYY-MM-DD_HH-MM-SS_Camera.jpg — sorts chronologically, reads against a clock.
    stamp, _, cam = name[:-4].rpartition("_")
    assert cam == "Living-Room"                  # spaces are not filename-friendly
    assert len(stamp) == len("2026-07-25_14-32-08")
    assert os.path.exists(store.path(name))


def test_two_shots_in_the_same_second_do_not_overwrite(tmp_path):
    store = ScreenshotStore(directory=str(tmp_path))
    a = store.save(b"one", "Entry")
    b = store.save(b"two", "Entry")
    assert a != b
    assert open(store.path(a), "rb").read() == b"one"
    assert open(store.path(b), "rb").read() == b"two"


def test_an_odd_camera_name_cannot_escape_the_folder(tmp_path):
    store = ScreenshotStore(directory=str(tmp_path))
    name = store.save(b"x", "../../etc/passwd")
    assert name and "/" not in name and "\\" not in name
    assert pathlib.Path(store.path(name)).parent == pathlib.Path(str(tmp_path))
    assert store.save(b"x", "") and store.save(b"x", "!!!")     # never empty-named


def test_screenshots_are_never_pruned_unlike_detection_snapshots(tmp_path):
    # The distinction that justifies a separate store: snapshots are evidence the
    # app produced on its own and may roll over; a screenshot was requested.
    shots = ScreenshotStore(directory=str(tmp_path / "shots"))
    for _ in range(SnapshotStore().max_files + 20):
        shots.save(b"x", "Cam")
    kept = list((tmp_path / "shots").glob("*.jpg"))
    assert len(kept) == SnapshotStore().max_files + 20

    snaps = SnapshotStore(directory=str(tmp_path / "snaps"), max_files=5)
    for _ in range(12):
        snaps.save(b"x")
    assert len(list((tmp_path / "snaps").glob("*.jpg"))) <= 5


def test_the_saved_picture_has_no_boxes_drawn_on_it():
    # The whole point: an annotated frame can't be compared against the camera's
    # own footage. plain_jpeg must not go near the box drawing.
    src = pathlib.Path(d20app.__file__).parent.joinpath("detector.py").read_text(
        encoding="utf-8")
    fn = src[src.index("def plain_jpeg"):]
    fn = fn[:fn.index("\n    def ")]
    assert "_draw_boxes" not in fn and "_last_boxes" not in fn
    assert "latest_frame" in fn                  # the published frame, as shown

    det = PersonDetector(source="unused")
    assert det.plain_jpeg() is None              # nothing published yet
    det._publish_frame(np.full((32, 48, 3), 120, np.uint8))
    jpeg = det.plain_jpeg()
    assert jpeg and jpeg[:2] == b"\xff\xd8"      # a real JPEG


def _client_with_detector(det, running=True):
    app = create_app()
    loop = app.config["loop"]
    loop.is_running = lambda: running
    loop.get_detector = lambda name: det if name == "Kitchen" else None
    return app.test_client(), loop


def test_endpoint_saves_the_frame_and_reports_where(tmp_path):
    det = PersonDetector(source="unused")
    det._publish_frame(np.full((32, 48, 3), 90, np.uint8))
    c, loop = _client_with_detector(det)
    loop.screenshots = ScreenshotStore(directory=str(tmp_path))

    body = c.post("/api/live/screenshot", json={"camera": "Kitchen"}).get_json()
    assert body["ok"] and body["file"].endswith("_Kitchen.jpg")
    assert body["url"] == "/screenshots/" + body["file"]
    assert os.path.exists(os.path.join(str(tmp_path), body["file"]))
    # …and it's fetchable at the URL it just handed back.
    assert c.get(body["url"]).status_code == 200
    assert c.get("/screenshots/nope.jpg").status_code == 404


def test_endpoint_says_why_rather_than_failing_silently(tmp_path):
    det = PersonDetector(source="unused")          # no frame published
    c, loop = _client_with_detector(det)
    loop.screenshots = ScreenshotStore(directory=str(tmp_path))

    assert c.post("/api/live/screenshot", json={"camera": "Nope"}).status_code == 409
    assert c.post("/api/live/screenshot", json={}).status_code == 409
    r = c.post("/api/live/screenshot", json={"camera": "Kitchen"})
    assert r.status_code == 409 and "frame" in r.get_json()["error"].lower()

    c2, loop2 = _client_with_detector(det, running=False)
    loop2.screenshots = ScreenshotStore(directory=str(tmp_path))
    r = c2.post("/api/live/screenshot", json={"camera": "Kitchen"})
    assert r.status_code == 409 and "watching" in r.get_json()["error"].lower()


# ---- the button ---------------------------------------------------------------
_BASE = pathlib.Path(d20app.__file__).parent
INDEX = (_BASE / "templates" / "index.html").read_text(encoding="utf-8")
APP_JS = (_BASE / "static" / "app.js").read_text(encoding="utf-8")
STYLE = (_BASE / "static" / "style.css").read_text(encoding="utf-8")


def test_both_feeds_have_a_screenshot_button_beside_the_lock():
    assert 'id="shot-0"' in INDEX and 'id="shot-1"' in INDEX
    for i in (0, 1):
        assert INDEX.index(f'id="shot-{i}"') < INDEX.index(f'id="lock-{i}"')


def test_the_button_does_not_sit_on_top_of_the_lock():
    shot = STYLE[STYLE.index(".feed-shot {"):]
    shot = shot[:shot.index("}")]
    lock = STYLE[STYLE.index(".feed-lock {"):]
    lock = lock[:lock.index("}")]
    assert "right: 8px" in lock and "right: 50px" in shot     # cleared, not stacked


def test_the_button_works_regardless_of_follow_mode():
    # The lock is only meaningful while Follow is routing feeds; a screenshot is
    # about the camera in front of you, so gating it the same way would be wrong.
    fn = APP_JS[APP_JS.index("function renderFeedShots"):]
    fn = fn[:fn.index("\n}")]
    assert "followOn" not in fn
    assert "isRunning" in fn and "liveCam" in fn and "feed2Cam" in fn


def test_the_button_is_rendered_even_when_the_feed_poll_short_circuits():
    # pollFeeds() returns early with Follow off and no second feed, so relying on
    # it alone would leave the main feed without a button.
    live = APP_JS[APP_JS.index("function updateLiveView"):]
    live = live[:live.index("\n}")]
    assert "renderFeedShots()" in live


def test_a_double_tap_cannot_save_two_copies():
    fn = APP_JS[APP_JS.index("async function takeFeedShot"):]
    fn = fn[:fn.index("\n}")]
    assert "btn.disabled = true" in fn and "btn.disabled = false" in fn
    # And a failure has to say so rather than looking like it worked.
    assert "body.error" in fn
