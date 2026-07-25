"""The Cat-cam button's ambient state.

The old button flashed continuously whenever a cat had been seen recently, which
made the flash its normal appearance and therefore meaningless — Kevin's words:
"it flashes too much, for everything". The glyph now carries status passively and
animation is reserved for the result of a click.

State is derived from records the app already keeps (sighting `source` tags and
per-room sweep verdicts) rather than from a second store that could drift.
"""

import time

from d20app.loop import DetectionLoop


def _loop(tmp_path):
    loop = DetectionLoop()
    loop.cats.path = str(tmp_path / "cats.log")     # never touch the real log
    loop.cats._sightings.clear()
    return loop


def _saw(loop, camera, source, ago=0.0):
    """Record a sighting `ago` seconds back. `_sightings` is oldest-first;
    recent() reverses it."""
    entry = {"ts": time.time() - ago, "camera": camera, "source": source,
             "box": [0, 0, 1, 1], "score": 0.9, "region": "", "label": "cat"}
    items = sorted(list(loop.cats._sightings) + [entry], key=lambda s: s["ts"])
    loop.cats._sightings.clear()
    loop.cats._sightings.extend(items)


def test_idle_when_nothing_has_happened(tmp_path):
    assert _loop(tmp_path).button_state()["state"] == "idle"


def test_one_room_moving_is_active(tmp_path):
    loop = _loop(tmp_path)
    _saw(loop, "Office", "motion", ago=5)
    out = loop.button_state()
    assert out["state"] == "active" and out["moving"] == ["Office"]


def test_two_rooms_moving_is_multi(tmp_path):
    loop = _loop(tmp_path)
    _saw(loop, "Office", "motion", ago=5)
    _saw(loop, "Hallway", "track", ago=2)        # fused tracks count as moving too
    assert loop.button_state()["state"] == "multi"


def test_movement_ages_out_of_the_window(tmp_path):
    loop = _loop(tmp_path)
    _saw(loop, "Office", "motion", ago=45)       # older than the 30 s window
    assert loop.button_state()["state"] == "idle"


def test_a_sweep_hit_makes_it_resting(tmp_path):
    loop = _loop(tmp_path)
    loop.record_sweep("Basement", True)
    assert loop.button_state()["state"] == "resting"


def test_resting_does_not_time_out(tmp_path):
    # A still cat stays still. Only a newer verdict for that room, or motion,
    # is evidence to the contrary — so this must not expire on a timer.
    loop = _loop(tmp_path)
    loop.record_sweep("Basement", True)
    with loop._scan_lock:
        loop._scan_last["Basement"]["ts"] -= 3600      # an hour ago
    assert loop.button_state()["state"] == "resting"


def test_a_later_sweep_of_the_same_room_clears_it(tmp_path):
    loop = _loop(tmp_path)
    loop.record_sweep("Basement", True)
    loop.record_sweep("Basement", False)
    assert loop.button_state()["state"] == "idle"


def test_another_rooms_empty_sweep_does_not_clear_it(tmp_path):
    # The reason verdicts are per room: seven cameras scan on staggered timers, so
    # a global "most recent sweep" would let any empty room wipe out a real cat
    # found three seconds earlier somewhere else.
    loop = _loop(tmp_path)
    loop.record_sweep("Basement", True)
    loop.record_sweep("Office", False)
    out = loop.button_state()
    assert out["state"] == "resting" and out["resting"] == ["Basement"]


def test_motion_outranks_a_sleeping_cat(tmp_path):
    # Motion is the fresher fact.
    loop = _loop(tmp_path)
    loop.record_sweep("Basement", True)
    _saw(loop, "Office", "motion", ago=3)
    assert loop.button_state()["state"] == "active"


def test_a_sweep_sighting_is_not_treated_as_movement(tmp_path):
    # A still-scan hit means she's THERE, not that she moved — otherwise every
    # sweep would put the button into its "active" state for 30 s.
    loop = _loop(tmp_path)
    _saw(loop, "Basement", "still-scan", ago=2)
    _saw(loop, "Kitchen", "find", ago=2)
    assert loop.button_state()["state"] == "idle"     # no sweep VERDICTS recorded


def test_resting_rooms_are_reported_for_the_status_line(tmp_path):
    # "Never more than two glyphs" — a cat resting in two rooms is still one
    # sleepy cat, and the count belongs in the text.
    loop = _loop(tmp_path)
    loop.record_sweep("Basement", True)
    loop.record_sweep("Office", True)
    out = loop.button_state()
    assert out["state"] == "resting" and out["resting"] == ["Basement", "Office"]


# ---- the button itself --------------------------------------------------------
import pathlib                                  # noqa: E402

import d20app                                   # noqa: E402

_BASE = pathlib.Path(d20app.__file__).parent
INDEX = (_BASE / "templates" / "index.html").read_text(encoding="utf-8")
APP_JS = (_BASE / "static" / "app.js").read_text(encoding="utf-8")
STYLE = (_BASE / "static" / "style.css").read_text(encoding="utf-8")


def test_the_glyph_slot_is_fixed_width():
    # Sized for two emoji so the button never resizes between ambient states, and
    # a single glyph stays optically centred — padding one side would not.
    css = STYLE[STYLE.index(".show-cat-emoji {"):]
    css = css[:css.index("}")]
    assert "width:" in css and "justify-content: center" in css


def test_ambient_states_never_animate():
    # The whole point of the rework: a signal that fires constantly stops being a
    # signal. Only the click-triggered classes may animate.
    for cls in (".show-cat.scanning .show-cat-dots::after", ".show-cat.found"):
        assert cls in STYLE
    assert ".show-cat.detecting" not in STYLE      # the old always-on flash is gone
    assert "detecting" not in APP_JS.split("function renderCatButton")[1][:1200]


def test_every_state_has_a_glyph_and_a_rank():
    block = APP_JS[APP_JS.index("const CAT_STATES"):]
    block = block[:block.index("};")]
    for state, glyph in (("idle", "🔍"), ("resting", "😴"),
                         ("active", "🐱"), ("multi", "🐱🐱")):
        assert state in block and glyph in block
    # Ranks drive the upgrade-only bounce; without them a downgrade would bounce.
    assert block.count("rank:") == 4


def test_the_label_is_the_same_in_every_ambient_state():
    fn = APP_JS[APP_JS.index("function renderCatButton"):]
    fn = fn[:fn.index("\n}")]
    assert fn.count('label.textContent') == 1
    assert '"Show me the cat!"' in fn


def test_a_click_owns_the_button_until_it_lands():
    # Without this the 1.2 s cats poll would overwrite "Cat spotted!" mid-hold.
    assert "catBtnBusy = true" in APP_JS
    show = APP_JS[APP_JS.index("async function showCat()"):]
    show = show[:show.index("async function showCatJump")]
    assert show.count("catBtnBusy = false") == 2       # the hit path and the miss path
    fn = APP_JS[APP_JS.index("function renderCatButton"):]
    assert "if (catBtnBusy) return;" in fn[:400]


def test_the_result_is_held_before_scrolling_away():
    fn = APP_JS[APP_JS.index("function catBtnResult"):]
    fn = fn[:fn.index("\n}")]
    assert "2000" in fn                                # the 2 s hold
    show = APP_JS[APP_JS.index("async function showCat()"):]
    show = show[:show.index("async function showCatJump")]
    # …and the hold happens BEFORE the scroll, or it buys nothing.
    assert show.index("catBtnResult(\"😻\"") < show.index("scrollIntoView")


def test_the_landing_state_is_re_derived_not_assumed():
    # "Re-check the conditions rather than hardcoding the landing state" — a miss
    # clears a stale sleepy cat, and motion during the hold wins.
    show = APP_JS[APP_JS.index("async function showCat()"):]
    show = show[:show.index("async function showCatJump")]
    assert show.count("await loadCats();") == 2
