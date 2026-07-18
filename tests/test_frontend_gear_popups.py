"""#117: per-section ⚙ popups hold a section's settings.

No JS runtime in the suite, so these are structural guards over the served markup
and script (behaviour was verified in a browser). The Cat-cam part is a
**relocation** — the same controls, same ids, same wiring, just moved out of the
inline collapsibles — so the tests check containment rather than re-testing the
controls themselves.
"""

import pathlib
from html.parser import HTMLParser

import d20app

_BASE = pathlib.Path(d20app.__file__).parent
INDEX = (_BASE / "templates" / "index.html").read_text(encoding="utf-8")
APP_JS = (_BASE / "static" / "app.js").read_text(encoding="utf-8")

_VOID = {"input", "img", "br", "hr", "meta", "link", "source", "option"}


class _Ancestry(HTMLParser):
    """Record, for every element carrying an id, the ids of its ancestors."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self._stack = []
        self.parents = {}      # id -> [ancestor ids, outermost first]
        self.gears = []        # (data-popup) targets
        self.popups = set()

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        el_id = a.get("id")
        if el_id:
            self.parents[el_id] = [i for i in self._stack if i]
        classes = (a.get("class") or "").split()
        if "gear" in classes and a.get("data-popup"):
            self.gears.append(a["data-popup"])
        if "gear-popup" in classes and el_id:
            self.popups.add(el_id)
        if tag not in _VOID:
            self._stack.append(el_id)

    def handle_endtag(self, tag):
        if tag not in _VOID and self._stack:
            self._stack.pop()


def _tree():
    p = _Ancestry()
    p.feed(INDEX)
    return p


def test_three_gears_open_three_section_popups():
    t = _tree()
    assert set(t.gears) == {"catcam-settings", "live-settings", "log-settings"}
    assert t.popups >= {"catcam-settings", "live-settings", "log-settings"}


def test_cat_cam_controls_were_relocated_into_the_popup():
    # A relocation, not a rebuild: every control keeps its id and now sits inside
    # the gear popup instead of the inline collapsibles.
    t = _tree()
    moved = ["cat_scan_interval", "cat_scan_model", "scan_tiling",
             "scan_tile_overlap", "scan_frames", "scan_confidence",
             "find_scan", "find_model", "find_tiling", "find_tile_overlap",
             "find_confidence"]
    for cid in moved:
        assert cid in t.parents, f"{cid} disappeared — this was meant to be a move"
        assert "catcam-settings" in t.parents[cid], f"{cid} is not in the gear popup"
    # …and the collapsibles they used to live in are gone from the scroll.
    assert "scan-settings" not in t.parents and "find-settings" not in t.parents


def test_new_settings_live_in_their_own_sections_popup():
    t = _tree()
    for cid in ("follow_hold_seconds", "follow_persist_seconds"):
        assert "live-settings" in t.parents[cid]
    assert "log-settings" in t.parents["fusion_debug"]


def test_gear_settings_save_through_the_normal_hot_reload_path():
    # The knobs are meant to be twisted while watching — they must ride the same
    # auto-save/hot-reload path as every other setting, not need a restart.
    for key in ("follow_hold_seconds", "follow_persist_seconds", "fusion_debug"):
        assert key in APP_JS
    save_list = APP_JS[APP_JS.index('"cat_scan_model", "scan_tiling"'):]
    save_list = save_list[:save_list.index("]")]
    for key in ("follow_hold_seconds", "follow_persist_seconds", "fusion_debug"):
        assert key in save_list, f"{key} is not auto-saved on change"
    # and they're actually sent in the payload
    gather = APP_JS[APP_JS.index("function gatherConfig"):]
    gather = gather[:gather.index("\n}")]
    for key in ("follow_hold_seconds", "follow_persist_seconds", "fusion_debug"):
        assert key in gather


def test_popups_are_dismissible():
    assert "closeGearPopups" in APP_JS
    assert 'e.key === "Escape"' in APP_JS          # Escape closes
    assert '.closest(".gear-popup")' in APP_JS     # click-outside closes
