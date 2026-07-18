"""H3: a failed first load must not brick the UI.

There is no JS runtime in the test env, so these are *source-level* guards over
the served frontend assets — they lock in the two mechanisms of the fix (api()
swallowing a network error into a handled shape, and the recurring pollers being
registered unconditionally). Behavioural verification is manual (load the page
with the backend down and confirm the retry banner + auto-recovery).
"""

import pathlib

import d20app

_BASE = pathlib.Path(d20app.__file__).parent
APP_JS = (_BASE / "static" / "app.js").read_text(encoding="utf-8")
INDEX = (_BASE / "templates" / "index.html").read_text(encoding="utf-8")


def _fn_body(src: str, header: str) -> str:
    """The body of a top-level function/const, header → its closing brace."""
    start = src.index(header)
    # api() is an arrow-const ending in `};`; the rest close with `}` at col 0.
    end = src.index("\n}", start)
    return src[start:end]


def test_api_helper_swallows_network_errors():
    # A network blip / unreachable server must resolve to a handled shape, not
    # reject out of api() (which would abort init() before its pollers register).
    body = _fn_body(APP_JS, "const api = async")
    assert "catch" in body
    assert "netError: true" in body        # the handled-shape marker
    assert "return { ok: false" in body


def test_pollers_start_even_when_initial_load_fails():
    # The recurring polls live in startPolling(), which init() calls
    # unconditionally after its try/catch — so a failed first load can't stop
    # them, and they recover the UI once the backend is reachable again.
    startpoll = _fn_body(APP_JS, "function startPolling(")
    assert startpoll.count("setInterval(") == 3
    for fn in ("refreshStatus", "loadLog", "loadCats", "rotateCatFeed"):
        assert fn in startpoll

    init = _fn_body(APP_JS, "async function init(")
    assert "try {" in init and "catch" in init
    assert "startPolling()" in init
    # Called AFTER the catch → not gated by a successful load.
    assert init.index("startPolling()") > init.index("catch")
    # And the pollers are not inside the retry-able data load.
    assert "setInterval(" not in _fn_body(APP_JS, "async function loadInitialData(")


def test_config_saves_are_serialized_and_coalesced():
    # Audit M4: every control posts the WHOLE form, so two overlapping saves can
    # land out of order and the earlier response silently reverts the later edit.
    # Guard the three pieces of the fix (behaviour verified in a browser: 8 rapid
    # edits → max 1 request in flight, 3 posts, final value = the last edit).
    save = _fn_body(APP_JS, "async function saveConfig(")
    assert "if (saveInFlight) { savePending = true; return; }" in save   # never overlap
    assert "savePending = false;" in save and "} while (savePending);" in save  # re-run once
    # The re-run must RE-GATHER the form, not replay a stale snapshot, so the
    # follow-up save carries every edit that arrived while the first was away.
    assert save.index("gatherConfig()") > save.index("savePending = false;")
    assert "saveInFlight = false;" in save                               # released in finally


def test_startup_error_banner_present_and_wired():
    assert 'id="init-error"' in INDEX
    assert 'id="init-retry"' in INDEX
    # The retry button re-runs only the data load (no re-wire, no stacked pollers).
    assert "retryInitialLoad" in APP_JS
    assert "loadInitialData().then(applyLoadResult)" in APP_JS


def test_banner_reflects_backend_reachability():
    # loadConfig() signals reachability (the /api/config bootstrap), which
    # loadInitialData() propagates so the banner shows on a genuine outage — not
    # only on an unexpected throw. Guards against loadConfig silently returning.
    cfg = _fn_body(APP_JS, "async function loadConfig(")
    assert "return false" in cfg and "return true" in cfg
    load = _fn_body(APP_JS, "async function loadInitialData(")
    assert "return reached" in load
