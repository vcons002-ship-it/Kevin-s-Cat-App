"""The in-app user & workflow guide (0.38.0): /guide serves docs/USER_GUIDE.md
through the tiny built-in markdown converter, and the GUI links to it."""

from d20app.webapp import _md_to_html, create_app


def _client():
    return create_app().test_client()


def test_guide_serves_the_markdown_as_html():
    r = _client().get("/guide")
    assert r.status_code == 200 and "text/html" in r.content_type
    page = r.get_data(as_text=True)
    assert "<h1>Kevin's Cat App — User &amp; Workflow Guide</h1>" in page.replace(
        "&#x27;", "'") or "User &amp; Workflow Guide" in page
    # the load-bearing sections made it through the converter
    for needle in ("default workflow", "Track fusion", "escalation ladder",
                   "Only YOLO confirms", "Live-camera escalation"):
        assert needle in page, needle


def test_index_links_the_guide_without_the_workflow_line():
    page = _client().get("/").get_data(as_text=True)
    assert 'href="/guide"' in page
    # The "Workflow: …" summary was removed: it read the GLOBAL config while model,
    # accelerator and track_fusion are per-camera, and it showed the *requested*
    # accelerator, so it could claim "onnx-cuda" while a camera ran TensorRT or had
    # silently fallen back to CPU. `ran_on`/`fallback` per camera is the honest one.
    assert 'id="workflow-line"' not in page


def test_md_converter_basics():
    html = _md_to_html(
        "# Title\n\nSome **bold** and `code` and [a link](/guide).\n\n"
        "- item one\n- item two\n\n```\nplain <code> block\n```\n\n---\n")
    assert "<h1>Title</h1>" in html
    assert "<strong>bold</strong>" in html and "<code>code</code>" in html
    assert "<a href='/guide'>a link</a>" in html
    assert "<ul>" in html and "<li>item one</li>" in html
    assert "plain &lt;code&gt; block" in html          # fenced block is escaped
    assert "<hr>" in html


def test_md_converter_escapes_html():
    html = _md_to_html("hello <script>alert(1)</script> world")
    assert "<script>" not in html and "&lt;script&gt;" in html
