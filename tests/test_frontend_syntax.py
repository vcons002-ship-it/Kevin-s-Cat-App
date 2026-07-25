"""A syntax guard for the served JavaScript.

0.59.0 shipped an app.js with an orphaned `};` — the tail of a handler whose
opening lines were deleted. Every Python test passed, because none of them parse
the JavaScript. The browser hit a SyntaxError, the whole script died before its
first request, and the GUI simply never loaded: the server log showed the page
and its two static files and then nothing at all.

There's no JS engine in the test environment, so this is a bracket/quote balance
check rather than a real parse. That is enough to catch a truncated or
over-deleted block, which is the failure edits actually cause. It must handle
regex literals, or it flags every file containing one.
"""

import pathlib

import pytest

import d20app

_STATIC = pathlib.Path(d20app.__file__).parent / "static"

# A `/` begins a regex (not division) when the previous meaningful token can't end
# an expression. This is the standard heuristic and covers everything in this app.
_REGEX_OK_AFTER = set("(,=:[!&|?{};+-*%~^<>") | {"return", "typeof", "case", "in", "of", "new", "delete"}


def _balance(src: str):
    """(depth, unterminated_string, line_of_first_negative) for a JS source."""
    i, n = 0, len(src)
    depth, line, first_bad = 0, 1, None
    prev_token = ""          # last non-space char or word seen at code level
    while i < n:
        c, nxt = src[i], (src[i + 1] if i + 1 < n else "")
        if c == "\n":
            line += 1; i += 1; continue
        if c in " \t\r":
            i += 1; continue
        if c == "/" and nxt == "/":
            i = src.find("\n", i)
            if i < 0:
                break
            continue
        if c == "/" and nxt == "*":
            end = src.find("*/", i + 2)
            line += src.count("\n", i, end if end > 0 else n)
            i = (end + 2) if end > 0 else n
            continue
        if c == "/" and (prev_token in _REGEX_OK_AFTER or prev_token == ""):
            # Regex literal: skip to the unescaped closing slash.
            j = i + 1
            in_class = False
            while j < n:
                if src[j] == "\\":
                    j += 2; continue
                if src[j] == "[":
                    in_class = True
                elif src[j] == "]":
                    in_class = False
                elif src[j] == "/" and not in_class:
                    break
                elif src[j] == "\n":
                    break            # unterminated — treat as division after all
                j += 1
            i = j + 1
            prev_token = "re"
            continue
        if c in "'\"`":
            quote, j = c, i + 1
            while j < n:
                if src[j] == "\\":
                    j += 2; continue
                if src[j] == quote:
                    break
                if src[j] == "\n":
                    line += 1
                    if quote != "`":
                        return depth, line, first_bad     # newline in a plain string
                j += 1
            if j >= n:
                return depth, line, first_bad
            i = j + 1
            prev_token = "str"
            continue
        if c in "{([":
            depth += 1
        elif c in "})]":
            depth -= 1
            if depth < 0 and first_bad is None:
                first_bad = line
        prev_token = c if not c.isalnum() else prev_token
        if c.isalpha():
            j = i
            while j < n and (src[j].isalnum() or src[j] == "_"):
                j += 1
            prev_token = src[i:j]
            i = j
            continue
        i += 1
    return depth, None, first_bad


@pytest.mark.parametrize("name", sorted(p.name for p in _STATIC.glob("*.js")))
def test_served_javascript_is_balanced(name):
    src = (_STATIC / name).read_text(encoding="utf-8")
    depth, bad_string_line, first_bad = _balance(src)
    assert bad_string_line is None, f"{name}: unterminated string near line {bad_string_line}"
    assert first_bad is None, f"{name}: a closing bracket with nothing open, line {first_bad}"
    assert depth == 0, f"{name}: {depth} bracket(s) left open at end of file"


def test_the_checker_catches_the_bug_that_shipped():
    # The exact shape of the 0.59.0 break: a handler's opening lines removed,
    # leaving its body and closing `};` behind.
    broken = """
function wire() {
  $("a").onclick = () => { go(); };
    if (liveOn) { liveOn = false; refreshStatus(); }
  };
}
"""
    depth, _s, first_bad = _balance(broken)
    assert first_bad is not None or depth != 0


def test_the_checker_is_not_fooled_by_regexes_or_strings():
    ok = r"""
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;" }[c]));
const div = total / count / 2;
const tpl = `a ${x} { not a brace }`;
const s = "he said \" and { too";
"""
    assert _balance(ok) == (0, None, None)
