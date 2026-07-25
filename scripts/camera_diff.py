#!/usr/bin/env python3
"""Diff two saved cameras' settings.

Written after deleting and re-adding a camera reliably made it fall ~45 s behind
per minute: a re-added camera fills every unspecified setting from the CURRENT
global defaults, while an older entry keeps whatever was stored when it was last
saved. So a camera you re-add is not necessarily the camera you had.

With a healthy camera and a freshly-broken one both in config.yaml, the cause is
whatever differs between them.

Usage:
    python scripts/camera_diff.py Office Kitchen
    python scripts/camera_diff.py            # list the saved camera names
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from d20app import config as config_mod            # noqa: E402
from d20app.detector import mask_credentials       # noqa: E402

_SECRET = {"password", "username"}


def _show(name, value):
    if name in _SECRET:
        return "(set)" if value else "(empty)"
    if name == "url":
        return mask_credentials(str(value))        # never print an inline password
    return repr(value)


def main() -> None:
    cfg = config_mod.load()
    cams = {c["name"]: c for c in (cfg.cameras or [])
            if isinstance(c, dict) and c.get("name")}
    if not cams:
        sys.exit("No saved cameras in config.yaml.")
    if len(sys.argv) < 3:
        print("saved cameras:")
        for n in sorted(cams):
            print(f"  {n}")
        sys.exit("\nUsage: python scripts/camera_diff.py <good-camera> <bad-camera>")

    a_name, b_name = sys.argv[1], sys.argv[2]
    for n in (a_name, b_name):
        if n not in cams:
            sys.exit(f"No saved camera named {n!r}. Known: {', '.join(sorted(cams))}")

    # Compare the FULLY COERCED specs, which is what the loop actually runs on —
    # a key missing from the stored dict still has an effective value.
    a = config_mod.coerce_camera(cams[a_name], cfg)
    b = config_mod.coerce_camera(cams[b_name], cfg)

    keys = [k for k in sorted(set(a) | set(b)) if a.get(k) != b.get(k)]
    width = max((len(k) for k in keys), default=0)
    print(f"{'setting':<{width}}  {a_name:<28}  {b_name}")
    print("-" * (width + 32 + len(b_name)))
    for k in keys:
        print(f"{k:<{width}}  {_show(k, a.get(k)):<28}  {_show(k, b.get(k))}")
    if not keys:
        print("(identical — the difference is not in the saved settings)")
    else:
        print(f"\n{len(keys)} setting(s) differ.")
    print("\nscan_fps is how many frames a SECOND the app reads. If the camera "
          "streams\nfaster than that, the surplus queues and the feed falls "
          "permanently behind.")


if __name__ == "__main__":
    main()
