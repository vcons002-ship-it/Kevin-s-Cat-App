#!/usr/bin/env python3
"""Entry point: start the web GUI (and with it, the detection loop manager).

Usage::

    python run.py

Then open the printed URL in a browser on the same WiFi to configure the
camera, speaker, sound, and game rules, and to Start/Stop detection.
"""

from __future__ import annotations

import logging
import sys

if sys.version_info < (3, 11):
    sys.exit(
        "Kevin's Cat App requires Python 3.11+ (its dependencies do). "
        f"You're running {sys.version.split()[0]}. "
        "Run it via the venv created by setup.sh: ./venv/bin/python run.py"
    )

from d20app import config as config_mod
from d20app.caster import detect_lan_ip
from d20app.loop import DetectionLoop
from d20app.webapp import create_app


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    cfg = config_mod.load()
    loop = DetectionLoop()
    app = create_app(loop)

    lan_ip = detect_lan_ip()
    port = cfg.web_port
    print("\n  Kevin's Cat App is running!")
    print(f"  Open the GUI:  http://{lan_ip}:{port}   (or http://localhost:{port})\n")

    # threaded=True so discovery endpoints (which block) don't freeze the page.
    app.run(host="0.0.0.0", port=port, threaded=True)


if __name__ == "__main__":
    main()
