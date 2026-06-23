#!/usr/bin/env python3
"""Diagnose the configured camera end-to-end.

Run it from the project folder:

    ./venv/bin/python check_camera.py

It reads your config.yaml, opens the camera exactly the way the app does, and
reports: can it open the stream, are frames actually decoding (and at what
resolution), and would the person-detector fire on what it sees. It also writes
``snapshot.jpg`` so you can eyeball framing and lighting.

This answers the common "the login works but nothing happens" case — usually a
codec the app can't decode (e.g. an H.265 main stream) or framing/confidence.
"""

from __future__ import annotations

import sys
import time

import cv2  # noqa: E402

from d20app import config as config_mod
from d20app.detector import (
    CLASSES,
    PERSON_CLASS_ID,
    PersonDetector,
    _quiet_cv2_logs,
    mask_credentials,
)
from d20app.loop import _camera_source

_quiet_cv2_logs(cv2)


def main() -> int:
    cfg = config_mod.load()
    if not cfg.camera_url:
        print("No camera configured yet — set one in the GUI first.")
        return 1

    source = _camera_source(cfg)
    print(f"Opening: {mask_credentials(source)}")
    cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        print("❌ Could NOT open the stream (auth / URL / network / wrong path).")
        print("   See the 'Camera connects in VLC but not here' part of the README.")
        return 2
    print("✅ Stream opened. Trying to read frames…")

    frames, last, t0 = 0, None, time.time()
    for _ in range(60):
        ok, frame = cap.read()
        if ok and frame is not None:
            frames += 1
            last = frame
    dt = max(time.time() - t0, 0.001)
    cap.release()

    if last is None:
        print(f"❌ Stream opened but decoded 0 frames in {dt:.1f}s.")
        print("   Most likely the codec can't be decoded here — many cameras")
        print("   default the MAIN stream to H.265/HEVC. Point the app at the")
        print("   camera's H.264 SUB-stream URL instead and try again.")
        return 3

    h, w = last.shape[:2]
    print(f"✅ Decoded {frames}/60 frames at {w}×{h} (~{frames / dt:.1f} fps).")
    cv2.imwrite("snapshot.jpg", last)
    print("   Wrote snapshot.jpg — open it to check framing and lighting.")

    # Run the actual detector on the last frame and report the best person score.
    det = PersonDetector(source=source, confidence=cfg.person_confidence)
    blob = cv2.dnn.blobFromImage(
        cv2.resize(last, (300, 300)), 0.007843, (300, 300), 127.5
    )
    net = det._ensure_net()
    net.setInput(blob)
    detections = net.forward()

    best_person, seen = 0.0, []
    for k in range(detections.shape[2]):
        score = float(detections[0, 0, k, 2])
        cid = int(detections[0, 0, k, 1])
        if cid == PERSON_CLASS_ID:
            best_person = max(best_person, score)
        if score >= 0.30 and 0 <= cid < len(CLASSES):
            seen.append(f"{CLASSES[cid]} {score:.2f}")

    print(
        f"\nPerson detector: best person score = {best_person:.2f} "
        f"(your threshold = {cfg.person_confidence})."
    )
    print("   Objects seen (≥0.30): " + (", ".join(seen) if seen else "none"))
    if best_person >= cfg.person_confidence:
        print("✅ A person WOULD trigger a roll on this frame.")
    else:
        print("⚠ No person above your threshold in this frame. Stand clearly in")
        print("   view and re-run, or lower 'Person confidence' in the GUI.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
