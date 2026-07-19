#!/usr/bin/env python3
"""Does the app's synchronous read path see CURRENT frames, or a growing backlog?

This settles one question before any motion tuning is trusted: when the loop reads
a camera slower than the camera streams, does FFmpeg **queue** the frames it
couldn't hand over (so ``cap.read()`` returns progressively older ones) or **drop**
them (so every read is near-current)?

It matters because the motion prefilter diffs *consecutive reads*. If frames queue,
those two frames are ~1/camera_fps apart; if they're dropped, they're ~1/scan_fps
apart. A cat walks ~3x further in 200 ms than in 67 ms, and the motion verdict is
an **area** test — so the same cat can clear the bar in one case and not the other.
An offline harness that samples a clip at scan_fps is only a valid proxy for the
live app in the "dropped" case.

How it decides, without needing you to wave at the camera: it reads slowly for a
while (like the app does), then reads as fast as it can and times each read. Frames
already sitting in a queue come back almost instantly; a live stream makes each
read wait for the next frame to arrive.

Usage (opens the stream exactly like the app does, credentials and all):
    python scripts/rtsp_latency_probe.py --camera Entry
    python scripts/rtsp_latency_probe.py --url rtsp://user:pass@host/stream --fps 5

Reads only — it never writes config and never touches the running app. Safe to run
while the app is watching, though it does open a second connection to the camera.
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from d20app import config as config_mod           # noqa: E402
from d20app.detector import _open_capture, mask_credentials   # noqa: E402


def _resolve(args):
    """(label, source) for the requested camera, from config unless --url given."""
    if args.url:
        return "(--url)", args.url
    cfg = config_mod.load()
    targets = config_mod.camera_targets(cfg)
    if not targets:
        sys.exit("No cameras configured. Pass --url instead.")
    if args.camera:
        for spec in targets:
            if spec.get("name") == args.camera:
                return spec["name"], spec["source"]
        names = ", ".join(repr(s.get("name")) for s in targets)
        sys.exit(f"No watched camera named {args.camera!r}. Known: {names}")
    spec = targets[0]
    return spec["name"], spec["source"]


def _read(cap, cv2=None):
    """One read → (ok, seconds_it_blocked, frame, stream_position_ms).

    ``CAP_PROP_POS_MSEC`` is the frame's own timestamp in the stream, which is
    what tells us how far apart two frames are *in the video* — as opposed to how
    far apart we happened to read them. Those differ whenever frames queue, which
    is the whole point of this probe. Not every camera provides it.
    """
    t0 = time.perf_counter()
    ok, frame = cap.read()
    blocked = time.perf_counter() - t0
    pos = None
    if cv2 is not None:
        try:
            raw = cap.get(cv2.CAP_PROP_POS_MSEC)
            pos = float(raw) if raw and raw > 0 else None
        except Exception:       # noqa: BLE001 — an unsupported property is fine
            pos = None
    return (ok and frame is not None), blocked, frame, pos


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--camera", help="watched camera name (default: the first one)")
    ap.add_argument("--url", help="stream URL instead of a configured camera")
    ap.add_argument("--fps", type=float, default=None,
                    help="read rate to imitate (default: the camera's scan_fps)")
    ap.add_argument("--seconds", type=float, default=20.0,
                    help="how long to read slowly before testing for a backlog")
    ap.add_argument("--drain", type=int, default=40,
                    help="how many fast reads to time afterwards")
    ap.add_argument("--save", metavar="DIR",
                    help="also write first/last frames here to eyeball the lag")
    args = ap.parse_args()

    label, source = _resolve(args)
    scan_fps = args.fps
    if scan_fps is None:
        cfg = config_mod.load()
        spec = next((s for s in config_mod.camera_targets(cfg)
                     if s.get("name") == label), None)
        scan_fps = float(spec["scan_fps"]) if spec else 5.0
    interval = 1.0 / max(0.1, scan_fps)

    print(f"camera : {label}  {mask_credentials(str(source))}")
    print(f"reading: {scan_fps:g} fps for {args.seconds:g}s, then {args.drain} fast reads\n")

    cap = _open_capture(source)
    if not cap.isOpened():
        cap.release()
        sys.exit(f"could not open {mask_credentials(str(source))}")

    import cv2

    try:
        # --- native rate. Reading flat out straight after connecting measures how
        # fast a BACKLOG drains, not how fast frames arrive — on a queueing stream
        # that over-reports badly. So: drain until reads stop coming back instantly,
        # then measure over a steady window.
        for _ in range(5):                       # connection warm-up
            _read(cap, cv2)
        drained, t_drain = 0, time.perf_counter()
        while time.perf_counter() - t_drain < 8.0:
            ok, blocked, _f, _p = _read(cap, cv2)
            drained += 1
            if ok and blocked > 0.008:           # waited for a frame → queue is empty
                break
        t0, n = time.perf_counter(), 0
        while time.perf_counter() - t0 < 4.0:
            ok, _dt, _f, _p = _read(cap, cv2)
            if ok:
                n += 1
        native = n / max(1e-6, time.perf_counter() - t0)
        print(f"stream delivers ~{native:.1f} fps "
              f"(measured after draining {drained} queued frames)")
        if native <= scan_fps * 1.15:
            print("  NOTE: the camera is no faster than the read rate, so there's "
                  "nothing to queue —\n        this test can't distinguish the two "
                  "cases. Raise the camera's fps or lower --fps.\n")

        # --- phase 1: read slowly, exactly like the app's synchronous path.
        first_frame = None
        slow_reads, slow_block, stream_gaps = 0, [], []
        prev_pos = None
        t_start = time.perf_counter()
        while time.perf_counter() - t_start < args.seconds:
            ok, blocked, frame, pos = _read(cap, cv2)
            if ok:
                slow_reads += 1
                slow_block.append(blocked)
                if pos is not None and prev_pos is not None and pos > prev_pos:
                    stream_gaps.append(pos - prev_pos)
                prev_pos = pos
                if first_frame is None:
                    first_frame = frame
            time.sleep(max(0.0, interval - blocked))
        expected = native * args.seconds
        print(f"\nphase 1: read {slow_reads} frames in {args.seconds:g}s "
              f"({slow_reads / args.seconds:.1f}/s); the camera produced ~{expected:.0f} "
              f"in that time")
        print(f"         each slow read blocked {statistics.median(slow_block) * 1000:.1f} ms "
              "(median)")

        # THE number the motion prefilter actually lives on: how far apart two
        # consecutive reads are *in the video*, not in wall clock.
        if stream_gaps:
            gaps = sorted(stream_gaps)
            med_gap = statistics.median(gaps)
            print(f"\n         stream-time between consecutive reads: "
                  f"median {med_gap:.0f} ms  (min {gaps[0]:.0f} / max {gaps[-1]:.0f})")
            print(f"         you read every {interval * 1000:.0f} ms of wall clock, so the "
                  f"prefilter sees {interval * 1000 / max(1e-6, med_gap):.1f}x less "
                  "movement per diff\n         than a test that samples a clip at "
                  "scan_fps.")
            print("         (timestamps ARE usable on this camera — a fix can select a "
                  "reference\n          frame by stream age, which is robust to jitter "
                  "and dropped frames.)")
        else:
            print("\n         stream-time between reads: UNAVAILABLE — this camera "
                  "doesn't report\n         CAP_PROP_POS_MSEC, so a fix has to count "
                  "frames and assume a steady rate.")

        # --- phase 2: the tell. Queued frames come back instantly; a live stream
        # makes each read wait for the next frame.
        fast = []
        last_frame = None
        for _ in range(args.drain):
            ok, blocked, frame, _pos = _read(cap, cv2)
            if not ok:
                break
            fast.append(blocked)
            last_frame = frame
        if not fast:
            sys.exit("no frames on the drain reads — stream dropped?")

        med = statistics.median(fast)
        live_gap = 1.0 / max(1e-6, native)
        instant = sum(1 for b in fast if b < live_gap * 0.5)
        print(f"\nphase 2: {len(fast)} fast reads, median block {med * 1000:.1f} ms "
              f"(a live frame would take ~{live_gap * 1000:.0f} ms)")
        print(f"         {instant}/{len(fast)} came back in under half that — "
              "i.e. they were already waiting\n")

        backlogged = instant >= max(3, len(fast) // 4)
        if backlogged:
            print("VERDICT: frames QUEUE.")
            print("  The app's reads return progressively older frames, so the motion")
            print("  prefilter diffs frames ~1/camera_fps apart, NOT 1/scan_fps.")
            print("  An offline harness that samples a clip at scan_fps is testing an")
            print("  EASIER problem than the live path — expect it to over-report motion.")
            print(f"  Roughly {native / max(1e-6, scan_fps):.1f}x less movement per diff live.")
        else:
            print("VERDICT: frames are DROPPED (reads are near-current).")
            print("  Consecutive reads are ~1/scan_fps apart, so sampling a clip at")
            print("  scan_fps IS a fair proxy for the live path, and the missed")
            print("  detection is not explained by frame spacing. Look elsewhere.")

        if args.save:
            os.makedirs(args.save, exist_ok=True)
            for name, frame in (("first", first_frame), ("last", last_frame)):
                if frame is not None:
                    path = os.path.join(args.save, f"probe_{name}.jpg")
                    cv2.imwrite(path, frame)
                    print(f"  wrote {path}")
            print("  (if frames queue, 'last' will visibly lag real time)")
    finally:
        cap.release()


if __name__ == "__main__":
    main()
