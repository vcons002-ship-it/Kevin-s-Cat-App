"""Generate the default celebratory treat chime as a WAV file.

Uses only the Python standard library (``wave``, ``math``, ``struct``) so the
default sound is self-contained and free of any licensing concerns. Run via::

    python d20app/sounds/generate_chime.py

It writes ``treat_chime.wav`` next to this script: a short ascending major
arpeggio (C-E-G-C) with a soft attack/decay envelope — a satisfying "you got
it!" fanfare.
"""

from __future__ import annotations

import math
import os
import struct
import wave

SAMPLE_RATE = 44100
AMPLITUDE = 0.45            # 0..1, headroom to avoid clipping when notes overlap
OUTPUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "treat_chime.wav")

# Ascending C-major arpeggio, ending on the octave. (note_hz, start_s, dur_s)
NOTES = [
    (523.25, 0.00, 0.18),   # C5
    (659.25, 0.12, 0.18),   # E5
    (783.99, 0.24, 0.18),   # G5
    (1046.50, 0.36, 0.55),  # C6 (held, the "ta-da")
]


def _envelope(t: float, dur: float) -> float:
    """Smooth attack/decay so notes don't click. Returns gain in 0..1."""
    attack = 0.015
    release = 0.12
    if t < attack:
        return t / attack
    if t > dur - release:
        return max(0.0, (dur - t) / release)
    return 1.0


def synthesize() -> bytearray:
    total_s = max(start + dur for _, start, dur in NOTES) + 0.05
    n_samples = int(total_s * SAMPLE_RATE)
    samples = [0.0] * n_samples

    for freq, start, dur in NOTES:
        start_i = int(start * SAMPLE_RATE)
        for i in range(int(dur * SAMPLE_RATE)):
            idx = start_i + i
            if idx >= n_samples:
                break
            t = i / SAMPLE_RATE
            # Fundamental plus a soft 2nd harmonic for a brighter, bell-like tone.
            value = math.sin(2 * math.pi * freq * t)
            value += 0.25 * math.sin(2 * math.pi * 2 * freq * t)
            samples[idx] += value * _envelope(t, dur)

    # Normalise then convert to signed 16-bit PCM.
    peak = max((abs(s) for s in samples), default=1.0) or 1.0
    frames = bytearray()
    for s in samples:
        # Normalise to the peak, then scale to AMPLITUDE for headroom.
        scaled = max(-1.0, min(1.0, (s / peak) * AMPLITUDE))
        frames += struct.pack("<h", int(scaled * 32767))
    return frames


def main(output: str = OUTPUT) -> str:
    frames = synthesize()
    with wave.open(output, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)        # 16-bit
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(bytes(frames))
    print(f"Wrote {output} ({len(frames) // 2} samples, {SAMPLE_RATE} Hz mono)")
    return output


if __name__ == "__main__":
    main()
