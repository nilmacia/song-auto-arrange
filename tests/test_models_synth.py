"""Runs the real beat and chord models on a synthesized 120 bpm I-V-vi-IV loop in C (~45 s on CPU).
Skip with: pytest -m 'not slow'"""

import numpy as np
import pytest

from arranger.audio import resample
from arranger.config import DEFAULT
from arranger.derive.bars import build_bars, label_spans
from arranger.derive.harmony import chord_chroma, estimate_keys, key_index_name

SR = 44100
BPM = 120
PROG = [("C", [48, 60, 64, 67]), ("G", [43, 59, 62, 67]), ("A:min", [45, 60, 64, 69]), ("F", [41, 60, 65, 69])]


def tone(midi, dur, sr=SR):
    t = np.arange(int(dur * sr)) / sr
    f = 440 * 2 ** ((midi - 69) / 12)
    env = np.minimum(1, t / 0.01) * np.exp(-t * 1.5)
    return env * sum(np.sin(2 * np.pi * f * h * t) / h for h in range(1, 6))


def drum(dur, accent, sr=SR):
    t = np.arange(int(dur * sr)) / sr
    kick = np.sin(2 * np.pi * (50 + 100 * np.exp(-t * 40)) * t) * np.exp(-t * 18)
    return kick * (1.0 if accent else 0.5)


def synth(n_bars=24):
    beat = 60 / BPM
    y = np.zeros(int((n_bars * 4 + 2) * beat * SR))
    for b in range(n_bars):
        name, notes = PROG[b % 4]
        start = int(b * 4 * beat * SR)
        chord = sum(tone(m, 4 * beat) for m in notes)
        y[start : start + len(chord)] += 0.15 * chord
        for k in range(4):
            d = drum(beat, k == 0)
            s = start + int(k * beat * SR)
            y[s : s + len(d)] += 0.6 * d
    return (y / np.abs(y).max() * 0.8).astype(np.float32)


@pytest.mark.slow
def test_beats_chords_key_on_synth():
    from arranger.models import beats, chords

    y = synth()
    bt = beats.run(np.stack([y, y]), SR, DEFAULT)
    bars = build_bars(bt["beats"], bt["downbeats"], duration=len(y) / SR)
    full = [b for b in bars if not b.pickup][:20]
    assert np.median([b.bpm for b in full]) == pytest.approx(BPM, abs=2)
    assert sum(b.time_signature == "4/4" for b in full) >= 18

    ch = chords.run(resample(y, SR, DEFAULT.sr_chords), DEFAULT.sr_chords, DEFAULT)["chords"]
    beat_chords = [label_spans(b.beat_spans(), ch) for b in full]
    # the bar grid may be offset from the synth's bar 0; compare against the expected chord at bar start time
    hits = 0
    for b, bc in zip(full, beat_chords):
        expected = PROG[int(round(b.start / (4 * 60 / BPM))) % 4][0]
        hits += max(set(bc), key=bc.count) == expected
    assert hits >= 0.75 * len(full)

    g, _ = estimate_keys(np.stack([sum(chord_chroma(c) for c in bc) for bc in beat_chords]), 8, 1.0)
    assert key_index_name(g) == "C major"
