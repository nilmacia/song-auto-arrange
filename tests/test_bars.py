import numpy as np

from arranger.derive.bars import bar_index, build_bars, label_spans


def grid(bpm, start, n_beats):
    return [start + i * 60 / bpm for i in range(n_beats)]


def test_meter_change_4_4_to_3_4():
    beats = grid(120, 0.0, 16)  # 4 bars of 4/4
    beats += grid(120, beats[-1] + 0.5, 12)  # 4 bars of 3/4
    downbeats = beats[0:16:4] + beats[16::3]
    bars = build_bars(beats, downbeats, duration=beats[-1] + 0.5)
    assert [b.time_signature for b in bars] == ["4/4"] * 4 + ["3/4"] * 4
    assert all(round(b.bpm) == 120 for b in bars)


def test_pickup_bar():
    beats = grid(100, 0.0, 18)
    bars = build_bars(beats, beats[2::4], duration=beats[-1] + 0.6)
    assert bars[0].pickup and bars[0].n_beats == 2
    assert all(not b.pickup and b.n_beats == 4 for b in bars[1:])


def test_missed_downbeat_is_split_and_spurious_one_merged():
    beats = grid(120, 0.0, 32)
    downbeats = [beats[i] for i in (0, 4, 12, 16, 18, 20, 24, 28)]  # 8 missing, 18 spurious
    bars = build_bars(beats, downbeats, duration=16.5)
    assert [b.n_beats for b in bars] == [4] * 8


def test_gap_and_double_time_are_regularised():
    beats = grid(120, 0.0, 8) + grid(120, 8.0, 8)  # 4 s dropout: 4.0 -> 8.0
    beats += list(np.arange(12.0, 14.0, 0.25))  # double-time section
    downbeats = [b for b in beats if abs(b % 2.0) < 1e-6]
    bars = build_bars(beats, downbeats, duration=14.0)
    assert all(b.n_beats == 4 for b in bars)
    assert all(round(b.bpm) == 120 for b in bars)


def test_bar_index_and_label_spans():
    bars = build_bars(grid(120, 1.0, 12), grid(120, 1.0, 12)[::4], duration=7.0)
    assert list(bar_index([0.2, 1.0, 2.99, 3.0, 6.9], bars)) == [0, 0, 0, 1, 2]
    chords = [[0.0, 1.6, "C"], [1.6, 3.0, "A:min"]]  # change just after beat 2 of bar 1
    assert label_spans(bars[0].beat_spans(), chords) == ["C", "A:min", "A:min", "A:min"]
