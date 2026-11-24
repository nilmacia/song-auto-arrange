import numpy as np

from arranger.derive.harmony import chord_chroma, estimate_keys, key_index_name, spell


def bars_of(chords, reps):
    return np.stack([chord_chroma(c) for c in chords] * reps)


def test_spelling():
    assert spell("A:min7") == "Am7"
    assert spell("C") == "C"
    assert spell("A#:hdim7") == "A#m7b5"
    assert spell("A#", key=5) == "Bb"  # F major uses flats
    assert spell("N") == "N.C."


def test_global_key_pop_progression():
    g, keys = estimate_keys(bars_of(["C", "G", "A:min", "F"], 4), window=8, switch_penalty=1.0)
    assert key_index_name(g) == "C major"
    assert {key_index_name(k) for k in keys} == {"C major"}


def test_modulation_detected_but_short_flip_ignored():
    c = bars_of(["C", "G", "A:min", "F"], 6)
    d = bars_of(["D", "A", "B:min", "G"], 6)  # up a whole step for 24 bars
    _, keys = estimate_keys(np.vstack([c, d]), window=8, switch_penalty=1.0)
    names = [key_index_name(k) for k in keys]
    assert names[:20] == ["C major"] * 20
    assert names[-20:] == ["D major"] * 20

    blip = np.vstack([c, bars_of(["E", "B"], 1), c])  # 2 bars of outside chords
    _, keys = estimate_keys(blip, window=8, switch_penalty=1.0)
    assert {key_index_name(k) for k in keys} == {"C major"}
