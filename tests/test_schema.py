import numpy as np

from arranger.schema import compact, lyric_text, resolve, tempo_track


def test_marks_only_where_they_change_and_resolve_round_trip():
    full = [
        {"bar": 1, "section": {"id": "A1", "type": "intro"}, "bpm": 120, "time_signature": "4/4", "key": "C major", "chords": ["C"] * 4},
        {"bar": 2, "section": {"id": "A1", "type": "intro"}, "bpm": 120, "time_signature": "4/4", "key": "C major", "chords": ["G"] * 4},
        {"bar": 3, "section": {"id": "B1", "type": "verse"}, "bpm": 120, "time_signature": "2/4", "key": "C major", "chords": ["F"] * 2},
        {"bar": 4, "section": {"id": "B1", "type": "verse"}, "bpm": 126, "time_signature": "4/4", "key": "D major", "chords": ["D"] * 4},
    ]
    bars = compact(full)
    assert set(bars[0]) == set(full[0])  # first bar carries everything
    assert set(bars[1]) == {"bar", "chords"}
    assert bars[2]["section"]["id"] == "B1" and bars[2]["time_signature"] == "2/4" and "bpm" not in bars[2]
    assert {k for k in bars[3] if k != "chords"} == {"bar", "bpm", "time_signature", "key"}

    resolved = resolve({"bars": bars})
    for r, f in zip(resolved, full):
        assert (r["bpm"], r["time_signature"], r["key"]) == (f["bpm"], f["time_signature"], f["key"])
        assert (r["section_id"], r["section_type"]) == (f["section"]["id"], f["section"]["type"])


def test_tempo_track_ignores_drift_and_keeps_real_changes():
    bpm = np.array([121, 124, 119, 135, 123, 122, 140, 141, 140, 139, 124, 122], dtype=float)
    assert tempo_track(bpm, 122, 0.04) == [122] * 6 + [140] * 4 + [122] * 2


def test_lyric_lines():
    words = [[0.0, 0.4, "hold", 1], [0.5, 0.9, "on", 1], [2.5, 2.9, "let", 1], [3.0, 3.3, "go", 1]]
    assert lyric_text(words, 1.0) == ["/ hold", "on", "/ let", "go"]
