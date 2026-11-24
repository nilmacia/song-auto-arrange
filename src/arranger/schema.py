"""Output format helpers.

The arrangement is a list of bars. Every bar carries its own content (chords per beat, lyrics,
dynamics, which stems play). Values that normally hold for many bars are written only on the bar
where they start or change, the way a score shows them:

    section          {"id": "B1", "type": "chorus"} on the first bar of each section
    bpm              tempo from this bar on
    time_signature   meter from this bar on
    key              key from this bar on

The first bar always carries all four. `resolve()` fills them forward for code that wants every
value on every bar.
"""

import numpy as np

MARKS = ("section", "bpm", "time_signature", "key")


def compact(bars: list[dict]) -> list[dict]:
    """Drop mark values that are equal to the previous bar's (input bars carry every mark)."""
    out, prev = [], {}
    for b in bars:
        b = dict(b)
        for m in MARKS:
            if m in b and prev.get(m) == b[m]:
                del b[m]
            elif m in b:
                prev[m] = b[m]
        out.append(b)
    return out


def resolve(arrangement: dict) -> list[dict]:
    """Bars with every mark filled forward; section gets "section_id" and "section_type"."""
    out, current = [], {}
    for b in arrangement["bars"]:
        current.update({m: b[m] for m in MARKS if m in b})
        r = {**b, **current}
        sec = r.pop("section", None) or {}
        r["section_id"], r["section_type"] = sec.get("id"), sec.get("type")
        out.append(r)
    return out


def tempo_track(bpm: np.ndarray, song_bpm: float, tolerance: float) -> list[int]:
    """Per-bar tempo that only changes for a sustained (>= 2 bars) deviation larger than `tolerance`.
    Bars within tolerance of the song tempo report the song tempo."""
    def near(a: float, b: float) -> bool:
        return abs(a / b - 1) <= tolerance

    out = []
    current = song_bpm if near(bpm[0], song_bpm) else bpm[0]
    for i, v in enumerate(bpm):
        nxt = bpm[i + 1] if i + 1 < len(bpm) else v
        if not near(v, current) and not near(nxt, current):
            current = song_bpm if near(v, song_bpm) else v
        out.append(int(round(current)))
    return out


def lyric_text(words: list[list], line_gap: float) -> list[str]:
    """Word texts with "/ " marking the start of a lyric line (a pause of >= line_gap seconds)."""
    out = []
    for i, (start, _, text, _) in enumerate(words):
        new_line = i == 0 or start - words[i - 1][1] >= line_gap
        out.append(f"/ {text}" if new_line else text)
    return out
