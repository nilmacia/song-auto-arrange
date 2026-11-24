"""Bar grid from beats/downbeats, plus helpers to map timed events onto bars and beats."""

from collections import Counter
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Bar:
    beats: list[float]  # beat onset times; beats[0] is the bar start
    end: float
    pickup: bool = False
    extra: dict = field(default_factory=dict)

    @property
    def start(self) -> float:
        return self.beats[0]

    @property
    def n_beats(self) -> int:
        return len(self.beats)

    def beat_spans(self) -> list[tuple[float, float]]:
        edges = self.beats + [self.end]
        return list(zip(edges[:-1], edges[1:]))

    @property
    def bpm(self) -> float:
        edges = np.array(self.beats + [self.end])
        return float(60.0 / np.median(np.diff(edges)))

    @property
    def time_signature(self) -> str:
        return f"{self.n_beats}/4"


def build_bars(beats: list[float], downbeats: list[float], duration: float) -> list[Bar]:
    beats_arr = _fill_gaps(np.asarray(beats, dtype=float))
    if len(beats_arr) < 2:
        return [Bar(beats=[0.0], end=duration)]
    ibi = float(np.median(np.diff(beats_arr)))

    # downbeats → indices into beats (nearest beat), dropping duplicates
    if len(downbeats):
        idx = np.abs(beats_arr[None, :] - np.asarray(downbeats)[:, None]).argmin(axis=1)
        db_idx = sorted(set(int(i) for i in idx))
    else:
        db_idx = list(range(0, len(beats_arr), 4))

    groups: list[list[float]] = []
    pickup = db_idx[0] > 0
    if pickup:
        groups.append(list(beats_arr[: db_idx[0]]))
    for a, b in zip(db_idx, db_idx[1:] + [len(beats_arr)]):
        groups.append(list(beats_arr[a:b]))

    body = groups[1:] if pickup else groups
    body = _regularize(body)
    groups = ([groups[0]] if pickup else []) + body

    bars = []
    for i, g in enumerate(groups):
        end = groups[i + 1][0] if i + 1 < len(groups) else min(g[-1] + ibi, duration)
        bars.append(Bar(beats=[float(t) for t in g], end=float(end), pickup=pickup and i == 0))
    return bars


def _fill_gaps(beats: np.ndarray) -> np.ndarray:
    """Make the beat list roughly evenly spaced at the song's main beat period:
    drop double-time beats and interpolate beats where the tracker dropped out (breaks, rubato)."""
    if len(beats) < 3:
        return beats
    period = float(np.median(np.diff(beats)))
    kept = [beats[0]]
    for b in beats[1:]:
        if b - kept[-1] >= 0.7 * period:
            kept.append(b)
    out = [kept[0]]
    for a, b in zip(kept[:-1], kept[1:]):
        k = int(round((b - a) / period))
        if b - a > 1.6 * period and k >= 2:
            out.extend(a + (b - a) * np.arange(1, k) / k)
        out.append(b)
    return np.array(out)


def _regularize(groups: list[list[float]]) -> list[list[float]]:
    """Fix likely downbeat errors against the song's dominant meter m:
    - a bar of k*m beats, or any bar of >= 2m beats (missed downbeats), is split into m-beat bars;
    - consecutive short bars that add up to exactly m (spurious downbeat) are merged."""
    if not groups:
        return groups
    # the song's dominant meter; a local estimate collapses where the tracker marks every beat as a downbeat
    m = Counter(len(g) for g in groups).most_common(1)[0][0]
    if m < 2:
        m = 4
    out: list[list[float]] = []
    i = 0
    while i < len(groups):
        g = groups[i]
        n = len(g)
        if n > m and (n % m == 0 or n >= 2 * m):
            out.extend(g[j : j + m] for j in range(0, n, m))  # any remainder becomes a short bar
            i += 1
            continue
        if n < m:
            merged, j = list(g), i + 1
            while j < len(groups) and len(merged) + len(groups[j]) <= m:
                merged += groups[j]
                j += 1
                if len(merged) == m:
                    break
            if len(merged) == m:
                out.append(merged)
                i = j
                continue
        out.append(g)
        i += 1
    return out


def bar_index(times: np.ndarray | list[float], bars: list[Bar]) -> np.ndarray:
    """Index of the bar containing each time (times before the first bar map to bar 0)."""
    starts = np.array([b.start for b in bars])
    return np.clip(np.searchsorted(starts, np.asarray(times), side="right") - 1, 0, len(bars) - 1)


def label_spans(spans: list[tuple[float, float]], intervals: list[list]) -> list[str]:
    """For each (start, end) span, the label of the [start, end, label] interval overlapping it most."""
    iv = np.array([[s, e] for s, e, _ in intervals], dtype=float)
    labels = [lab for _, _, lab in intervals]
    out = []
    for s, e in spans:
        overlap = np.minimum(iv[:, 1], e) - np.maximum(iv[:, 0], s)
        k = int(overlap.argmax())
        out.append(labels[k] if overlap[k] > 0 else "N")
    return out
