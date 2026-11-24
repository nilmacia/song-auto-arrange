"""Chord label parsing/spelling and chord-based key estimation (global + per-bar with smoothing)."""

import numpy as np

SHARPS = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
FLATS = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]

QUALITY_INTERVALS = {
    "maj": (0, 4, 7), "min": (0, 3, 7), "dim": (0, 3, 6), "aug": (0, 4, 8),
    "min6": (0, 3, 7, 9), "maj6": (0, 4, 7, 9), "min7": (0, 3, 7, 10), "minmaj7": (0, 3, 7, 11),
    "maj7": (0, 4, 7, 11), "7": (0, 4, 7, 10), "dim7": (0, 3, 6, 9), "hdim7": (0, 3, 6, 10),
    "sus2": (0, 2, 7), "sus4": (0, 5, 7),
}
QUALITY_SUFFIX = {
    "maj": "", "min": "m", "dim": "dim", "aug": "aug", "min6": "m6", "maj6": "6", "min7": "m7",
    "minmaj7": "m(maj7)", "maj7": "maj7", "7": "7", "dim7": "dim7", "hdim7": "m7b5", "sus2": "sus2", "sus4": "sus4",
}

# Krumhansl–Kessler key profiles
MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
KEY_NAMES_MAJOR = ["C", "Db", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]
KEY_NAMES_MINOR = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "G#", "A", "Bb", "B"]


def parse(label: str) -> tuple[int, str] | None:
    """'A:min7' -> (9, 'min7'); 'C' -> (0, 'maj'); 'N'/'X' -> None."""
    if label in ("N", "X"):
        return None
    root, _, quality = label.partition(":")
    return SHARPS.index(root), quality or "maj"


def chord_chroma(label: str) -> np.ndarray:
    v = np.zeros(12)
    p = parse(label)
    if p:
        root, q = p
        for iv in QUALITY_INTERVALS[q]:
            v[(root + iv) % 12] += 1.0
        v[root] += 0.5  # the root is the strongest tonal cue
    return v


def key_index_name(k: int) -> str:
    """k in 0..23: 0-11 major, 12-23 minor."""
    return f"{KEY_NAMES_MAJOR[k]} major" if k < 12 else f"{KEY_NAMES_MINOR[k - 12]} minor"


def uses_flats(k: int) -> bool:
    if k < 12:
        return "b" in KEY_NAMES_MAJOR[k] or k == 5  # F major
    return "b" in KEY_NAMES_MINOR[k - 12] or (k - 12) in (0, 2, 5, 7)  # C, D, F, G minor


def spell(label: str, key: int | None = None) -> str:
    """BTC label -> lead-sheet symbol spelled for the key: 'A#:min7' in F major -> 'Bbm7'."""
    if label == "N":
        return "N.C."
    if label == "X":
        return "?"
    root, q = parse(label)
    names = FLATS if key is not None and uses_flats(key) else SHARPS
    return names[root] + QUALITY_SUFFIX[q]


def root_chroma(label: str) -> np.ndarray:
    v = np.zeros(12)
    if p := parse(label):
        v[p[0]] = 1.0
    return v


def bar_audio_chroma(y: np.ndarray, sr: int, starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
    """Mean CQT chroma per bar, each bar normalised to sum 1 (silent bars stay zero)."""
    import librosa

    hop = 512
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop)
    times = librosa.frames_to_time(np.arange(chroma.shape[1]), sr=sr, hop_length=hop)
    out = np.zeros((len(starts), 12))
    for i, (s, e) in enumerate(zip(starts, ends)):
        m = (times >= s) & (times < e)
        if m.any():
            v = chroma[:, m].mean(1)
            out[i] = v / (v.sum() + 1e-8)
    return out


def _key_scores(chroma: np.ndarray) -> np.ndarray:
    """chroma (n, 12) -> correlation with the 24 key profiles, (n, 24)."""
    profiles = np.stack(
        [np.roll(MAJOR_PROFILE, k) for k in range(12)] + [np.roll(MINOR_PROFILE, k) for k in range(12)]
    )
    p = (profiles - profiles.mean(1, keepdims=True)) / profiles.std(1, keepdims=True)
    c = chroma - chroma.mean(1, keepdims=True)
    norm = c.std(1, keepdims=True)
    c = np.divide(c, norm, out=np.zeros_like(c), where=norm > 0)
    return c @ p.T / 12.0


def estimate_keys(bar_chroma: np.ndarray, window: int, switch_penalty: float) -> tuple[int, list[int]]:
    """Global key and per-bar keys (Viterbi over 24 keys; switching costs `switch_penalty`)."""
    global_key = int(_key_scores(bar_chroma.sum(0, keepdims=True))[0].argmax())
    n = len(bar_chroma)
    if n == 0:
        return global_key, []
    half = window // 2
    ctx = np.stack([bar_chroma[max(0, i - half) : i + half + 1].sum(0) for i in range(n)])
    scores = _key_scores(ctx)
    scores[ctx.sum(1) == 0] = 0.0  # no chords: no evidence
    scores[:, global_key] += 0.05  # tie-break towards the global key

    acc = scores[0].copy()
    back = np.zeros((n, 24), dtype=int)
    for i in range(1, n):
        best_prev = int(acc.argmax())
        stay = acc
        switch = acc[best_prev] - switch_penalty
        back[i] = np.where(stay >= switch, np.arange(24), best_prev)
        acc = np.maximum(stay, switch) + scores[i]
    path = [int(acc.argmax())]
    for i in range(n - 1, 0, -1):
        path.append(int(back[i, path[-1]]))
    return global_key, path[::-1]
