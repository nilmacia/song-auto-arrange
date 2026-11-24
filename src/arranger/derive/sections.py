"""Bar-level structure.

1. Two bar-by-bar self-similarity matrices: harmony (audio chroma + chord chroma) and arrangement
   (which stems are playing + loudness).
2. Boundaries = peaks of their combined checkerboard novelty; overlong segments are split at the
   strongest phrase-aligned (every 4 bars) point.
3. Segments are clustered by repetition into material letters (A, B, ...).
4. Functional labels (intro/verse/pre-chorus/chorus/bridge/inst/outro) come from heuristics on
   repetition, vocals, loudness and position. They are a draft for humans to correct.
"""

import string
from dataclasses import dataclass

import librosa
import numpy as np
from scipy.ndimage import binary_closing, uniform_filter1d
from scipy.signal import find_peaks

from arranger.config import Config
from arranger.derive.bars import Bar
from arranger.derive.harmony import chord_chroma

SUBDIV = 4  # sub-segments per bar, keeps the within-bar harmonic rhythm
STEMS = ("vocals", "drums", "bass", "other")


@dataclass
class Segment:
    start: int  # first bar index
    end: int  # one past last bar index
    letter: str = ""
    label: str = ""
    occurrence: int = 0

    @property
    def length(self) -> int:
        return self.end - self.start


# ---------------------------------------------------------------- features

def harmony_features(harmonic: np.ndarray, sr: int, bars: list[Bar], beat_chords: list[list[str]]) -> np.ndarray:
    hop = 512
    chroma = librosa.feature.chroma_cqt(y=harmonic, sr=sr, hop_length=hop)
    times = librosa.frames_to_time(np.arange(chroma.shape[1]), sr=sr, hop_length=hop)
    feats = []
    for b, chords in zip(bars, beat_chords):
        edges = np.linspace(b.start, b.end, SUBDIV + 1)
        beat_starts = [s for s, _ in b.beat_spans()]
        parts = []
        for s, e in zip(edges[:-1], edges[1:]):
            m = (times >= s) & (times < e)
            parts.append(chroma[:, m].mean(1) if m.any() else np.zeros(12))
        for t in (edges[:-1] + edges[1:]) / 2:
            k = min(int(np.searchsorted(beat_starts, t, "right")) - 1, len(chords) - 1)
            parts.append(chord_chroma(chords[max(k, 0)]))
        feats.append(np.concatenate(parts))
    x = np.array(feats)
    x = (x - x.mean(0)) / (x.std(0) + 1e-8)
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)


def arrangement_features(activity: dict[str, np.ndarray], vocal: np.ndarray, loud_db: np.ndarray) -> np.ndarray:
    cols = [activity[k] for k in STEMS if k in activity and k != "vocals"] + [vocal]
    cols.append((loud_db - loud_db.mean()) / (loud_db.std() + 1e-8) * 0.3)
    return uniform_filter1d(np.stack(cols, 1), 2, axis=0)


def harmony_ssm(h: np.ndarray) -> np.ndarray:
    """Cosine SSM smoothed along diagonals so repeated sequences beat single similar bars."""
    s = h @ h.T
    n = len(s)
    out, cnt = np.zeros_like(s), np.zeros_like(s)
    for k in (-1, 0, 1, 2):
        dst = slice(max(0, -k), n - max(0, k))
        src = slice(max(0, k), n - max(0, -k))
        out[dst, dst] += s[src, src]
        cnt[dst, dst] += 1
    return out / cnt


def arrangement_ssm(a: np.ndarray) -> np.ndarray:
    return np.exp(-((a[:, None, :] - a[None, :, :]) ** 2).sum(-1) / 0.25)


def novelty(s: np.ndarray, half: int) -> np.ndarray:
    g = np.exp(-0.5 * np.linspace(-1.5, 1.5, 2 * half) ** 2)
    sign = np.r_[-np.ones(half), np.ones(half)]
    kernel = np.outer(g, g) * np.outer(sign, sign)
    pad = np.pad(s, half, mode="edge")
    nov = np.array([np.sum(kernel * pad[i : i + 2 * half, i : i + 2 * half]) for i in range(len(s))])
    nov = np.maximum(nov, 0)
    return nov / (np.percentile(nov, 95) + 1e-8)


def vocal_bars(vocal_act: np.ndarray, words_per_bar: np.ndarray, thr: float) -> np.ndarray:
    """Bars with singing: vocal stem active (or weaker activity + recognised words), gaps <= 2 bars closed."""
    v = (vocal_act >= thr) | ((vocal_act >= thr / 2) & (words_per_bar > 0))
    v = binary_closing(np.r_[False, v, False], structure=np.ones(3))[1:-1]
    # drop isolated 1-bar blips
    v = v & (np.r_[v[1:], False] | np.r_[False, v[:-1]])
    return v.astype(float)


# ---------------------------------------------------------------- segmentation

def boundaries(nov: np.ndarray, cfg: Config) -> list[int]:
    n = len(nov)
    peaks, _ = find_peaks(nov, distance=cfg.min_section_bars, prominence=0.5)
    bounds = [0]
    for p in sorted(int(p) for p in peaks):
        if p - bounds[-1] >= cfg.min_section_bars and n - p >= cfg.min_section_bars:
            bounds.append(p)
    bounds.append(n)

    # split overlong segments at the strongest phrase-aligned position
    out = [0]
    for a, b in zip(bounds[:-1], bounds[1:]):
        out.extend(_split(nov, a, b, cfg.max_section_bars))
    return out


def _split(nov: np.ndarray, a: int, b: int, max_len: int) -> list[int]:
    if b - a <= max_len:
        return [b]
    cands = [a + k for k in range(4, b - a - 3, 4)] or [a + (b - a) // 2]
    c = max(cands, key=lambda i: nov[i])
    return _split(nov, a, c, max_len) + _split(nov, c, b, max_len)


def _seg_similarity(sh: np.ndarray, arr: np.ndarray, a: Segment, b: Segment) -> float:
    L = min(a.length, b.length)
    harm = float(np.mean([sh[a.start + k, b.start + k] for k in range(L)]))
    pa, pb = arr[a.start : a.end].mean(0), arr[b.start : b.end].mean(0)
    arrange = float(np.exp(-((pa - pb) ** 2).sum() / 0.1))
    length = (L / max(a.length, b.length)) ** 0.5
    return (0.5 * harm + 0.5 * arrange) * length


def cluster(segments: list[Segment], sh: np.ndarray, arr: np.ndarray, threshold: float) -> None:
    """Greedy repetition clustering; letters in order of first appearance."""
    groups: list[list[Segment]] = []
    for seg in segments:
        sims = [max(_seg_similarity(sh, arr, seg, m) for m in g) for g in groups]
        if sims and max(sims) >= threshold:
            g = groups[int(np.argmax(sims))]
        else:
            groups.append([])
            g = groups[-1]
        g.append(seg)
        seg.occurrence = len(g)
    for i, g in enumerate(groups):
        for seg in g:
            seg.letter = string.ascii_uppercase[i] if i < 26 else f"Z{i}"


def label(segments: list[Segment], vocal: np.ndarray, words_per_bar: np.ndarray, loud_db: np.ndarray) -> None:
    def is_vocal(seg: Segment) -> bool:
        # stem activity alone is fooled by bleed; require some recognised words too
        return bool(vocal[seg.start : seg.end].mean() >= 0.5 and words_per_bar[seg.start : seg.end].sum() >= 2)

    by_letter: dict[str, list[Segment]] = {}
    for seg in segments:
        by_letter.setdefault(seg.letter, []).append(seg)
    vocal_letters = [L for L, segs in by_letter.items() if any(is_vocal(x) for x in segs)]

    def loudness(L: str) -> float:
        return float(np.mean([loud_db[x.start : x.end].mean() for x in by_letter[L]]) - loud_db.mean())

    chorus = None
    repeated = [L for L in vocal_letters if len(by_letter[L]) >= 2]
    if len(vocal_letters) >= 2 and repeated:
        chorus = max(repeated, key=lambda L: len(by_letter[L]) + loudness(L) / 3.0)

    def is_pre_chorus(L: str) -> bool:
        """Every occurrence sits between other (non-chorus) vocal material and the chorus."""
        idx = [k for k, x in enumerate(segments) if x.letter == L]
        return all(
            0 < k < len(segments) - 1
            and segments[k + 1].letter == chorus
            and segments[k - 1].letter not in (L, chorus)
            and is_vocal(segments[k - 1])
            for k in idx
        )

    # a bridge is new vocal material appearing once, after the chorus has been heard twice
    second_chorus = next((x.start for x in by_letter.get(chorus, [])[1:2]), None) if chorus else None
    last_vocal = max((k for k, x in enumerate(segments) if is_vocal(x)), default=-1)

    for i, seg in enumerate(segments):
        if not is_vocal(seg):
            seg.label = "intro" if i == 0 else "outro" if i > last_vocal else "inst"
        elif seg.letter == chorus:
            seg.label = "chorus"
        elif chorus and seg.length <= 8 and is_pre_chorus(seg.letter):
            seg.label = "pre-chorus"
        elif second_chorus is not None and seg.start > second_chorus and len(by_letter[seg.letter]) == 1:
            seg.label = "bridge"
        else:
            seg.label = "verse"


# ---------------------------------------------------------------- entry point

def analyse(
    harmonic: np.ndarray, sr: int, bars: list[Bar], beat_chords: list[list[str]],
    activity: dict[str, np.ndarray], words_per_bar: np.ndarray, loud_db: np.ndarray, cfg: Config,
) -> list[Segment]:
    n = len(bars)
    if n < 2 * cfg.min_section_bars:
        seg = Segment(0, n, letter="A", label="verse", occurrence=1)
        return [seg]
    vocal = vocal_bars(activity.get("vocals", np.zeros(n)), words_per_bar, cfg.vocal_activity_threshold)
    sh = harmony_ssm(harmony_features(harmonic, sr, bars, beat_chords))
    arr = arrangement_features(activity, vocal, loud_db)
    h = cfg.novelty_half_kernel_bars
    nov = novelty(sh, h) + novelty(arrangement_ssm(arr), h)
    segs = [Segment(a, b) for a, b in zip(*(lambda bs: (bs[:-1], bs[1:]))(boundaries(nov, cfg)))]
    cluster(segs, sh, arr, cfg.cluster_threshold)
    label(segs, vocal, words_per_bar, loud_db)
    return segs
