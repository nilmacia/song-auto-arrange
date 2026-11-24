import numpy as np

from arranger.derive.sections import Segment, cluster, label

PATTERN = "ABABCB"
LEN = 8


def toy():
    """Per-bar harmony/arrangement features for A B A B C B (8 bars each)."""
    rng = np.random.default_rng(0)
    proto = {k: rng.normal(size=(LEN, 16)) for k in "ABC"}
    arr_proto = {"A": [0.8, 0.7, 0.5, 1.0, 0.0], "B": [1.0, 1.0, 1.0, 1.0, 0.6], "C": [0.2, 0.6, 0.9, 1.0, -0.3]}
    h = np.vstack([proto[k] + 0.2 * rng.normal(size=(LEN, 16)) for k in PATTERN])
    h /= np.linalg.norm(h, axis=1, keepdims=True)
    arr = np.vstack([np.tile(arr_proto[k], (LEN, 1)) for k in PATTERN])
    segs = [Segment(i * LEN, (i + 1) * LEN) for i in range(len(PATTERN))]
    return h @ h.T, arr, segs


def test_cluster_letters_follow_repetition():
    sh, arr, segs = toy()
    cluster(segs, sh, arr, threshold=0.55)
    assert "".join(s.letter for s in segs) == PATTERN
    assert [s.occurrence for s in segs] == [1, 1, 2, 2, 1, 3]


def test_labels():
    sh, arr, segs = toy()
    cluster(segs, sh, arr, threshold=0.55)
    n = LEN * len(PATTERN)
    vocal = np.ones(n)
    words = np.full(n, 3)
    loud = np.array([-14.0 if s.letter == "A" else -11.0 for s in segs for _ in range(LEN)])
    label(segs, vocal, words, loud)
    assert [s.label for s in segs] == ["verse", "chorus", "verse", "chorus", "bridge", "chorus"]


def test_pre_chorus_and_instrumental_edges():
    letters = "IAPBAPBO"  # intro, verse, pre, chorus, verse, pre, chorus, outro
    segs = [Segment(i * 4, (i + 1) * 4, letter=L) for i, L in enumerate(letters)]
    for L in set(letters):
        for k, s in enumerate(x for x in segs if x.letter == L):
            s.occurrence = k + 1
    n = 4 * len(letters)
    vocal = np.array([0.0 if s.letter in "IO" else 1.0 for s in segs for _ in range(4)])
    words = (vocal * 3).astype(int)
    loud = np.array([-10.0 if s.letter == "B" else -13.0 for s in segs for _ in range(4)])
    label(segs, vocal, words, loud)
    assert [s.label for s in segs] == ["intro", "verse", "pre-chorus", "chorus", "verse", "pre-chorus", "chorus", "outro"]
