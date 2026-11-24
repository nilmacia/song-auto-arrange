"""Per-bar loudness → dynamic marks, and per-stem activity."""

import numpy as np

from arranger.config import Config
from arranger.derive.bars import Bar


def bar_db(y: np.ndarray, sr: int, bars: list[Bar]) -> np.ndarray:
    out = []
    for b in bars:
        seg = y[int(b.start * sr) : max(int(b.end * sr), int(b.start * sr) + 1)]
        out.append(10 * np.log10(np.mean(seg**2) + 1e-10))
    return np.array(out)


def marks(db: np.ndarray, cfg: Config) -> tuple[list[str], float]:
    """Dynamic mark per bar relative to the song's median bar; also returns the loud reference (95th pct)."""
    med = float(np.median(db))
    out = [next((m for thr, m in cfg.dyn_thresholds if d >= thr), "pp") for d in db - med]
    return out, float(np.percentile(db, 95))


def stem_activity(stem_db: dict[str, np.ndarray], mix_ref_db: float) -> dict[str, np.ndarray]:
    """0..1 per stem and bar. A stem 30 dB below the mix's loud level counts as silent."""
    floor = mix_ref_db - 30.0
    out = {}
    for name, db in stem_db.items():
        top = max(float(np.percentile(db, 95)), mix_ref_db - 20.0)
        out[name] = np.clip((db - floor) / (top - floor), 0.0, 1.0)
    return out
