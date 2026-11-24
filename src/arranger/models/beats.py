"""Beat and downbeat tracking with beat_this (CPJKU)."""

import numpy as np
import torch
from beat_this.inference import Audio2Beats

from arranger.audio import to_mono
from arranger.config import Config


def run(mix: np.ndarray, sr: int, cfg: Config) -> dict:
    torch.set_num_threads(cfg.cpu_threads)
    tracker = Audio2Beats(checkpoint_path=cfg.beat_checkpoint, device="cpu", dbn=False)
    beats, downbeats = tracker(to_mono(mix), sr)
    return {"beats": [float(b) for b in beats], "downbeats": [float(d) for d in downbeats]}
