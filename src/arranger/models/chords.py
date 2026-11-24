"""Large-vocabulary chord recognition with BTC (Park et al., ISMIR 2019), vendored under vendor/btc."""

from pathlib import Path

import librosa
import numpy as np
import torch

from arranger.config import Config
from arranger.vendor.btc.btc_model import BTC_model

WEIGHTS = Path(__file__).resolve().parent.parent / "vendor" / "btc" / "btc_model_large_voca.pt"

ROOTS = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
QUALITIES = ["min", "maj", "dim", "aug", "min6", "maj6", "min7", "minmaj7", "maj7", "7", "dim7", "hdim7", "sus2", "sus4"]

# BTC run_config.yaml, large vocabulary
MODEL_CFG = dict(
    feature_size=144, timestep=108, num_chords=170, input_dropout=0.2, layer_dropout=0.2,
    attention_dropout=0.2, relu_dropout=0.2, num_layers=8, num_heads=4, hidden_size=128,
    total_key_depth=128, total_value_depth=128, filter_size=128, loss="ce", probs_out=False,
)
INST_LEN = 10.0
N_BINS, BINS_PER_OCTAVE, HOP = 144, 24, 2048


def idx_to_label(i: int) -> str:
    """Harte-style label: 'A:min7', 'C' (major), 'N' (no chord), 'X' (unknown)."""
    if i == 169:
        return "N"
    if i == 168:
        return "X"
    root, q = ROOTS[i // 14], QUALITIES[i % 14]
    return root if q == "maj" else f"{root}:{q}"


def _features(y: np.ndarray, sr: int) -> np.ndarray:
    """Log-CQT in 10 s chunks, exactly as BTC's training pipeline computed it."""
    step = int(sr * INST_LEN)
    parts = [
        librosa.cqt(y[i : i + step], sr=sr, n_bins=N_BINS, bins_per_octave=BINS_PER_OCTAVE, hop_length=HOP)
        for i in range(0, len(y), step)
        if len(y[i : i + step]) > HOP
    ]
    return np.log(np.abs(np.concatenate(parts, axis=1)) + 1e-6)


def run(y: np.ndarray, sr: int, cfg: Config) -> dict:
    """y: mono float32 at cfg.sr_chords. Returns {"chords": [[start, end, label], ...]}."""
    torch.set_num_threads(cfg.cpu_threads)
    ckpt = torch.load(WEIGHTS, map_location="cpu", weights_only=False)
    model = BTC_model(config=MODEL_CFG)
    model.load_state_dict(ckpt["model"])
    model.eval()

    feat = (_features(y, sr).T - ckpt["mean"]) / ckpt["std"]  # (frames, 144)
    n_frames = feat.shape[0]
    T = MODEL_CFG["timestep"]
    feat = np.pad(feat, ((0, (-n_frames) % T), (0, 0)))
    x = torch.tensor(feat, dtype=torch.float32).view(-1, T, N_BINS)
    with torch.no_grad():
        hidden, _ = model.self_attn_layers(x)
        pred, _ = model.output_layer(hidden)
    pred = pred.reshape(-1)[:n_frames].numpy()

    frame_dur = INST_LEN / T
    chords, start = [], 0
    for i in range(1, n_frames + 1):
        if i == n_frames or pred[i] != pred[start]:
            chords.append([round(start * frame_dur, 3), round(i * frame_dur, 3), idx_to_label(int(pred[start]))])
            start = i
    return {"chords": chords}
