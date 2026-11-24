"""Audio decoding (PyAV bundles ffmpeg, so no system ffmpeg is needed) and resampling."""

from pathlib import Path

import av
import numpy as np
import soundfile as sf
import soxr


def decode(path: str | Path, sr: int, mono: bool = False) -> np.ndarray:
    """Decode any ffmpeg-readable file to float32, shape (channels, samples)."""
    layout = "mono" if mono else "stereo"
    resampler = av.AudioResampler(format="fltp", layout=layout, rate=sr)
    chunks = []
    with av.open(str(path)) as container:
        stream = container.streams.audio[0]
        for frame in container.decode(stream):
            for out in resampler.resample(frame):
                chunks.append(out.to_ndarray())
        for out in resampler.resample(None):
            chunks.append(out.to_ndarray())
    if not chunks:
        raise ValueError(f"no audio decoded from {path}")
    return np.concatenate(chunks, axis=1).astype(np.float32)


def resample(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    """Resample (channels, samples) or (samples,) float audio."""
    if sr_in == sr_out:
        return x
    if x.ndim == 1:
        return soxr.resample(x, sr_in, sr_out).astype(np.float32)
    return soxr.resample(x.T, sr_in, sr_out).T.astype(np.float32)


def to_mono(x: np.ndarray) -> np.ndarray:
    return x.mean(axis=0) if x.ndim == 2 else x


def write_flac(path: Path, x: np.ndarray, sr: int) -> None:
    sf.write(str(path), np.clip(x, -1, 1).T, sr, format="FLAC", subtype="PCM_16")


def read_audio(path: Path) -> tuple[np.ndarray, int]:
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    return data.T, sr
