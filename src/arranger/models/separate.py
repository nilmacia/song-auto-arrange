"""Source separation with Demucs: writes mix + 4 stems as FLAC into the work dir."""

from pathlib import Path

import torch
from demucs.apply import apply_model
from demucs.pretrained import get_model

from arranger.audio import decode, write_flac
from arranger.config import Config
from arranger.models.hub import offline_first

STEMS = ("vocals", "drums", "bass", "other")


def run(song: Path, out_dir: Path, cfg: Config, progress: bool = True) -> dict:
    torch.set_num_threads(cfg.cpu_threads)
    sr = cfg.sr_stems
    mix = decode(song, sr=sr)  # (2, n)
    write_flac(out_dir / "mix.flac", mix, sr)

    model = offline_first(cfg.demucs_model, lambda: get_model(cfg.demucs_model))
    model.eval()
    wav = torch.from_numpy(mix)
    ref = wav.mean(0)
    mean, std = ref.mean(), ref.std() + 1e-8
    with torch.no_grad():
        sources = apply_model(model, ((wav - mean) / std)[None], split=True, overlap=0.25, progress=progress)[0]
    sources = sources * std + mean

    files = {"mix": "mix.flac"}
    for name, src in zip(model.sources, sources):
        write_flac(out_dir / f"{name}.flac", src.numpy(), sr)
        files[name] = f"{name}.flac"
    return {"sr": sr, "duration": mix.shape[1] / sr, "files": files}
