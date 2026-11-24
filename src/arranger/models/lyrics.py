"""Lyrics transcription with faster-whisper on the separated vocal stem."""

import numpy as np
from faster_whisper import WhisperModel

from arranger.config import Config
from arranger.models.hub import offline_first


def run(vocals: np.ndarray, cfg: Config) -> dict:
    """vocals: mono float32 at 16 kHz. Returns language and word timings."""
    model = offline_first(
        f"whisper {cfg.whisper_model}",
        lambda: WhisperModel(
            cfg.whisper_model, device="cpu", compute_type=cfg.whisper_compute_type, cpu_threads=cfg.cpu_threads
        ),
    )
    segments, info = model.transcribe(
        vocals,
        language=cfg.whisper_language,
        word_timestamps=True,
        vad_filter=True,
        condition_on_previous_text=False,  # avoids runaway repetition on sung audio
    )
    words = [
        [round(w.start, 3), round(w.end, 3), w.word.strip(), round(w.probability, 3)]
        for seg in segments
        for w in (seg.words or [])
        if w.word.strip()
    ]
    return {"language": info.language, "words": words}
