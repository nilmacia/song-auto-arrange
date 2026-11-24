"""Model ids and tunable thresholds. Change these rather than editing stage code."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    # --- models ---
    demucs_model: str = "htdemucs"
    beat_checkpoint: str = "final0"
    whisper_model: str = "large-v3-turbo"
    whisper_compute_type: str = "int8"
    whisper_language: str | None = None  # None = auto-detect
    cpu_threads: int = 4

    # --- sample rates ---
    sr_stems: int = 44100
    sr_chords: int = 22050  # BTC was trained at 22.05 kHz
    sr_whisper: int = 16000

    # --- tempo ---
    tempo_tolerance: float = 0.04  # local tempo within ±4% of the song tempo is reported as the song tempo

    # --- key ---
    key_window_bars: int = 8  # chord context per bar for local key
    key_switch_penalty: float = 1.0  # higher = fewer modulations
    key_chord_quality_weight: float = 0.25  # weight of chord thirds/7ths vs bass notes + chord roots

    # --- dynamics: dB relative to the song's median bar (= "mf"); mastered songs have little range ---
    dyn_thresholds: tuple[tuple[float, str], ...] = (
        (2.5, "ff"),
        (1.0, "f"),
        (-1.5, "mf"),
        (-4.0, "mp"),
        (-9.0, "p"),
    )  # anything quieter -> "pp"

    # --- per-bar extras ---
    playing_threshold: float = 0.35  # stem activity needed to list a stem in "playing"
    lyric_line_gap: float = 1.0  # seconds of silence that start a new lyric line

    # --- sections ---
    novelty_half_kernel_bars: int = 4
    min_section_bars: int = 2
    max_section_bars: int = 16  # longer segments get split on a 4-bar phrase grid
    cluster_threshold: float = 0.55  # segment similarity needed to share a letter
    vocal_activity_threshold: float = 0.25


DEFAULT = Config()
