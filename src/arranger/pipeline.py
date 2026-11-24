"""Run the model stages (cached) and derive the per-bar arrangement."""

from pathlib import Path

import numpy as np

from arranger.audio import read_audio, resample, to_mono
from arranger.cache import Workdir
from arranger.config import Config
from arranger import schema
from arranger.derive import dynamics, harmony, sections
from arranger.derive.bars import Bar, bar_index, build_bars, label_spans

STAGES = ("separate", "beats", "chords", "lyrics")


def run(song: Path, cfg: Config, work: Workdir) -> dict:
    from arranger.models import beats as beats_m, chords as chords_m, lyrics as lyrics_m, separate as separate_m

    sep = work.stage("separate", lambda: separate_m.run(song, work.dir, cfg), {"model": cfg.demucs_model})
    audio = {name: read_audio(work.path(f))[0] for name, f in sep["files"].items()}
    sr = sep["sr"]
    mix = audio["mix"]
    harmonic = resample(to_mono(audio["bass"] + audio["other"]), sr, cfg.sr_chords)

    beat = work.stage("beats", lambda: beats_m.run(mix, sr, cfg), {"checkpoint": cfg.beat_checkpoint})
    chord = work.stage("chords", lambda: chords_m.run(harmonic, cfg.sr_chords, cfg), {"model": "btc-large-voca"})
    lyric = work.stage(
        "lyrics",
        lambda: lyrics_m.run(resample(to_mono(audio["vocals"]), sr, cfg.sr_whisper), cfg),
        {"model": cfg.whisper_model, "compute_type": cfg.whisper_compute_type, "language": cfg.whisper_language},
    )

    return assemble(
        song=song, duration=sep["duration"], sr=sr, audio=audio, harmonic=harmonic,
        beats=beat, chords=chord["chords"], lyrics=lyric, cfg=cfg, stems_dir=work.dir,
    )


def assemble(
    *, song: Path, duration: float, sr: int, audio: dict[str, np.ndarray], harmonic: np.ndarray,
    beats: dict, chords: list, lyrics: dict, cfg: Config, stems_dir: Path | None = None,
) -> dict:
    bars: list[Bar] = build_bars(beats["beats"], beats["downbeats"], duration)
    n = len(bars)

    # chords: one label per beat
    beat_chords = [label_spans(b.beat_spans(), chords) for b in bars]

    # key evidence: bass-stem notes + chord roots, with chord qualities down-weighted. Thirds are the
    # unreliable part: rock power chords have none (the chord model calls them major) and distorted
    # guitars add a strong major-third overtone, so guitar chroma and chord qualities lean major.
    def per_bar(fn):
        m = np.stack([
            sum((fn(c) * (e - s) for c, (s, e) in zip(bc, b.beat_spans())), np.zeros(12))
            for b, bc in zip(bars, beat_chords)
        ])
        return m / (m.sum(1, keepdims=True) + 1e-8)

    bass_chroma = harmony.bar_audio_chroma(
        resample(to_mono(audio["bass"]), sr, cfg.sr_chords), cfg.sr_chords,
        np.array([b.start for b in bars]), np.array([b.end for b in bars]),
    )
    key_evidence = bass_chroma + per_bar(harmony.root_chroma) + cfg.key_chord_quality_weight * per_bar(harmony.chord_chroma)
    global_key, bar_keys = harmony.estimate_keys(key_evidence, cfg.key_window_bars, cfg.key_switch_penalty)

    # dynamics + stem activity
    loud = dynamics.bar_db(to_mono(audio["mix"]), sr, bars)
    dyn_marks, ref = dynamics.marks(loud, cfg)
    stem_names = [k for k in ("vocals", "drums", "bass", "other") if k in audio]
    activity = dynamics.stem_activity({k: dynamics.bar_db(to_mono(audio[k]), sr, bars) for k in stem_names}, ref)

    # lyrics: words to bars by onset, "/ " marks the start of a lyric line
    words = lyrics["words"]
    bar_words: list[list[str]] = [[] for _ in range(n)]
    if words:
        texts = schema.lyric_text(words, cfg.lyric_line_gap)
        for t, i in zip(texts, bar_index([w[0] for w in words], bars)):
            bar_words[i].append(t)
    words_per_bar = np.array([len(w) for w in bar_words])

    segs = sections.analyse(harmonic, cfg.sr_chords, bars, beat_chords, activity, words_per_bar, loud, cfg)
    seg_at = {seg.start: seg for seg in segs}

    # tempo: beat positions are quantised (~20 ms), so average over ±2 bars, then keep only sustained changes
    beats_n = np.array([b.n_beats for b in bars])
    dur = np.array([b.end - b.start for b in bars])
    bpm = np.array([60 * beats_n[max(0, i - 2) : i + 3].sum() / dur[max(0, i - 2) : i + 3].sum() for i in range(n)])
    song_bpm = float(np.median(bpm))
    tempo = schema.tempo_track(bpm, song_bpm, cfg.tempo_tolerance)

    out_bars = []
    number = 0 if bars and bars[0].pickup else 1
    for i, b in enumerate(bars):
        rec = {"bar": number, "start": round(b.start, 3), "end": round(b.end, 3)}
        if b.pickup:
            rec["pickup"] = True
        if i in seg_at:
            rec["section"] = {"id": f"{seg_at[i].letter}{seg_at[i].occurrence}", "type": seg_at[i].label}
        rec |= {
            "bpm": tempo[i],
            "time_signature": b.time_signature,
            "key": harmony.key_index_name(bar_keys[i]),
            "chords": [harmony.spell(c, bar_keys[i]) for c in beat_chords[i]],
            "lyrics": " ".join(bar_words[i]),
            "dynamics": dyn_marks[i],
            "playing": [k for k in stem_names if activity[k][i] >= cfg.playing_threshold],
        }
        out_bars.append(rec)
        number += 1

    meters = [b.time_signature for b in bars if not b.pickup] or ["4/4"]
    return {
        "song": {
            "file": song.name,
            "duration": round(duration, 2),
            "bpm": int(round(song_bpm)) if n else None,
            "time_signature": max(set(meters), key=meters.count),
            "key": harmony.key_index_name(global_key),
            "lyrics_language": lyrics.get("language"),
            "stems_dir": str(stems_dir) if stems_dir else None,
        },
        "bars": schema.compact(out_bars),
    }
