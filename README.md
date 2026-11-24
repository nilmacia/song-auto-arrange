# arranger

Draft a per-bar arrangement from an audio file, locally on CPU: tempo, time signature, key, chords, dynamics, section type, section id and lyrics. It's meant as a first pass for the band to correct.

```bash
uv sync
uv run arrange song.mp3                # writes song.arrangement.json
uv run arrange song.mp3 --lang ca      # force the lyrics language
uv run arrange song.mp3 --whisper-model small   # faster, worse lyrics
uv run arrange song.mp3 --redo chords  # recompute one cached stage (separate, beats, chords, lyrics, all)
```

On an i5-6300U a song takes roughly 2.3× its length. Demucs and Whisper account for about 95% of that. All caches live inside the project, under `.cache/` (git-ignored):

| Path | What |
|---|---|
| `.cache/models/` | Model weights (~1.8 GB), downloaded once on first use. After that, runs don't touch the network. |
| `.cache/songs/<song>-<hash>/` | Per-song stems (FLAC) and model outputs |

A song's key is its file *content* hash, so a renamed or moved copy is still a cache hit. A stage reruns by itself when its settings change, e.g. `--whisper-model` or `--lang`. Set `ARRANGER_CACHE=/some/dir` to put the caches elsewhere.

## Output format
The output is a list of bars. Each bar always has its own content. Section, tempo, meter and key are written only on the bar where they start or change, the way a score shows them. The first bar carries all of them.

```json
{"song": {"file": "song.mp3", "duration": 331.56, "bpm": 124, "time_signature": "4/4", "key": "E minor",
          "lyrics_language": "en", "stems_dir": ".../.cache/songs/song-300ecfe344ce"},
 "bars": [
  {"bar": 1, "start": 1.3, "end": 3.28, "section": {"id": "A1", "type": "intro"},
   "bpm": 124, "time_signature": "4/4", "key": "E minor",
   "chords": ["Em", "Em", "Em", "A7"], "lyrics": "", "dynamics": "pp", "playing": ["other"]},
  {"bar": 2, "start": 3.28, "end": 5.28,
   "chords": ["Em", "Em", "Em", "A7"], "lyrics": "", "dynamics": "pp", "playing": ["other"]},
  {"bar": 46, "start": 86.16, "end": 88.1, "section": {"id": "B1", "type": "chorus"}, "key": "F# minor",
   "chords": ["F#m", "F#m", "Bm", "F#m"], "lyrics": "/ first words of a line", "dynamics": "f", "playing": ["vocals", "drums", "bass", "other"]}
 ]}
```

| Field | Meaning |
|---|---|
| `bar` | Bar number. A pickup before the first downbeat is bar 0, with `"pickup": true`. |
| `start` / `end` | Seconds in the audio, for playback sync |
| `section` | On the first bar of a section. `id` is the repetition letter plus occurrence: `B2` = 2nd time of material B. |
| `bpm`, `time_signature`, `key` | From this bar on |
| `chords` | One chord per beat (`N.C.` = no chord) |
| `lyrics` | Words starting in this bar; `/` marks the start of a lyric line |
| `dynamics` | pp…ff |
| `playing` | Stems audible in this bar: vocals, drums, bass, other |
| `stems_dir` | Folder with `mix/vocals/drums/bass/other.flac` for playback or practice |

`arranger.schema.resolve(data)` returns every bar with section, tempo, meter and key filled in.

## How each field is produced
| Field | Source |
|---|---|
| bars, bpm, time signature | [beat_this](https://github.com/CPJKU/beat_this) beats + downbeats. Missed or extra downbeats and beat dropouts are repaired against the song's main meter. A tempo change is written only when the tempo moves more than 4% for at least 2 bars. |
| chords | [BTC](https://github.com/jayg996/BTC-ISMIR19) large vocabulary (vendored in `src/arranger/vendor/btc`, MIT), run on the bass + other stems. No inversions, no power chords (an E5 shows as E). |
| key | Bass-stem notes plus chord roots, with chord qualities weighted down. Thirds are unreliable in rock: power chords and distortion both lean major. Krumhansl profiles, with Viterbi smoothing so it changes only on real modulations. |
| dynamics | Bar RMS relative to the song's median bar (median = mf) |
| playing | Per-stem bar level from [Demucs](https://github.com/facebookresearch/demucs) `htdemucs` |
| sections | Harmony and arrangement self-similarity, then novelty boundaries, then repetition letters. Types come from heuristics. |
| lyrics | [faster-whisper](https://github.com/SYSTRAN/faster-whisper) `large-v3-turbo` on the vocal stem; each word goes to the bar it starts in |

Thresholds are in `src/arranger/config.py`.

## Known limits
- 6/8 comes out as 3/4 (or 2/4). The tracker counts quarter-note-level beats.
- Section *types* are guesses. Boundaries and repetition letters are more reliable.
- Lyrics on dense mixes or in less common languages will need editing.
- Key from chords alone (jazz ii–V–I with little bass) can lean towards the V key.

## Tests
```bash
uv run pytest            # includes a ~15 s test running the real beat/chord models on synthetic audio
uv run pytest -m 'not slow'
```
