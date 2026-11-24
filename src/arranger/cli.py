"""arrange song.mp3 [-o song.arrangement.json]"""

import argparse
import dataclasses
import json
import logging
import sys
from pathlib import Path

from arranger.cache import Workdir, default_root
from arranger.config import DEFAULT
from arranger.pipeline import STAGES, run


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="arrange", description="Extract a per-bar arrangement draft from a song.")
    p.add_argument("song", type=Path)
    p.add_argument("-o", "--output", type=Path, help="output JSON (default: <song>.arrangement.json)")
    p.add_argument("--work-dir", type=Path, default=default_root(), help="stage cache root (default: %(default)s)")
    p.add_argument("--redo", default="", help=f"comma-separated stages to recompute: {','.join(STAGES)},all")
    p.add_argument("--lang", help="lyrics language code, e.g. en, ca, es (default: auto-detect)")
    p.add_argument("--whisper-model", default=DEFAULT.whisper_model, help="e.g. large-v3-turbo, medium, small")
    p.add_argument("-q", "--quiet", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    logging.getLogger("arranger").setLevel(logging.WARNING if args.quiet else logging.INFO)
    if not args.song.is_file():
        p.error(f"no such file: {args.song}")
    redo = {s.strip() for s in args.redo.split(",") if s.strip()}
    if bad := redo - set(STAGES) - {"all"}:
        p.error(f"unknown stage(s): {', '.join(sorted(bad))}")

    cfg = dataclasses.replace(DEFAULT, whisper_model=args.whisper_model, whisper_language=args.lang)
    work = Workdir(args.work_dir, args.song, redo)
    logging.getLogger("arranger").info("cache: %s", work.dir)
    result = run(args.song, cfg, work)

    out = args.output or args.song.with_suffix(".arrangement.json")
    out.write_text(json.dumps(result, indent=1, ensure_ascii=False))
    logging.getLogger("arranger").info("wrote %s (%d bars)", out, len(result["bars"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
