"""Per-song cache so expensive model stages run once per song.

A song is identified by the SHA-1 of its file *contents*, so renaming or moving the file still hits
the cache. Each stage result is stored with the settings that produced it (model, language, ...) and
recomputed automatically when those settings change."""

import hashlib
import json
import logging
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from arranger.paths import SONGS_DIR

log = logging.getLogger(__name__)


def default_root() -> Path:
    return SONGS_DIR


def song_hash(path: Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()[:12]


class Workdir:
    def __init__(self, root: Path, song: Path, redo: set[str] | None = None):
        digest = song_hash(song)
        existing = sorted(root.glob(f"*-{digest}")) if root.exists() else []
        self.dir = existing[0] if existing else root / f"{song.stem}-{digest}"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.redo = redo or set()
        self.timings: dict[str, float] = {}

    def path(self, name: str) -> Path:
        return self.dir / name

    def stage(self, name: str, fn: Callable[[], Any], params: dict | None = None) -> Any:
        """Cached result of `name`; recomputed if missing, forced via `redo`, or `params` changed."""
        out = self.path(f"{name}.json")
        params = params or {}
        if out.exists() and name not in self.redo and "all" not in self.redo:
            cached = json.loads(out.read_text())
            if cached.get("params") == params:
                log.info("%-9s cached", name)
                return cached["result"]
            log.info("%-9s settings changed %s -> %s", name, cached.get("params"), params)
        log.info("%-9s running...", name)
        t0 = time.perf_counter()
        result = fn()
        self.timings[name] = time.perf_counter() - t0
        log.info("%-9s done in %.0fs", name, self.timings[name])
        out.write_text(json.dumps({"params": params, "result": result}))
        return result
