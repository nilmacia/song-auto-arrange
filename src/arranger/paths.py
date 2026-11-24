"""Where the tool keeps its caches: everything lives under <project>/.cache so the tool is self-contained.

    .cache/models/huggingface   Demucs + Whisper weights (HF_HOME)
    .cache/models/torch         beat_this weights (TORCH_HOME)
    .cache/songs/<name>-<hash>  per-song stems and model outputs

Override the root with ARRANGER_CACHE. If the package is installed outside a checkout (no
pyproject.toml next to src/), the root falls back to ~/.cache/arranger."""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def cache_root() -> Path:
    if env := os.environ.get("ARRANGER_CACHE"):
        return Path(env).expanduser()
    if (PROJECT_ROOT / "pyproject.toml").is_file():
        return PROJECT_ROOT / ".cache"
    return Path.home() / ".cache" / "arranger"


CACHE_ROOT = cache_root()
MODELS_DIR = CACHE_ROOT / "models"
SONGS_DIR = CACHE_ROOT / "songs"


def use_local_model_cache() -> None:
    """Point Hugging Face and torch hub downloads at MODELS_DIR. Must run before huggingface_hub is
    imported (it reads HF_HOME at import time); arranger/__init__.py does this."""
    os.environ["HF_HOME"] = str(MODELS_DIR / "huggingface")
    os.environ["TORCH_HOME"] = str(MODELS_DIR / "torch")
