"""Load models from the local Hugging Face cache without touching the network; go online only
if the model is not cached yet (first run)."""

import logging
from collections.abc import Callable
from typing import TypeVar

from huggingface_hub import constants

log = logging.getLogger(__name__)
T = TypeVar("T")


def offline_first(what: str, load: Callable[[], T]) -> T:
    previous = constants.HF_HUB_OFFLINE
    constants.HF_HUB_OFFLINE = True
    try:
        return load()
    except Exception:
        if previous:  # the user asked for offline mode: don't go online behind their back
            raise
        log.info("downloading %s (first run only)", what)
    finally:
        constants.HF_HUB_OFFLINE = previous
    return load()
