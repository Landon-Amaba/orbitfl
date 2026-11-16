"""Random seed utilities.

seed_everything seeds Python's random module, numpy and torch. spawn_rng derives
an independent, named numpy Generator from a master seed, so that separate
random streams (for example world noise and client sampling) do not interact.
"""

from __future__ import annotations

import hashlib
import random

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """
    Seed Python's random, numpy, and torch with the given integer.

    Parameters
    ----------
    seed : int  Master seed for the current run. Should match cfg.seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def spawn_rng(seed: int, stream_name: str) -> np.random.Generator:
    """Derive a named, independent numpy Generator from a master seed.

    The stream name is hashed with SHA-256 and folded into a 32-bit integer, which
    is used as a second seed word next to the master seed. Streams with different
    names are therefore independent even under the same master seed.

    Parameters
    ----------
    seed        : int  Master seed (e.g. cfg.seed).
    stream_name : str  Stream identifier, e.g. "world_noise" or "fl_sample".

    Returns
    -------
    np.random.Generator  A fresh Generator seeded from (seed, stream_hash).
    """
    digest = hashlib.sha256(f"{seed}:{stream_name}".encode()).hexdigest()
    stream_int = int(digest, 16) & 0xFFFF_FFFF   # fold to 32-bit
    return np.random.default_rng([seed, stream_int])
