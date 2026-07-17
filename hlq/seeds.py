"""Reproducibility utilities (the standard ``set_seed`` block, per plan Sec. 8)."""
from __future__ import annotations

import os
import random

import numpy as np


def set_seed(seed: int) -> np.random.Generator:
    """Seed all RNGs used in the project and return a fresh NumPy Generator.

    Seeding ``random``, ``numpy`` (legacy + default hash), ``torch`` (CPU) and
    ``PYTHONHASHSEED`` makes an experiment cell fully determined by ``seed``.
    The returned :class:`numpy.random.Generator` is the *only* stochastic source
    the attacks/oracle should draw from, so that a run is reproducible from the
    single integer ``seed`` regardless of global-state interference.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(False)  # small circuits; keep speed
    except Exception:  # torch optional at import time
        pass
    return np.random.default_rng(seed)
