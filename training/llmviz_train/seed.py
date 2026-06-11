"""Single-source seeding for all randomness in the training pipeline.

Every random source (random, numpy, torch CPU/MPS/CUDA, DataLoader
generators) must derive from the one seed in config.yaml via seed_all().
"""

import random

import numpy as np
import torch


def seed_all(seed: int) -> torch.Generator:
    """Seed every global RNG and return a fresh CPU torch.Generator.

    Seeds `random`, `numpy`, and `torch` (CPU, plus MPS/CUDA when
    available). The returned generator is independent of the global
    torch RNG; pass it to DataLoaders / samplers that accept one.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return torch.Generator().manual_seed(seed)
