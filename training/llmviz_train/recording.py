"""Shared activation-recording helper for the reference models."""

import torch


def snapshot(tensor: torch.Tensor) -> torch.Tensor:
    """Detached float32 CPU copy of ``tensor`` for activation recording.

    The trailing clone matters: on a CPU float32 model, ``.to`` is a no-op
    that would alias live forward-pass storage, so later in-place ops could
    silently corrupt the snapshot.
    """
    return tensor.detach().to("cpu", torch.float32).clone()
