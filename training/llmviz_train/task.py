"""Selective-copying task: batch generator and task config.

Sequence layout (length context_len + 1 + n_data = 21, positions 0..20):

  0..15   context: exactly n_data data tokens (ids in data_ids) at random
          positions, NOISE everywhere else
  16      GO token
  17..20  the n_data data tokens in context order, teacher-forced as inputs

Targets ``y`` are next-token aligned: ``y[t]`` is the target for input
position ``t``, so ``y[16..19]`` hold the data tokens. All other positions
are 0 (PAD) and excluded from the loss via ``mask``, which is True exactly
at positions 16..19.
"""

from dataclasses import dataclass

import torch

from llmviz_train.config import load_config


@dataclass(frozen=True)
class TaskConfig:
    vocab_size: int
    n_data: int
    context_len: int
    data_ids: list[int]
    noise_id: int
    go_id: int

    @property
    def seq_len(self) -> int:
        return self.context_len + 1 + self.n_data


TASK = TaskConfig(**load_config()["task"])


def make_batch(
    batch: int, gen: torch.Generator
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate a batch of selective-copying examples.

    All randomness flows through ``gen``; the global torch RNG is untouched.

    Returns:
        x:    (batch, 21) long — input tokens.
        y:    (batch, 21) long — next-token targets, 0 outside the mask.
        mask: (batch, 21) bool — True at the n_data answer positions (16..19).
    """
    L, n = TASK.context_len, TASK.n_data
    T = TASK.seq_len

    # Data token positions: a random subset of the L context slots, sorted.
    scores = torch.rand(batch, L, generator=gen)
    pos = scores.topk(n, dim=1).indices.sort(dim=1).values

    # Data token values: sampled with replacement from data_ids.
    ids = torch.tensor(TASK.data_ids, dtype=torch.long)
    vals = ids[torch.randint(len(ids), (batch, n), generator=gen)]

    x = torch.full((batch, T), TASK.noise_id, dtype=torch.long)
    x[:, :L].scatter_(1, pos, vals)
    x[:, L] = TASK.go_id
    x[:, L + 1 :] = vals  # teacher forcing: previous answer tokens as inputs

    y = torch.zeros(batch, T, dtype=torch.long)
    y[:, L : L + n] = vals

    mask = torch.zeros(batch, T, dtype=torch.bool)
    mask[:, L : L + n] = True

    return x, y, mask
