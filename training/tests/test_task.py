"""Tests for the selective-copying task generator.

Sequence layout (length 21, positions 0..20):
  0..15   context: exactly n_data=4 data tokens (ids 1..8) at random
          positions, NOISE everywhere else
  16      GO token
  17..20  the 4 data tokens, teacher-forced as inputs
  y[t]    next-token target for input position t, so y[16..19] are the
          4 data tokens in context order; y[20] is unused (0/PAD)
  mask    True exactly at positions 16..19
"""

import torch

from llmviz_train.task import TASK, make_batch


def test_batch_shapes_and_structure():
    x, y, mask = make_batch(32, torch.Generator().manual_seed(0))
    assert x.shape == (32, 21) and y.shape == (32, 21) and mask.shape == (32, 21)
    assert (x[:, 16] == TASK.go_id).all()
    assert mask[:, 16:20].all() and mask[:, :16].sum() == 0
    ctx = x[:, :16]
    assert ((ctx == TASK.noise_id) | ((ctx >= 1) & (ctx <= 8))).all()
    for b in range(32):
        data = ctx[b][ctx[b] != TASK.noise_id]
        assert torch.equal(y[b, 16:20], data)


def test_determinism():
    a = make_batch(8, torch.Generator().manual_seed(7))
    b = make_batch(8, torch.Generator().manual_seed(7))
    assert all(torch.equal(p, q) for p, q in zip(a, b))


def test_exactly_n_data_tokens_per_context():
    x, _, _ = make_batch(64, torch.Generator().manual_seed(1))
    ctx = x[:, :16]
    n_data_per_row = (ctx != TASK.noise_id).sum(dim=1)
    assert (n_data_per_row == TASK.n_data).all()


def test_teacher_forcing_inputs_match_data_tokens():
    x, y, _ = make_batch(64, torch.Generator().manual_seed(2))
    # Inputs at 17..20 hold the previous answer tokens: x[17] = 1st data
    # token, ..., x[20] = 4th data token — identical to y[16..19].
    assert torch.equal(x[:, 17:21], y[:, 16:20])
    for b in range(64):
        data = x[b, :16][x[b, :16] != TASK.noise_id]
        assert torch.equal(x[b, 17:21], data)


def test_mask_and_target_padding():
    x, y, mask = make_batch(16, torch.Generator().manual_seed(3))
    assert mask.dtype == torch.bool
    assert not mask[:, 20].any()
    assert (mask.sum(dim=1) == TASK.n_data).all()
    # y[20] is unused -> 0/PAD.
    assert (y[:, 20] == 0).all()


def test_task_config_matches_yaml():
    assert TASK.vocab_size == 16
    assert TASK.n_data == 4
    assert TASK.context_len == 16
    assert TASK.data_ids == [1, 2, 3, 4, 5, 6, 7, 8]
    assert TASK.noise_id == 9
    assert TASK.go_id == 10


def test_different_seeds_differ():
    a, _, _ = make_batch(32, torch.Generator().manual_seed(0))
    b, _, _ = make_batch(32, torch.Generator().manual_seed(1))
    assert not torch.equal(a, b)
