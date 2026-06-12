"""Smoke test for the training loop: 30 steps must reduce the masked CE loss.

Relaxed criterion (per plan): mean of the last 5 step losses < mean of the
first 5, and every loss is finite. Small batch (32) keeps the whole test
under ~60 s for both architectures combined.
"""

import math

import pytest

from llmviz_train.config import load_config
from llmviz_train.train import train_loop


@pytest.mark.parametrize("arch", ["mamba", "gpt"])
def test_loss_decreases_over_30_steps(arch):
    cfg = load_config()
    result = train_loop(arch, cfg, steps=30, batch_size=32, eval_every=0, out_dir=None)

    losses = result["step_losses"]
    assert len(losses) == 30
    assert all(math.isfinite(loss) for loss in losses), "non-finite training loss"
    first5 = sum(losses[:5]) / 5
    last5 = sum(losses[-5:]) / 5
    assert last5 < first5, f"{arch}: loss did not decrease ({first5:.4f} -> {last5:.4f})"
