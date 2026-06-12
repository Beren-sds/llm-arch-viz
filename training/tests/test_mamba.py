"""Tests for the minimal Mamba reference implementation.

The model is the golden-activation source for the TypeScript port, so the
tests pin down the externally observable contract:
  - logits shape (B, T, vocab)
  - causality (token t cannot influence logits at positions < t)
  - the exact set of recorded activation names and shapes
  - determinism under seed_all
"""

import torch

from llmviz_train.config import load_config
from llmviz_train.mamba import Mamba
from llmviz_train.seed import seed_all

VOCAB = 16
T = 21
CFG = load_config()["mamba"]

D_MODEL = CFG["d_model"]  # 48
D_STATE = CFG["d_state"]  # 8
D_CONV = CFG["d_conv"]  # 4
D_INNER = CFG["expand"] * CFG["d_model"]  # 96
DT_RANK = -(-D_MODEL // 16)  # ceil(d_model / 16) = 3
N_LAYER = CFG["n_layer"]  # 2


def make_model(seed: int = 0) -> Mamba:
    seed_all(seed)
    return Mamba(CFG, vocab_size=VOCAB)


def make_tokens(batch: int, seed: int = 0) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    return torch.randint(0, VOCAB, (batch, T), generator=gen)


def test_logits_shape():
    model = make_model()
    tokens = make_tokens(4)
    logits = model(tokens)
    assert logits.shape == (4, T, VOCAB)


def test_causality():
    """Changing the token at position 10 must not change logits before 10,
    and must change logits at position 10 (checked over several seeds)."""
    model = make_model()
    for seed in range(3):
        a = make_tokens(2, seed=seed)
        b = a.clone()
        b[:, 10] = (a[:, 10] + 1) % VOCAB
        assert not torch.equal(a, b)
        with torch.no_grad():
            la = model(a)
            lb = model(b)
        assert torch.allclose(la[:, :10], lb[:, :10], atol=1e-6)
        assert not torch.allclose(la[:, 10], lb[:, 10], atol=1e-6)


def expected_recording_keys() -> dict[str, tuple[int, ...]]:
    keys: dict[str, tuple[int, ...]] = {
        "embed.out": (T, D_MODEL),
        "final_norm.out": (T, D_MODEL),
        "head.logits": (T, VOCAB),
    }
    for i in range(N_LAYER):
        keys[f"layer{i}.norm.out"] = (T, D_MODEL)
        keys[f"layer{i}.in_proj.out"] = (T, 2 * D_INNER)
        keys[f"layer{i}.conv.out"] = (T, D_INNER)
        keys[f"layer{i}.x_proj.out"] = (T, DT_RANK + 2 * D_STATE)
        keys[f"layer{i}.dt.out"] = (T, D_INNER)
        for t in range(T):
            keys[f"layer{i}.ssm.h.t{t}"] = (D_INNER, D_STATE)
        keys[f"layer{i}.ssm.out"] = (T, D_INNER)
        keys[f"layer{i}.gate.out"] = (T, D_INNER)
        keys[f"layer{i}.out_proj.out"] = (T, D_MODEL)
    return keys


def test_recording_complete_key_set_and_shapes():
    model = make_model()
    tokens = make_tokens(3)
    logits, acts = model(tokens, record=True)
    assert logits.shape == (3, T, VOCAB)

    expected = expected_recording_keys()
    assert set(acts.keys()) == set(expected.keys())
    for name, shape in expected.items():
        tensor = acts[name]
        assert tensor.shape == torch.Size(shape), f"{name}: {tensor.shape}"
        assert tensor.dtype == torch.float32, name
        assert tensor.device.type == "cpu", name
        assert not tensor.requires_grad, name


def test_recording_matches_unrecorded_logits():
    model = make_model()
    tokens = make_tokens(2)
    with torch.no_grad():
        plain = model(tokens)
        recorded, acts = model(tokens, record=True)
    assert torch.equal(plain, recorded)
    # Recorded logits are for batch item 0.
    assert torch.allclose(acts["head.logits"], plain[0].float(), atol=1e-7)


def test_determinism_under_seed_all():
    tokens = make_tokens(4)
    with torch.no_grad():
        la = make_model(seed=1128)(tokens)
        lb = make_model(seed=1128)(tokens)
    assert torch.equal(la, lb)
