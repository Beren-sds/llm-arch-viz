"""Tests for the minimal GPT reference implementation.

The model is the golden-activation source for the TypeScript port, so the
tests pin down the externally observable contract:
  - logits shape (B, T, vocab)
  - causality (token t cannot influence logits at positions < t)
  - the exact set of recorded activation names and shapes
  - recorded tensors are detached, cloned, float32, CPU
  - determinism under seed_all
  - finite gradients through a masked-CE training step
"""

import torch

from llmviz_train.config import load_config
from llmviz_train.gpt import GPT
from llmviz_train.seed import seed_all
from llmviz_train.task import make_batch

VOCAB = 16
T = 21
CFG = load_config()["gpt"]

D_MODEL = CFG["d_model"]  # 48
N_HEAD = CFG["n_head"]  # 3
HEAD_DIM = D_MODEL // N_HEAD  # 16
D_MLP = CFG["mlp_ratio"] * D_MODEL  # 192
N_LAYER = CFG["n_layer"]  # 2


def make_model(seed: int = 0) -> GPT:
    seed_all(seed)
    return GPT(CFG, vocab_size=VOCAB)


def make_tokens(batch: int, seed: int = 0) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    return torch.randint(0, VOCAB, (batch, T), generator=gen)


def test_logits_shape():
    model = make_model()
    tokens = make_tokens(4)
    logits = model(tokens)
    assert logits.shape == (4, T, VOCAB)


def test_causality():
    """Changing the token at position 10 must not change logits before 10
    (bit-exactly: the causal mask zeroes future contributions), and must
    change logits at position 10 (checked over several seeds)."""
    model = make_model()
    for seed in range(3):
        a = make_tokens(2, seed=seed)
        b = a.clone()
        b[:, 10] = (a[:, 10] + 1) % VOCAB
        assert not torch.equal(a, b)
        with torch.no_grad():
            la = model(a)
            lb = model(b)
        assert torch.equal(la[:, :10], lb[:, :10])
        assert not torch.allclose(la[:, 10], lb[:, 10], atol=1e-6)


def expected_recording_keys() -> dict[str, tuple[int, ...]]:
    keys: dict[str, tuple[int, ...]] = {
        "embed.out": (T, D_MODEL),
        "final_norm.out": (T, D_MODEL),
        "head.logits": (T, VOCAB),
    }
    for i in range(N_LAYER):
        keys[f"layer{i}.ln1.out"] = (T, D_MODEL)
        keys[f"layer{i}.attn.q"] = (N_HEAD, T, HEAD_DIM)
        keys[f"layer{i}.attn.k"] = (N_HEAD, T, HEAD_DIM)
        keys[f"layer{i}.attn.v"] = (N_HEAD, T, HEAD_DIM)
        keys[f"layer{i}.attn.scores"] = (N_HEAD, T, T)
        keys[f"layer{i}.attn.weights"] = (N_HEAD, T, T)
        keys[f"layer{i}.attn.out"] = (T, D_MODEL)
        keys[f"layer{i}.ln2.out"] = (T, D_MODEL)
        keys[f"layer{i}.mlp.fc"] = (T, D_MLP)
        keys[f"layer{i}.mlp.act"] = (T, D_MLP)
        keys[f"layer{i}.mlp.proj"] = (T, D_MODEL)
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


def test_recording_masked_scores_and_normalized_weights():
    """scores are post-mask (upper triangle -inf), weights post-softmax
    (rows sum to 1, upper triangle exactly 0)."""
    model = make_model()
    tokens = make_tokens(2)
    with torch.no_grad():
        _, acts = model(tokens, record=True)
    upper = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
    for i in range(N_LAYER):
        scores = acts[f"layer{i}.attn.scores"]
        weights = acts[f"layer{i}.attn.weights"]
        assert torch.isinf(scores[:, upper]).all() and (scores[:, upper] < 0).all()
        assert torch.isfinite(scores[:, ~upper]).all()
        assert (weights[:, upper] == 0).all()
        assert torch.allclose(weights.sum(dim=-1), torch.ones(N_HEAD, T), atol=1e-6)


def test_recording_matches_unrecorded_logits():
    model = make_model()
    tokens = make_tokens(2)
    with torch.no_grad():
        plain = model(tokens)
        recorded, acts = model(tokens, record=True)
    assert torch.equal(plain, recorded)
    # Recorded logits are for batch item 0.
    assert torch.allclose(acts["head.logits"], plain[0].float(), atol=1e-7)
    # Snapshots are clones: they must not alias the live logits storage.
    assert acts["head.logits"].data_ptr() != recorded.data_ptr()


def test_determinism_under_seed_all():
    tokens = make_tokens(4)
    with torch.no_grad():
        la = make_model(seed=1128)(tokens)
        lb = make_model(seed=1128)(tokens)
    assert torch.equal(la, lb)


def test_gradients_finite():
    """Masked-CE training step: every parameter receives a finite gradient."""
    model = make_model()
    gen = torch.Generator().manual_seed(0)
    x, y, mask = make_batch(8, gen)
    logits = model(x)
    loss = torch.nn.functional.cross_entropy(logits[mask], y[mask])
    assert torch.isfinite(loss)
    loss.backward()
    for name, param in model.named_parameters():
        assert param.grad is not None, name
        assert torch.isfinite(param.grad).all(), name
