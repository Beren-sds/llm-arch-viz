"""Tests for the weight + golden-activation export (Task 7).

Contract under test (consumed by the TS site in src/compute/):
  - weights.bin round-trips bit-exactly through manifest.json offsets
    (offsets/lengths in float32-element units, little-endian float32 data).
  - goldens.json holds 2 inputs the checkpointed model predicts correctly
    (head.logits argmax at answer positions 16..19 == stored answer).
  - non-finite floats (GPT attn.scores hold -inf) are encoded as the JSON
    strings "Infinity"/"-Infinity"/"NaN" — never as bare JSON literals —
    and decode back to the corresponding floats.
  - golden activation key sets equal the model's recorded key sets exactly.
  - checkpoint selection is strictly via runs.json; missing files fail loudly.
"""

import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from llmviz_train.export import (
    decode_data,
    export_arch,
    load_checkpoint,
)
from llmviz_train.task import TASK
from llmviz_train.train import build_model

ARCHS = ["mamba", "gpt"]
ANSWER_LO = TASK.context_len  # 16
ANSWER_HI = TASK.context_len + TASK.n_data  # 20


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    """Run the real export once per arch into tmpdirs (separate models +
    goldens roots, mirroring public/models/ vs goldens/); return both."""
    models_root = tmp_path_factory.mktemp("models")
    goldens_root = tmp_path_factory.mktemp("goldens")
    summaries = {
        arch: export_arch(arch, models_root / arch, goldens_root / arch) for arch in ARCHS
    }
    return (models_root, goldens_root), summaries


def _load(roots, arch):
    models_root, goldens_root = roots
    manifest = json.loads((models_root / arch / "manifest.json").read_text())
    goldens = json.loads((goldens_root / arch / "goldens.json").read_text())
    blob = np.fromfile(models_root / arch / "weights.bin", dtype="<f4")
    return manifest, goldens, blob


# --- 1. weights round-trip -------------------------------------------------


@pytest.mark.parametrize("arch", ARCHS)
def test_weights_roundtrip_bit_exact(exported, arch):
    root, _ = exported
    manifest, _, blob = _load(root, arch)

    assert manifest["arch"] == arch
    assert manifest["offset_unit"] == "float32"

    state_dict = load_checkpoint(arch)["model_state_dict"]
    names = [t["name"] for t in manifest["tensors"]]
    assert names == sorted(state_dict.keys()), "tensor order must be sorted by name"

    total = 0
    for entry in manifest["tensors"]:
        ref = state_dict[entry["name"]].detach().to(torch.float32).numpy()
        assert entry["offset"] == total, "tensors must be contiguous in manifest order"
        got = blob[entry["offset"] : entry["offset"] + entry["length"]]
        got = got.reshape(entry["shape"])
        assert got.shape == ref.shape
        assert got.tobytes() == ref.astype("<f4").tobytes(), (
            f"{arch}/{entry['name']}: weights.bin not bit-equal to checkpoint"
        )
        total += entry["length"]
    assert total == blob.size, "weights.bin length must equal manifest total"


@pytest.mark.parametrize("arch", ARCHS)
def test_manifest_dims_and_checkpoint(exported, arch):
    root, _ = exported
    manifest, _, _ = _load(root, arch)
    dims = manifest["dims"]
    assert dims["vocab_size"] == TASK.vocab_size
    if arch == "mamba":
        assert dims["dt_rank"] == math.ceil(dims["d_model"] / 16)
    else:
        assert dims["head_dim"] == dims["d_model"] // dims["n_head"]
        assert dims["max_seq_len"] >= TASK.seq_len

    ckpt_meta = manifest["checkpoint"]
    runs_path = Path(__file__).resolve().parent.parent / "runs.json"
    runs = json.loads(runs_path.read_text())
    assert ckpt_meta["run_dir"] == runs[arch]["run_dir"]
    assert isinstance(ckpt_meta["step"], int)
    assert 0.0 <= ckpt_meta["val_exact"] <= 1.0
    assert ckpt_meta["golden_candidates_skipped"] >= 0


# --- 2. goldens correctness ------------------------------------------------


@pytest.mark.parametrize("arch", ARCHS)
def test_golden_inputs_predicted_correctly(exported, arch):
    root, _ = exported
    _, goldens, _ = _load(root, arch)

    assert len(goldens["inputs"]) == 2
    assert len(goldens["activations"]) == 2
    for inp, acts in zip(goldens["inputs"], goldens["activations"]):
        assert len(inp["tokens"]) == TASK.seq_len
        assert len(inp["answer"]) == TASK.n_data
        logits = acts["head.logits"]
        arr = np.array(decode_data(logits["data"])).reshape(logits["shape"])
        preds = arr[ANSWER_LO:ANSWER_HI].argmax(axis=-1)
        assert preds.tolist() == inp["answer"], (
            f"{arch}: stored head.logits argmax disagrees with stored answer"
        )


# --- 3. non-finite encoding ------------------------------------------------


def test_gpt_nonfinite_encoded_as_strings(exported):
    (_, goldens_root), _ = exported
    text = (goldens_root / "gpt" / "goldens.json").read_text()

    def _reject_constant(s):
        raise AssertionError(f"bare non-finite JSON literal {s!r} in goldens.json")

    goldens = json.loads(text, parse_constant=_reject_constant)  # strict parse

    found_neg_inf = False
    for acts in goldens["activations"]:
        for name, tensor in acts.items():
            if name.endswith("attn.scores"):
                assert "-Infinity" in tensor["data"], f"{name}: expected -inf entries"
                found_neg_inf = True
            decoded = decode_data(tensor["data"])
            assert all(isinstance(v, float) for v in decoded)
            if name.endswith("attn.scores"):
                assert any(v == float("-inf") for v in decoded), (
                    f"{name}: '-Infinity' must decode back to -inf"
                )
    assert found_neg_inf, "no attn.scores activation found in gpt goldens"


# --- 4. activation completeness ---------------------------------------------


@pytest.mark.parametrize("arch", ARCHS)
def test_golden_activation_keys_match_model(exported, arch):
    root, _ = exported
    _, goldens, _ = _load(root, arch)

    ckpt = load_checkpoint(arch)
    model = build_model(arch, ckpt["config"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    for inp, acts in zip(goldens["inputs"], goldens["activations"]):
        tokens = torch.tensor([inp["tokens"]], dtype=torch.long)
        with torch.no_grad():
            _, ref_acts = model(tokens, record=True)
        assert set(acts.keys()) == set(ref_acts.keys())
        for name, tensor in acts.items():
            assert tensor["shape"] == list(ref_acts[name].shape)


# --- 5. loud failure on missing checkpoint -----------------------------------


def test_missing_checkpoint_fails_loudly(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_checkpoint("mamba", training_dir=tmp_path)
