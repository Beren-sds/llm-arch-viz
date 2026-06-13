"""Export trained weights + golden activations for the TS site.

Artifacts per arch, written by the ``training/export.py`` CLI wrapper
(this module holds all the logic). Site assets go to
``<repo>/public/models/<arch>/`` (shipped in the bundle); goldens go to
``<repo>/goldens/<arch>/`` (test fixtures only, read from fs by vitest —
kept OUT of public/ so ~4MB of JSON never ships with the site):

  public/models/<arch>/manifest.json
                  ``arch``, ``dims`` (arch config + vocab_size + derived
                  dims), ``checkpoint`` provenance, and the ``tensors``
                  table. Offsets and lengths are in FLOAT-COUNT units
                  (float32 elements, NOT bytes); the manifest documents
                  this itself via ``"offset_unit": "float32"``.
  public/models/<arch>/weights.bin
                  every state_dict tensor as little-endian float32,
                  concatenated in manifest order (sorted by tensor name).
  goldens/<arch>/goldens.json
                  2 fixed eval inputs plus the full recorded activation
                  dict per input. Non-finite floats (GPT attn.scores hold
                  -inf from the causal mask) are encoded as the strings
                  "Infinity" / "-Infinity" / "NaN" because strict JSON
                  forbids those literals; every finite value is a plain
                  JSON number at full repr precision (no rounding).

Seed streams (single config seed, documented contract): train data =
``seed + 1``, fixed val set = ``seed + 2``, export goldens = ``seed + 3``.
Golden candidates are drawn ONE AT A TIME from the export stream and a
candidate is kept only if the checkpointed model predicts all answer
tokens correctly; the number skipped before collecting 2 keepers is
recorded in the manifest checkpoint section as
``"golden_candidates_skipped"`` (expected 0 at val_exact ~1.0).

Checkpoint selection is strictly via ``training/runs.json`` (best.pt of
the recorded run_dir). Missing runs.json / entry / checkpoint fails
loudly; nothing is ever regenerated here.
"""

import json
import math
from pathlib import Path

import torch

from llmviz_train.task import TASK, make_batch
from llmviz_train.train import build_model

TRAINING_DIR = Path(__file__).resolve().parent.parent
DEFAULT_MODELS_ROOT = TRAINING_DIR.parent / "public" / "models"
DEFAULT_GOLDENS_ROOT = TRAINING_DIR.parent / "goldens"

GOLDEN_SEED_OFFSET = 3  # train=+1, val=+2, export goldens=+3
N_GOLDEN_INPUTS = 2
MAX_GOLDEN_CANDIDATES = 64  # loud failure beyond this, never silent retry


# --- checkpoint loading (strictly via runs.json) ----------------------------


def load_checkpoint(arch: str, training_dir: Path | str = TRAINING_DIR) -> dict:
    """Load best.pt for ``arch`` as recorded in runs.json; fail loudly if absent.

    Returns the checkpoint dict (arch, step, val_exact, model_state_dict,
    config) with ``run_dir`` (the runs.json-relative path string) added.
    """
    training_dir = Path(training_dir)
    runs_path = training_dir / "runs.json"
    if not runs_path.exists():
        raise FileNotFoundError(f"runs.json not found at {runs_path} — cannot select checkpoint")
    runs = json.loads(runs_path.read_text())
    if arch not in runs:
        raise KeyError(f"no '{arch}' entry in {runs_path} (have: {sorted(runs)})")
    run_dir = runs[arch]["run_dir"]
    ckpt_path = training_dir / run_dir / "best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"checkpoint missing: {ckpt_path} (runs.json points at run_dir={run_dir!r}); "
            "refusing to regenerate — restore the run or retrain explicitly"
        )
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    ckpt["run_dir"] = run_dir
    return ckpt


# --- manifest + weights.bin --------------------------------------------------


def build_dims(arch: str, cfg: dict, state_dict: dict) -> dict:
    """Arch config dict + vocab_size + derived dims the TS side needs."""
    dims = dict(cfg[arch])
    dims["vocab_size"] = cfg["task"]["vocab_size"]
    if arch == "mamba":
        dims["dt_rank"] = math.ceil(dims["d_model"] / 16)
    elif arch == "gpt":
        dims["head_dim"] = dims["d_model"] // dims["n_head"]
        dims["max_seq_len"] = int(state_dict["pos_embedding.weight"].shape[0])
    elif arch == "rwkv":
        dims["d_ffn"] = dims["ffn_mult"] * dims["d_model"]
    elif arch in ("moe", "kan"):
        dims["head_dim"] = dims["d_model"] // dims["n_head"]
        dims["max_seq_len"] = int(state_dict["pos_embedding.weight"].shape[0])
    else:
        raise ValueError(f"unknown arch {arch!r}")
    return dims


def build_tensor_table(state_dict: dict) -> tuple[list[dict], bytes]:
    """Sorted-by-name tensor table (float-count offsets) + concatenated LE f32 blob."""
    table: list[dict] = []
    chunks: list[bytes] = []
    offset = 0
    for name in sorted(state_dict):
        arr = state_dict[name].detach().to(torch.float32).numpy().astype("<f4")
        table.append(
            {
                "name": name,
                "shape": list(arr.shape),
                "offset": offset,
                "length": int(arr.size),
            }
        )
        chunks.append(arr.tobytes())
        offset += int(arr.size)
    return table, b"".join(chunks)


# --- non-finite-safe value encoding ------------------------------------------

_DECODE_SPECIAL = {
    "Infinity": float("inf"),
    "-Infinity": float("-inf"),
    "NaN": float("nan"),
}


def encode_value(v: float) -> float | str:
    """Finite floats pass through; non-finite become their JSON-safe string."""
    if math.isfinite(v):
        return v
    if math.isnan(v):
        return "NaN"
    return "Infinity" if v > 0 else "-Infinity"


def decode_value(v: float | int | str) -> float:
    """Inverse of encode_value (unknown strings raise KeyError, loudly)."""
    if isinstance(v, str):
        return _DECODE_SPECIAL[v]
    return float(v)


def encode_data(tensor: torch.Tensor) -> list[float | str]:
    """Flatten (row-major) and encode every element."""
    return [encode_value(v) for v in tensor.reshape(-1).tolist()]


def decode_data(data: list[float | int | str]) -> list[float]:
    return [decode_value(v) for v in data]


# --- golden inputs ------------------------------------------------------------


def select_golden_inputs(
    model: torch.nn.Module, seed: int
) -> tuple[list[dict], list[dict], int]:
    """Draw candidates from the export stream (seed+3) until 2 are predicted
    correctly; return (inputs, activations, n_skipped). Deterministic: one
    candidate per make_batch(1, gen) call, kept in draw order.
    """
    gen = torch.Generator().manual_seed(seed + GOLDEN_SEED_OFFSET)
    lo, hi = TASK.context_len, TASK.context_len + TASK.n_data
    inputs: list[dict] = []
    activations: list[dict] = []
    skipped = 0

    model.eval()
    with torch.no_grad():
        for _ in range(MAX_GOLDEN_CANDIDATES):
            x, y, _mask = make_batch(1, gen)
            logits, acts = model(x, record=True)
            answer = y[0, lo:hi]
            preds = logits[0, lo:hi].argmax(dim=-1)
            if not torch.equal(preds, answer):
                skipped += 1
                continue
            inputs.append({"tokens": x[0].tolist(), "answer": answer.tolist()})
            activations.append(
                {
                    name: {"shape": list(t.shape), "data": encode_data(t)}
                    for name, t in acts.items()
                }
            )
            if len(inputs) == N_GOLDEN_INPUTS:
                return inputs, activations, skipped

    raise RuntimeError(
        f"could not find {N_GOLDEN_INPUTS} correctly-predicted golden inputs in "
        f"{MAX_GOLDEN_CANDIDATES} candidates (got {len(inputs)}, skipped {skipped}) — "
        "the checkpoint is far below its recorded val_exact; investigate, don't retry"
    )


# --- top-level export ----------------------------------------------------------


def export_arch(
    arch: str,
    models_dir: Path | str | None = None,
    goldens_dir: Path | str | None = None,
    training_dir: Path | str = TRAINING_DIR,
) -> dict:
    """Export manifest.json + weights.bin (``models_dir``, defaults to
    ``<repo>/public/models/<arch>``) and goldens.json (``goldens_dir``,
    defaults to ``<repo>/goldens/<arch>``) for ``arch``.

    Returns a summary dict (paths, sizes, tensor/float counts, skip count).
    """
    models_dir = Path(models_dir) if models_dir is not None else DEFAULT_MODELS_ROOT / arch
    goldens_dir = Path(goldens_dir) if goldens_dir is not None else DEFAULT_GOLDENS_ROOT / arch
    ckpt = load_checkpoint(arch, training_dir)
    cfg = ckpt["config"]
    state_dict = ckpt["model_state_dict"]

    model = build_model(arch, cfg)
    model.load_state_dict(state_dict)

    inputs, activations, skipped = select_golden_inputs(model, cfg["seed"])
    table, blob = build_tensor_table(state_dict)

    manifest = {
        "arch": arch,
        "offset_unit": "float32",  # offsets/lengths are float32-element counts, not bytes
        "dims": build_dims(arch, cfg, state_dict),
        "checkpoint": {
            "run_dir": ckpt["run_dir"],
            "step": int(ckpt["step"]),
            "val_exact": float(ckpt["val_exact"]),
            "golden_seed": cfg["seed"] + GOLDEN_SEED_OFFSET,
            "golden_candidates_skipped": skipped,
        },
        "tensors": table,
    }
    goldens = {"inputs": inputs, "activations": activations}

    models_dir.mkdir(parents=True, exist_ok=True)
    goldens_dir.mkdir(parents=True, exist_ok=True)
    (models_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (models_dir / "weights.bin").write_bytes(blob)
    # allow_nan=False guarantees no bare Infinity/NaN literal ever reaches disk;
    # full-precision default float repr, no rounding.
    (goldens_dir / "goldens.json").write_text(json.dumps(goldens, allow_nan=False) + "\n")

    return {
        "arch": arch,
        "models_dir": str(models_dir),
        "goldens_dir": str(goldens_dir),
        "n_tensors": len(table),
        "n_floats": sum(t["length"] for t in table),
        "golden_candidates_skipped": skipped,
        "bytes": {
            "manifest.json": (models_dir / "manifest.json").stat().st_size,
            "weights.bin": len(blob),
            "goldens.json": (goldens_dir / "goldens.json").stat().st_size,
        },
    }
