"""Training loop for the selective-copying toy models (Mamba / GPT).

Seeding contract (single source of randomness, per repo standards):
  - ``seed_all(cfg["seed"])`` is called exactly once at the start of
    ``train_loop``. It seeds the global RNGs (used for weight init) and
    returns the train-stream data generator, seeded ``seed + 1``.
  - The fixed validation set is drawn ONCE from a dedicated generator
    seeded ``seed + 2``. Train stream = seed+1, val stream = seed+2:
    the two streams cannot overlap.

Device: CPU, always. The Mamba selective scan is a plain Python loop over
time (one op dispatch per token), so MPS kernel-launch overhead dominates
at this scale; CPU is both faster here and deterministic.

LR schedule: linear warmup then cosine decay to 0 over the configured
horizon. Warmup is 200 steps, capped at ``steps // 5`` so that short runs
(e.g. the 30-step smoke test) still reach full LR — cheap insurance for
AdamW second-moment estimates, documented per plan.

Early stopping: val exact-match >= 0.995 at two consecutive evals
(target 0.99 with margin), capped at the configured step budget.

NaN/Inf policy: a non-finite loss or gradient norm stops the run loudly
via SystemExit(1) — no nan_to_num, no silent recovery.
"""

import json
import math
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from llmviz_train.diffusion import Diffusion
from llmviz_train.gpt import GPT
from llmviz_train.kan import KAN
from llmviz_train.mamba import Mamba
from llmviz_train.moe import MoEModel
from llmviz_train.rwkv import RWKV
from llmviz_train.seed import seed_all
from llmviz_train.task import make_batch

DEVICE = "cpu"  # see module docstring
EARLY_STOP_THRESHOLD = 0.995
EARLY_STOP_PATIENCE = 2  # consecutive evals at/above threshold


def build_model(arch: str, cfg: dict) -> nn.Module:
    """Construct the requested architecture from the resolved config."""
    vocab_size = cfg["task"]["vocab_size"]
    if arch == "mamba":
        return Mamba(cfg["mamba"], vocab_size=vocab_size)
    if arch == "gpt":
        return GPT(cfg["gpt"], vocab_size=vocab_size)
    if arch == "rwkv":
        return RWKV(cfg["rwkv"], vocab_size=vocab_size)
    if arch == "moe":
        return MoEModel(cfg["moe"], vocab_size=vocab_size)
    if arch == "kan":
        return KAN(cfg["kan"], vocab_size=vocab_size)
    if arch == "diffusion":
        return Diffusion(cfg["diffusion"], vocab_size=vocab_size)
    raise ValueError(f"unknown arch {arch!r}")


def atomic_save(obj: object, path: Path) -> None:
    """torch.save via tmp file + os.replace, so readers never see a partial file."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    mask: torch.Tensor,
    chunk: int = 512,
) -> tuple[float, float]:
    """Exact-match and per-token accuracy on a fixed validation set.

    Exact match: all n_data answer tokens correct for an example -> 1, else 0.
    Per-token: fraction of individual answer tokens correct (diagnostic).
    """
    model.eval()
    exact_hits = 0
    token_hits = 0
    token_total = 0
    for i in range(0, x.shape[0], chunk):
        xb, yb, mb = x[i : i + chunk], y[i : i + chunk], mask[i : i + chunk]
        preds = model(xb).argmax(dim=-1)
        n_ans = int(mb[0].sum())  # same answer positions for every example
        correct = (preds[mb] == yb[mb]).view(xb.shape[0], n_ans)
        exact_hits += int(correct.all(dim=1).sum())
        token_hits += int(correct.sum())
        token_total += correct.numel()
    model.train()
    return exact_hits / x.shape[0], token_hits / token_total


def _grad_norm(model: nn.Module) -> float:
    """Global L2 grad norm, computed without mutating any gradient."""
    with torch.no_grad():
        total = torch.zeros(())
        for p in model.parameters():
            if p.grad is not None:
                total += p.grad.float().pow(2).sum()
        return float(total.sqrt())


def _die_nonfinite(what: str, value: float, step: int) -> None:
    print(
        f"\n!!! NON-FINITE {what} ({value}) at step {step} — aborting run. "
        "No nan_to_num, no recovery; inspect the run before retrying.",
        flush=True,
    )
    raise SystemExit(1)


def train_loop(
    arch: str,
    cfg: dict,
    *,
    steps: int | None = None,
    batch_size: int | None = None,
    eval_every: int = 250,
    out_dir: Path | str | None = None,
    log_every: int = 250,
) -> dict:
    """Train ``arch`` on selective copying; return a summary dict.

    With ``out_dir`` set, writes best.pt / latest.pt (atomic) and appends one
    JSON line per eval to metrics.jsonl. With ``out_dir=None`` (tests),
    nothing touches disk. ``eval_every=0`` disables evaluation entirely.
    """
    tcfg = cfg["train"]
    steps = steps if steps is not None else tcfg["steps"]
    batch_size = batch_size if batch_size is not None else tcfg["batch_size"]
    lr = tcfg["lr"]
    val_size = tcfg["val_size"]
    seed = cfg["seed"]
    out_dir = Path(out_dir) if out_dir is not None else None

    # --- the single seeding point of the run (init RNG + train stream) ---
    train_gen = seed_all(seed)
    model = build_model(arch, cfg).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())

    # Fixed val set: dedicated stream, seed+2 (train stream is seed+1).
    val_gen = torch.Generator().manual_seed(seed + 2)
    val_x, val_y, val_mask = make_batch(val_size, val_gen)

    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    warmup = min(200, max(1, steps // 5))

    def lr_lambda(step_idx: int) -> float:
        if step_idx < warmup:
            return (step_idx + 1) / warmup
        progress = (step_idx - warmup) / max(1, steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    print(
        f"[{arch}] {n_params} params | device={DEVICE} | steps<={steps} "
        f"batch={batch_size} lr={lr} warmup={warmup} | val_size={val_size} "
        f"(train stream seed+1={seed + 1}, val stream seed+2={seed + 2})",
        flush=True,
    )

    metrics_path = out_dir / "metrics.jsonl" if out_dir is not None else None
    start = time.monotonic()
    step_losses: list[float] = []
    evals: list[dict] = []
    best_val_exact = -1.0
    best_step = -1
    above_threshold_streak = 0
    stopped_early = False
    final_step = 0

    model.train()
    for step in range(1, steps + 1):
        x, y, mask = make_batch(batch_size, train_gen)
        logits = model(x)
        loss = F.cross_entropy(logits[mask], y[mask])
        loss_val = loss.detach().item()
        if not math.isfinite(loss_val):
            _die_nonfinite("loss", loss_val, step)

        opt.zero_grad()
        loss.backward()
        gnorm = _grad_norm(model)
        if not math.isfinite(gnorm):
            _die_nonfinite("grad norm", gnorm, step)
        opt.step()
        sched.step()
        step_losses.append(loss_val)
        final_step = step

        if eval_every and step % eval_every == 0:
            val_exact, val_per_token = evaluate(model, val_x, val_y, val_mask)
            elapsed = time.monotonic() - start
            current_lr = opt.param_groups[0]["lr"]
            record = {
                "step": step,
                "train_loss": loss_val,
                "val_exact": val_exact,
                "val_per_token": val_per_token,
                "lr": current_lr,
                "elapsed_s": round(elapsed, 2),
            }
            evals.append(record)
            if step % max(log_every, eval_every) == 0:
                print(
                    f"[{arch}] step {step:>6} loss {loss_val:.4f} "
                    f"val_exact {val_exact:.4f} val_tok {val_per_token:.4f} "
                    f"lr {current_lr:.2e} {elapsed:.1f}s",
                    flush=True,
                )
            if metrics_path is not None:
                with open(metrics_path, "a") as f:
                    f.write(json.dumps(record) + "\n")

            if out_dir is not None:
                ckpt = {
                    "arch": arch,
                    "step": step,
                    "val_exact": val_exact,
                    "val_per_token": val_per_token,
                    "model_state_dict": model.state_dict(),
                    "config": cfg,
                }
                atomic_save(ckpt, out_dir / "latest.pt")
                if val_exact > best_val_exact:
                    atomic_save(ckpt, out_dir / "best.pt")
            if val_exact > best_val_exact:
                best_val_exact = val_exact
                best_step = step

            if val_exact >= EARLY_STOP_THRESHOLD:
                above_threshold_streak += 1
                if above_threshold_streak >= EARLY_STOP_PATIENCE:
                    stopped_early = True
                    print(
                        f"[{arch}] early stop at step {step}: val_exact >= "
                        f"{EARLY_STOP_THRESHOLD} for {EARLY_STOP_PATIENCE} "
                        "consecutive evals",
                        flush=True,
                    )
                    break
            else:
                above_threshold_streak = 0

    wall_s = time.monotonic() - start
    print(
        f"[{arch}] done: step {final_step}, best val_exact {best_val_exact:.4f} "
        f"@ step {best_step}, wall {wall_s:.1f}s",
        flush=True,
    )
    return {
        "arch": arch,
        "device": DEVICE,
        "n_params": n_params,
        "step_losses": step_losses,
        "evals": evals,
        "final_step": final_step,
        "best_val_exact": best_val_exact,
        "best_step": best_step,
        "stopped_early": stopped_early,
        "wall_s": wall_s,
    }
