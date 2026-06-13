"""Training loop for the masked text-diffusion denoiser.

Diffusion diverges from the shared autoregressive loop in two places, so it
gets its own loop (the rest of train.py is reused): (1) each batch is
CORRUPTED — every token is replaced by [MASK] with a per-example rate
~U(0,1) — and the loss is cross-entropy on the masked positions only;
(2) "accuracy" is the denoising test — mask the four answer positions, run
the bidirectional denoiser once, and check it recovers the data tokens.

Same seeding contract and NaN/Inf policy as train.py.
"""

import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from llmviz_train.task import TASK, make_batch
from llmviz_train.seed import seed_all
from llmviz_train.train import (
    DEVICE,
    EARLY_STOP_PATIENCE,
    EARLY_STOP_THRESHOLD,
    _die_nonfinite,
    _grad_norm,
    atomic_save,
    build_model,
)

# Answer tokens live at positions L+1 .. L+n (the teacher-forced tail).
ANS_LO = TASK.context_len + 1
ANS_HI = TASK.context_len + 1 + TASK.n_data


def mask_tokens(
    x: torch.Tensor, gen: torch.Generator, mask_id: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Corrupt x: per-example rate ~U(0,1), each position masked iid."""
    b, t = x.shape
    rate = torch.rand(b, 1, generator=gen)
    m = torch.rand(b, t, generator=gen) < rate
    x_masked = torch.where(m, torch.full_like(x, mask_id), x)
    return x_masked, m


@torch.no_grad()
def evaluate_diffusion(model, x: torch.Tensor, mask_id: int, chunk: int = 512) -> float:
    """Exact-match recovery of the answer tail when it is fully masked."""
    model.eval()
    exact = 0
    for i in range(0, x.shape[0], chunk):
        xb = x[i : i + chunk].clone()
        target = xb[:, ANS_LO:ANS_HI].clone()
        xb[:, ANS_LO:ANS_HI] = mask_id
        preds = model(xb).argmax(dim=-1)[:, ANS_LO:ANS_HI]
        exact += int((preds == target).all(dim=1).sum())
    model.train()
    return exact / x.shape[0]


def train_diffusion(
    cfg: dict,
    *,
    eval_every: int = 250,
    out_dir: Path | str | None = None,
    log_every: int = 250,
) -> dict:
    """Train the diffusion denoiser; return a summary dict like train_loop."""
    tcfg = cfg["train"]
    steps = tcfg["steps"]
    batch_size = tcfg["batch_size"]
    lr = tcfg["lr"]
    seed = cfg["seed"]
    mask_id = cfg["diffusion"]["mask_id"]
    out_dir = Path(out_dir) if out_dir is not None else None

    train_gen = seed_all(seed)
    model = build_model("diffusion", cfg).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    # Two RNG streams as in train.py: train data seed+1 (returned), val seed+2.
    val_gen = torch.Generator().manual_seed(seed + 2)
    val_x, _vy, _vm = make_batch(tcfg["val_size"], val_gen)
    # Masking draws from a third stream so it never aliases the data streams.
    mask_gen = torch.Generator().manual_seed(seed + 7)

    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    warmup = min(200, max(1, steps // 5))

    def lr_lambda(s: int) -> float:
        if s < warmup:
            return (s + 1) / warmup
        return 0.5 * (1.0 + math.cos(math.pi * (s - warmup) / max(1, steps - warmup)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    print(f"[diffusion] {n_params} params | device={DEVICE} | steps<={steps} batch={batch_size}", flush=True)

    metrics_path = out_dir / "metrics.jsonl" if out_dir is not None else None
    start = time.monotonic()
    evals: list[dict] = []
    best_val_exact = -1.0
    best_step = -1
    streak = 0
    final_step = 0

    model.train()
    for step in range(1, steps + 1):
        x, _y, _m = make_batch(batch_size, train_gen)
        x_masked, loss_mask = mask_tokens(x, mask_gen, mask_id)
        logits = model(x_masked)
        loss = F.cross_entropy(logits[loss_mask], x[loss_mask])
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
        final_step = step

        if eval_every and step % eval_every == 0:
            val_exact = evaluate_diffusion(model, val_x, mask_id)
            elapsed = time.monotonic() - start
            rec = {
                "step": step,
                "train_loss": loss_val,
                "val_exact": val_exact,
                "lr": opt.param_groups[0]["lr"],
                "elapsed_s": round(elapsed, 2),
            }
            evals.append(rec)
            if step % max(log_every, eval_every) == 0:
                print(f"[diffusion] step {step:>6} loss {loss_val:.4f} val_exact {val_exact:.4f} {elapsed:.1f}s", flush=True)
            if metrics_path is not None:
                with open(metrics_path, "a") as f:
                    f.write(json.dumps(rec) + "\n")
            if out_dir is not None:
                ckpt = {"arch": "diffusion", "step": step, "val_exact": val_exact, "model_state_dict": model.state_dict(), "config": cfg}
                atomic_save(ckpt, out_dir / "latest.pt")
                if val_exact > best_val_exact:
                    atomic_save(ckpt, out_dir / "best.pt")
            if val_exact > best_val_exact:
                best_val_exact = val_exact
                best_step = step
            if val_exact >= EARLY_STOP_THRESHOLD:
                streak += 1
                if streak >= EARLY_STOP_PATIENCE:
                    print(f"[diffusion] early stop at step {step}", flush=True)
                    break
            else:
                streak = 0

    wall_s = time.monotonic() - start
    print(f"[diffusion] done: step {final_step}, best val_exact {best_val_exact:.4f} @ {best_step}, wall {wall_s:.1f}s", flush=True)
    return {
        "arch": "diffusion",
        "device": DEVICE,
        "n_params": n_params,
        "evals": evals,
        "final_step": final_step,
        "best_val_exact": best_val_exact,
        "best_step": best_step,
        "wall_s": wall_s,
    }
