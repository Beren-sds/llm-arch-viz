"""Minimal Mamba reference implementation with activation recording.

This module is the golden source for the TypeScript port: every named
intermediate recorded here is compared against the TS re-implementation
at 1e-4 tolerance. Clarity therefore beats speed throughout — the
selective scan is a plain Python loop over time, one step per token.

Architecture (pre-norm residual stack):

    tokens -> embedding -> [x = x + block_i(rmsnorm_i(x))] * n_layer
           -> final_norm -> lm_head (untied) -> logits

MambaBlock (d_inner = expand * d_model):

    in_proj -> split (x_part, z)
    x_part -> depthwise causal conv1d -> SiLU
           -> x_proj -> split (dt, B, C)
           -> delta = softplus(dt_proj(dt))
           -> selective scan with A = -exp(A_log), D skip
    out = out_proj(y * silu(z))
"""

import math

import torch
import torch.nn.functional as F
from torch import nn


def _snapshot(tensor: torch.Tensor) -> torch.Tensor:
    """Detached float32 CPU copy of ``tensor`` for activation recording.

    The trailing clone matters: on a CPU float32 model, ``.to`` is a no-op
    that would alias live forward-pass storage, so later in-place ops could
    silently corrupt the snapshot.
    """
    return tensor.detach().to("cpu", torch.float32).clone()


class RMSNorm(nn.Module):
    """x * rsqrt(mean(x^2, dim=-1) + eps) * weight (no bias)."""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps) * self.weight


class MambaBlock(nn.Module):
    """One selective-state-space block (no residual; the Mamba module adds it)."""

    def __init__(self, d_model: int, d_state: int, d_conv: int, expand: int):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.d_inner = expand * d_model
        self.dt_rank = math.ceil(d_model / 16)

        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)
        self.conv1d = nn.Conv1d(
            self.d_inner,
            self.d_inner,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            bias=True,
        )
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        # Standard Mamba dt bias init: dt ~ exp(U(log 1e-3, log 1e-1)),
        # bias = inverse-softplus(dt) = dt + log(-expm1(-dt)).
        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(1e-1) - math.log(1e-3)) + math.log(1e-3)
        ).clamp(min=1e-4)
        with torch.no_grad():
            self.dt_proj.bias.copy_(dt + torch.log(-torch.expm1(-dt)))

        # A_log init: each of the d_inner rows is log(1, 2, ..., d_state).
        a = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(a))
        self.D = nn.Parameter(torch.ones(self.d_inner))

        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(
        self, x: torch.Tensor, record: dict[str, torch.Tensor] | None = None, prefix: str = ""
    ) -> torch.Tensor:
        """x: (B, T, d_model) -> (B, T, d_model).

        When ``record`` is a dict, named intermediates for batch item 0 are
        stored into it (detached float32 CPU tensors) under ``prefix``.
        """
        B_, T, _ = x.shape

        def rec(name: str, tensor: torch.Tensor) -> None:
            if record is not None:
                record[prefix + name] = _snapshot(tensor)

        xz = self.in_proj(x)  # (B, T, 2*d_inner)
        rec("in_proj.out", xz[0])
        x_part, z = xz.chunk(2, dim=-1)  # each (B, T, d_inner)

        # Depthwise causal conv along time; padding=d_conv-1 makes the output
        # longer than T, so truncate back to the first T steps (causal).
        x_conv = self.conv1d(x_part.transpose(1, 2))[:, :, :T].transpose(1, 2)
        x_conv = F.silu(x_conv)  # (B, T, d_inner)
        rec("conv.out", x_conv[0])

        dbc = self.x_proj(x_conv)  # (B, T, dt_rank + 2*d_state)
        rec("x_proj.out", dbc[0])
        dt, B_mat, C_mat = torch.split(
            dbc, [self.dt_rank, self.d_state, self.d_state], dim=-1
        )
        delta = F.softplus(self.dt_proj(dt))  # (B, T, d_inner)
        rec("delta.out", delta[0])

        A = -torch.exp(self.A_log)  # (d_inner, d_state)

        # Selective scan, one token at a time.
        h = x.new_zeros(B_, self.d_inner, self.d_state)
        ys = []
        for t in range(T):
            delta_t = delta[:, t]  # (B, d_inner)
            B_t = B_mat[:, t]  # (B, d_state)
            C_t = C_mat[:, t]  # (B, d_state)
            x_t = x_conv[:, t]  # (B, d_inner)
            h = (
                torch.exp(delta_t[..., None] * A) * h
                + delta_t[..., None] * B_t[:, None, :] * x_t[..., None]
            )
            rec(f"ssm.h.t{t}", h[0])
            y_t = torch.einsum("bds,bs->bd", h, C_t) + self.D * x_t
            ys.append(y_t)
        y = torch.stack(ys, dim=1)  # (B, T, d_inner)
        rec("ssm.out", y[0])

        gated = y * F.silu(z)
        rec("gate.out", gated[0])

        out = self.out_proj(gated)
        rec("out_proj.out", out[0])
        return out


class Mamba(nn.Module):
    """Minimal Mamba language model over a small token vocabulary."""

    def __init__(self, cfg: dict, vocab_size: int):
        super().__init__()
        n_layer = cfg["n_layer"]
        d_model = cfg["d_model"]

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.norms = nn.ModuleList(RMSNorm(d_model) for _ in range(n_layer))
        self.blocks = nn.ModuleList(
            MambaBlock(d_model, cfg["d_state"], cfg["d_conv"], cfg["expand"])
            for _ in range(n_layer)
        )
        self.final_norm = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)  # untied

    def forward(
        self, tokens: torch.Tensor, record: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """tokens: (B, T) long -> logits (B, T, vocab).

        With ``record=True``, also returns a dict of named activations for
        batch item 0 (detached float32 CPU tensors).
        """
        acts: dict[str, torch.Tensor] | None = {} if record else None

        def rec(name: str, tensor: torch.Tensor) -> None:
            if acts is not None:
                acts[name] = _snapshot(tensor)

        x = self.embedding(tokens)
        rec("embed.out", x[0])

        for i, (norm, block) in enumerate(zip(self.norms, self.blocks)):
            normed = norm(x)
            rec(f"layer{i}.norm.out", normed[0])
            x = x + block(normed, record=acts, prefix=f"layer{i}.")

        x = self.final_norm(x)
        rec("final_norm.out", x[0])

        logits = self.lm_head(x)
        rec("head.logits", logits[0])

        if acts is not None:
            return logits, acts
        return logits
