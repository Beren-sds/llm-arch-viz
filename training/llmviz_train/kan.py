"""Minimal KAN (Kolmogorov–Arnold) reference with activation recording.

A GPT whose feed-forward is a KAN instead of an MLP. The distinctive KAN
idea is a learnable univariate function on every edge; classic KANs use
B-splines, but those are awkward to port exactly, so this uses the FastKAN
radial-basis form (Li 2024): each edge function is a weighted sum of fixed
Gaussian bumps on a grid, which is a plain `exp` and is trivial to mirror
in TS at 1e-4.

KAN feed-forward over a pre-LayerNorm'd input x̂ (d_model -> d_model):
    rbf  = exp(-((x̂[...,None] - grid) / denom)^2)     # (T, d_model, G)
    out  = spline_linear(rbf.flatten) + base_linear(silu(x̂))

Attention is exactly the GPT CausalSelfAttention (imported, not re-derived).
"""

import torch
import torch.nn.functional as F
from torch import nn

from llmviz_train.gpt import CausalSelfAttention
from llmviz_train.recording import snapshot


class KANLayer(nn.Module):
    """FastKAN feed-forward (no residual; the model adds it). d_in == d_out."""

    def __init__(self, d_model: int, num_grids: int, grid_min: float, grid_max: float):
        super().__init__()
        self.d_model = d_model
        self.num_grids = num_grids
        grid = torch.linspace(grid_min, grid_max, num_grids)
        self.register_buffer("grid", grid, persistent=False)  # fixed, not exported
        self.denom = (grid_max - grid_min) / (num_grids - 1)
        self.spline_linear = nn.Linear(d_model * num_grids, d_model, bias=False)
        self.base_linear = nn.Linear(d_model, d_model, bias=True)

    def forward(
        self, x: torch.Tensor, record: dict[str, torch.Tensor] | None = None, prefix: str = ""
    ) -> torch.Tensor:
        """x: (B, T, d_model) (already LayerNorm'd) -> (B, T, d_model)."""

        def rec(name: str, tensor: torch.Tensor) -> None:
            if record is not None:
                record[prefix + name] = snapshot(tensor)

        # Gaussian basis on each scalar input: (B, T, d_model, G)
        rbf = torch.exp(-(((x[..., None] - self.grid) / self.denom) ** 2))
        flat = rbf.reshape(*x.shape[:-1], self.d_model * self.num_grids)
        rec("rbf", flat[0])  # (T, d_model*G) — the learnable edge functions, evaluated
        spline = self.spline_linear(flat)
        rec("spline", spline[0])
        base = self.base_linear(F.silu(x))
        rec("base", base[0])
        out = spline + base
        rec("out", out[0])
        return out


class KAN(nn.Module):
    """Minimal KAN-GPT language model over a small token vocabulary."""

    def __init__(self, cfg: dict, vocab_size: int, max_seq_len: int = 32):
        super().__init__()
        n_layer = cfg["n_layer"]
        d_model = cfg["d_model"]
        self.max_seq_len = max_seq_len

        self.tok_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)
        self.ln1s = nn.ModuleList(nn.LayerNorm(d_model) for _ in range(n_layer))
        self.attns = nn.ModuleList(CausalSelfAttention(d_model, cfg["n_head"]) for _ in range(n_layer))
        self.ln2s = nn.ModuleList(nn.LayerNorm(d_model) for _ in range(n_layer))
        self.kans = nn.ModuleList(
            KANLayer(d_model, cfg["num_grids"], cfg["grid_min"], cfg["grid_max"])
            for _ in range(n_layer)
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)  # untied

    def forward(
        self, tokens: torch.Tensor, record: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """tokens: (B, T) long -> logits (B, T, vocab)."""
        T = tokens.shape[1]
        assert T <= self.max_seq_len, f"T={T} exceeds max_seq_len={self.max_seq_len}"
        acts: dict[str, torch.Tensor] | None = {} if record else None

        def rec(name: str, tensor: torch.Tensor) -> None:
            if acts is not None:
                acts[name] = snapshot(tensor)

        positions = torch.arange(T, device=tokens.device)
        x = self.tok_embedding(tokens) + self.pos_embedding(positions)
        rec("embed.out", x[0])

        for i in range(len(self.attns)):
            normed = self.ln1s[i](x)
            rec(f"layer{i}.ln1.out", normed[0])
            x = x + self.attns[i](normed, record=acts, prefix=f"layer{i}.attn.")
            normed = self.ln2s[i](x)
            rec(f"layer{i}.ln2.out", normed[0])
            x = x + self.kans[i](normed, record=acts, prefix=f"layer{i}.kan.")

        x = self.ln_f(x)
        rec("final_norm.out", x[0])

        logits = self.lm_head(x)
        rec("head.logits", logits[0])

        if acts is not None:
            return logits, acts
        return logits
