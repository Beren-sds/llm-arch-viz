"""Minimal GPT reference implementation with activation recording.

This module is the golden source for the TypeScript port: every named
intermediate recorded here is compared against the TS re-implementation
at 1e-4 tolerance. Clarity therefore beats speed throughout — attention
is written out as explicit matmuls, no fused kernels.

Architecture (pre-norm residual stack):

    tokens -> tok_embedding + pos_embedding
           -> [x = x + attn_i(ln1_i(x)); x = x + mlp_i(ln2_i(x))] * n_layer
           -> ln_f -> lm_head (untied) -> logits

Norms are standard LayerNorm (eps 1e-5, with bias) — deliberately not
RMSNorm, because that is what GPTs use; the Mamba model uses RMSNorm.

CausalSelfAttention (head_dim = d_model / n_head):

    qkv_proj -> split (q, k, v), reshape to (B, n_head, T, head_dim)
    scores = q @ k^T / sqrt(head_dim), upper triangle masked to -inf
    weights = softmax(scores); out = out_proj(weights @ v)

MLP: fc (d_model -> mlp_ratio*d_model) -> GELU (exact erf) -> proj.
"""

import math

import torch
import torch.nn.functional as F
from torch import nn

from llmviz_train.mamba import _snapshot


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention (no residual; the block adds it)."""

    def __init__(self, d_model: int, n_head: int):
        super().__init__()
        assert d_model % n_head == 0
        self.n_head = n_head
        self.head_dim = d_model // n_head

        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=True)
        self.out_proj = nn.Linear(d_model, d_model, bias=True)

    def forward(
        self, x: torch.Tensor, record: dict[str, torch.Tensor] | None = None, prefix: str = ""
    ) -> torch.Tensor:
        """x: (B, T, d_model) -> (B, T, d_model).

        When ``record`` is a dict, named intermediates for batch item 0 are
        stored into it (detached cloned float32 CPU tensors) under ``prefix``.
        """
        B, T, d_model = x.shape

        def rec(name: str, tensor: torch.Tensor) -> None:
            if record is not None:
                record[prefix + name] = _snapshot(tensor)

        qkv = self.qkv_proj(x)  # (B, T, 3*d_model)
        q, k, v = qkv.chunk(3, dim=-1)  # each (B, T, d_model)
        # (B, T, d_model) -> (B, n_head, T, head_dim)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        rec("q", q[0])
        rec("k", k[0])
        rec("v", v[0])

        scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)  # (B, n_head, T, T)
        causal_mask = torch.triu(
            torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1
        )
        scores = scores.masked_fill(causal_mask, float("-inf"))
        rec("scores", scores[0])  # post-mask, pre-softmax

        weights = F.softmax(scores, dim=-1)  # (B, n_head, T, T)
        rec("weights", weights[0])

        out = weights @ v  # (B, n_head, T, head_dim)
        out = out.transpose(1, 2).reshape(B, T, d_model)
        out = self.out_proj(out)
        rec("out", out[0])
        return out


class MLP(nn.Module):
    """fc -> GELU (exact erf, nn.GELU default) -> proj."""

    def __init__(self, d_model: int, mlp_ratio: int):
        super().__init__()
        d_mlp = mlp_ratio * d_model
        self.fc = nn.Linear(d_model, d_mlp, bias=True)
        self.act = nn.GELU()
        self.proj = nn.Linear(d_mlp, d_model, bias=True)

    def forward(
        self, x: torch.Tensor, record: dict[str, torch.Tensor] | None = None, prefix: str = ""
    ) -> torch.Tensor:
        """x: (B, T, d_model) -> (B, T, d_model), recording like the attn."""

        def rec(name: str, tensor: torch.Tensor) -> None:
            if record is not None:
                record[prefix + name] = _snapshot(tensor)

        pre = self.fc(x)  # (B, T, d_mlp)
        rec("fc", pre[0])
        post = self.act(pre)
        rec("act", post[0])
        out = self.proj(post)
        rec("proj", out[0])
        return out


class GPT(nn.Module):
    """Minimal GPT language model over a small token vocabulary."""

    def __init__(self, cfg: dict, vocab_size: int, max_seq_len: int = 32):
        super().__init__()
        n_layer = cfg["n_layer"]
        d_model = cfg["d_model"]
        self.max_seq_len = max_seq_len

        self.tok_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)
        self.ln1s = nn.ModuleList(nn.LayerNorm(d_model) for _ in range(n_layer))
        self.attns = nn.ModuleList(
            CausalSelfAttention(d_model, cfg["n_head"]) for _ in range(n_layer)
        )
        self.ln2s = nn.ModuleList(nn.LayerNorm(d_model) for _ in range(n_layer))
        self.mlps = nn.ModuleList(MLP(d_model, cfg["mlp_ratio"]) for _ in range(n_layer))
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)  # untied

    def forward(
        self, tokens: torch.Tensor, record: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """tokens: (B, T) long -> logits (B, T, vocab).

        With ``record=True``, also returns a dict of named activations for
        batch item 0 (detached cloned float32 CPU tensors).
        """
        T = tokens.shape[1]
        assert T <= self.max_seq_len, f"T={T} exceeds max_seq_len={self.max_seq_len}"
        acts: dict[str, torch.Tensor] | None = {} if record else None

        def rec(name: str, tensor: torch.Tensor) -> None:
            if acts is not None:
                acts[name] = _snapshot(tensor)

        positions = torch.arange(T, device=tokens.device)
        x = self.tok_embedding(tokens) + self.pos_embedding(positions)
        rec("embed.out", x[0])

        for i in range(len(self.attns)):
            normed = self.ln1s[i](x)
            rec(f"layer{i}.ln1.out", normed[0])
            x = x + self.attns[i](normed, record=acts, prefix=f"layer{i}.attn.")
            normed = self.ln2s[i](x)
            rec(f"layer{i}.ln2.out", normed[0])
            x = x + self.mlps[i](normed, record=acts, prefix=f"layer{i}.mlp.")

        x = self.ln_f(x)
        rec("final_norm.out", x[0])

        logits = self.lm_head(x)
        rec("head.logits", logits[0])

        if acts is not None:
            return logits, acts
        return logits
