"""Minimal masked text-diffusion reference with activation recording.

A discrete (absorbing-state) diffusion language model: a BIDIRECTIONAL
Transformer denoiser. Training corrupts a sequence by replacing tokens
with a [MASK] id at a random rate; the model predicts the originals at the
masked positions from full (non-causal) context. Sampling starts from an
all-masked sequence and unmasks iteratively. Here the model is exactly a
GPT with the causal mask removed (so every position attends to all
positions) and an extra [MASK] vocabulary id.

Architecture (pre-norm residual stack, like GPT but non-causal):

    tokens -> tok_embedding + pos_embedding
           -> [x = x + attn_i(ln1_i(x)); x = x + mlp_i(ln2_i(x))] * n_layer
           -> ln_f -> head (untied) -> logits   # predicts the clean tokens
"""

import math

import torch
import torch.nn.functional as F
from torch import nn

from llmviz_train.gpt import MLP
from llmviz_train.recording import snapshot


class BidirAttention(nn.Module):
    """Multi-head self-attention with NO causal mask (no residual)."""

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
        B, T, d_model = x.shape

        def rec(name: str, t: torch.Tensor) -> None:
            if record is not None:
                record[prefix + name] = snapshot(t)

        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        rec("q", q[0])
        rec("k", k[0])
        rec("v", v[0])

        # No causal mask: a denoiser sees the whole (corrupted) sequence.
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)
        rec("scores", scores[0])
        weights = F.softmax(scores, dim=-1)
        rec("weights", weights[0])

        out = (weights @ v).transpose(1, 2).reshape(B, T, d_model)
        out = self.out_proj(out)
        rec("out", out[0])
        return out


class Diffusion(nn.Module):
    """Minimal bidirectional masked-diffusion denoiser."""

    def __init__(self, cfg: dict, vocab_size: int, max_seq_len: int = 32):
        super().__init__()
        n_layer = cfg["n_layer"]
        d_model = cfg["d_model"]
        self.max_seq_len = max_seq_len
        self.mask_id = cfg["mask_id"]

        self.tok_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)
        self.ln1s = nn.ModuleList(nn.LayerNorm(d_model) for _ in range(n_layer))
        self.attns = nn.ModuleList(BidirAttention(d_model, cfg["n_head"]) for _ in range(n_layer))
        self.ln2s = nn.ModuleList(nn.LayerNorm(d_model) for _ in range(n_layer))
        self.mlps = nn.ModuleList(MLP(d_model, cfg["mlp_ratio"]) for _ in range(n_layer))
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)  # untied

    def forward(
        self, tokens: torch.Tensor, record: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """tokens: (B, T) long (some positions = mask_id) -> logits (B, T, vocab)."""
        T = tokens.shape[1]
        assert T <= self.max_seq_len, f"T={T} exceeds max_seq_len={self.max_seq_len}"
        acts: dict[str, torch.Tensor] | None = {} if record else None

        def rec(name: str, t: torch.Tensor) -> None:
            if acts is not None:
                acts[name] = snapshot(t)

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
        logits = self.head(x)
        rec("head.logits", logits[0])

        if acts is not None:
            return logits, acts
        return logits
