"""Minimal Mixture-of-Experts (MoE) reference with activation recording.

A GPT whose dense MLP is replaced by a sparse MoE block: a router scores
the experts per token, the top-k are selected and renormalized, and each
expert (an MLP) contributes its output weighted by the gate. Attention is
exactly the GPT CausalSelfAttention (imported, not re-derived).

Computed DENSELY for clarity and exact portability: every expert runs on
every token, then combined with the top-k-masked gate weights — the result
is identical to a gather/scatter sparse kernel but trivial to mirror in TS.

Architecture (pre-norm residual stack):

    tokens -> tok_embedding + pos_embedding
           -> [x = x + attn_i(ln1_i(x)); x = x + moe_i(ln2_i(x))] * n_layer
           -> ln_f -> lm_head (untied) -> logits
"""

import torch
import torch.nn.functional as F
from torch import nn

from llmviz_train.gpt import CausalSelfAttention
from llmviz_train.recording import snapshot


class Expert(nn.Module):
    """One expert MLP: fc -> GELU (exact erf) -> proj."""

    def __init__(self, d_model: int, d_expert: int):
        super().__init__()
        self.fc = nn.Linear(d_model, d_expert, bias=True)
        self.act = nn.GELU()
        self.proj = nn.Linear(d_expert, d_model, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.act(self.fc(x)))


class MoE(nn.Module):
    """Top-k routed mixture of experts (no residual; the model adds it)."""

    def __init__(self, d_model: int, d_expert: int, n_experts: int, top_k: int):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k
        self.router = nn.Linear(d_model, n_experts, bias=False)
        self.experts = nn.ModuleList(Expert(d_model, d_expert) for _ in range(n_experts))

    def forward(
        self, x: torch.Tensor, record: dict[str, torch.Tensor] | None = None, prefix: str = ""
    ) -> torch.Tensor:
        """x: (B, T, d_model) -> (B, T, d_model)."""

        def rec(name: str, tensor: torch.Tensor) -> None:
            if record is not None:
                record[prefix + name] = snapshot(tensor)

        logits = self.router(x)  # (B, T, E)
        probs = F.softmax(logits, dim=-1)
        rec("router", probs[0])

        # top-k select + renormalize; scatter back to a dense (B, T, E) gate.
        topv, topi = probs.topk(self.top_k, dim=-1)  # (B, T, k)
        gates = topv / topv.sum(dim=-1, keepdim=True)
        gate_full = torch.zeros_like(probs).scatter(-1, topi, gates)  # (B, T, E)
        rec("gates", gate_full[0])

        out = torch.zeros_like(x)
        for e, expert in enumerate(self.experts):
            he = expert(x)  # (B, T, d_model), dense
            rec(f"expert{e}.out", he[0])
            out = out + gate_full[..., e : e + 1] * he
        rec("out", out[0])
        return out


class MoEModel(nn.Module):
    """Minimal MoE-GPT language model over a small token vocabulary."""

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
        self.moes = nn.ModuleList(
            MoE(d_model, cfg["d_expert"], cfg["n_experts"], cfg["top_k"]) for _ in range(n_layer)
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
            x = x + self.moes[i](normed, record=acts, prefix=f"layer{i}.moe.")

        x = self.ln_f(x)
        rec("final_norm.out", x[0])

        logits = self.lm_head(x)
        rec("head.logits", logits[0])

        if acts is not None:
            return logits, acts
        return logits
