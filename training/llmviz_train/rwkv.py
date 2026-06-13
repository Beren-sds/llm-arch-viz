"""Minimal RWKV-4 reference implementation with activation recording.

Golden source for the TypeScript port: every recorded intermediate is
compared against the TS re-implementation at 1e-4. Clarity beats speed —
the WKV recurrence is a plain Python loop over time, one step per token,
in the numerically-stable max-shifted form used by the RWKV CUDA kernel.

Architecture (two residual sub-blocks per layer, pre-LayerNorm):

    tokens -> embedding
           -> [ x = x + time_mix(ln1(x)) ;  x = x + channel_mix(ln2(x)) ] * n_layer
           -> ln_out -> head (untied) -> logits

Time-mixing (linear attention with per-channel decay):
    token-shift mixes x with the previous position; r,k,v are linear maps of
    the mixed input; the WKV recurrence keeps a numerator/denominator state
    per channel with decay w = -exp(time_decay) and bonus u = time_first;
    out = (sigmoid(r) * wkv) @ output.

Channel-mixing (gated squared-ReLU MLP):
    token-shift; k = relu(xk @ key)^2 (width d_ffn); out = sigmoid(xr @ rec) * (k @ value).
"""

import torch
import torch.nn.functional as F
from torch import nn

from llmviz_train.recording import snapshot


def _token_shift(x: torch.Tensor) -> torch.Tensor:
    """Each position sees the previous position's vector (zeros before t=0)."""
    return F.pad(x, (0, 0, 1, -1))  # shift along time (dim=1) by one, zero-fill


class RWKVTimeMix(nn.Module):
    """WKV linear-attention sub-block (no residual; the model adds it)."""

    def __init__(self, d_model: int):
        super().__init__()
        self.time_decay = nn.Parameter(torch.linspace(-3.0, 3.0, d_model))
        self.time_first = nn.Parameter(torch.zeros(d_model))
        self.time_mix_k = nn.Parameter(torch.full((d_model,), 0.5))
        self.time_mix_v = nn.Parameter(torch.full((d_model,), 0.5))
        self.time_mix_r = nn.Parameter(torch.full((d_model,), 0.5))
        self.key = nn.Linear(d_model, d_model, bias=False)
        self.value = nn.Linear(d_model, d_model, bias=False)
        self.receptance = nn.Linear(d_model, d_model, bias=False)
        self.output = nn.Linear(d_model, d_model, bias=False)

    def forward(
        self, x: torch.Tensor, record: dict[str, torch.Tensor] | None = None, prefix: str = ""
    ) -> torch.Tensor:
        """x: (B, T, d_model) (already LayerNorm'd) -> (B, T, d_model)."""
        B_, T, C = x.shape

        def rec(name: str, tensor: torch.Tensor) -> None:
            if record is not None:
                record[prefix + name] = snapshot(tensor)

        xs = _token_shift(x)
        xk = x * self.time_mix_k + xs * (1 - self.time_mix_k)
        xv = x * self.time_mix_v + xs * (1 - self.time_mix_v)
        xr = x * self.time_mix_r + xs * (1 - self.time_mix_r)

        r = torch.sigmoid(self.receptance(xr))  # (B, T, C)
        k = self.key(xk)
        v = self.value(xv)
        rec("r", r[0])
        rec("k", k[0])
        rec("v", v[0])

        w = -torch.exp(self.time_decay)  # (C,) per-channel decay, negative
        u = self.time_first  # (C,) bonus for the current token

        # Numerically-stable WKV: keep (aa, bb, pp) state per channel, with pp
        # the running max-exponent so the exponentials never overflow.
        aa = x.new_zeros(B_, C)
        bb = x.new_zeros(B_, C)
        pp = x.new_full((B_, C), -1e38)
        wkv = x.new_zeros(B_, T, C)
        for t in range(T):
            kt = k[:, t]
            vt = v[:, t]
            ww = u + kt
            q = torch.maximum(pp, ww)
            e1 = torch.exp(pp - q)
            e2 = torch.exp(ww - q)
            wkv[:, t] = (e1 * aa + e2 * vt) / (e1 * bb + e2)
            ww = pp + w
            q = torch.maximum(ww, kt)
            e1 = torch.exp(ww - q)
            e2 = torch.exp(kt - q)
            aa = e1 * aa + e2 * vt
            bb = e1 * bb + e2
            pp = q
        rec("wkv", wkv[0])

        out = self.output(r * wkv)
        rec("out", out[0])
        return out


class RWKVChannelMix(nn.Module):
    """Gated squared-ReLU MLP sub-block (no residual; the model adds it)."""

    def __init__(self, d_model: int, d_ffn: int):
        super().__init__()
        self.time_mix_k = nn.Parameter(torch.full((d_model,), 0.5))
        self.time_mix_r = nn.Parameter(torch.full((d_model,), 0.5))
        self.key = nn.Linear(d_model, d_ffn, bias=False)
        self.receptance = nn.Linear(d_model, d_model, bias=False)
        self.value = nn.Linear(d_ffn, d_model, bias=False)

    def forward(
        self, x: torch.Tensor, record: dict[str, torch.Tensor] | None = None, prefix: str = ""
    ) -> torch.Tensor:
        """x: (B, T, d_model) (already LayerNorm'd) -> (B, T, d_model)."""

        def rec(name: str, tensor: torch.Tensor) -> None:
            if record is not None:
                record[prefix + name] = snapshot(tensor)

        xs = _token_shift(x)
        xk = x * self.time_mix_k + xs * (1 - self.time_mix_k)
        xr = x * self.time_mix_r + xs * (1 - self.time_mix_r)

        k = torch.square(torch.relu(self.key(xk)))  # (B, T, d_ffn)
        rec("k", k[0])
        r = torch.sigmoid(self.receptance(xr))
        out = r * self.value(k)
        rec("out", out[0])
        return out


class RWKV(nn.Module):
    """Minimal RWKV-4 language model over a small token vocabulary."""

    def __init__(self, cfg: dict, vocab_size: int):
        super().__init__()
        n_layer = cfg["n_layer"]
        d_model = cfg["d_model"]
        d_ffn = cfg["ffn_mult"] * d_model

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.ln1 = nn.ModuleList(nn.LayerNorm(d_model) for _ in range(n_layer))
        self.ln2 = nn.ModuleList(nn.LayerNorm(d_model) for _ in range(n_layer))
        self.att = nn.ModuleList(RWKVTimeMix(d_model) for _ in range(n_layer))
        self.ffn = nn.ModuleList(RWKVChannelMix(d_model, d_ffn) for _ in range(n_layer))
        self.ln_out = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)  # untied

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
                acts[name] = snapshot(tensor)

        x = self.embedding(tokens)
        rec("embed.out", x[0])

        for i in range(len(self.att)):
            n1 = self.ln1[i](x)
            rec(f"layer{i}.ln1.out", n1[0])
            x = x + self.att[i](n1, record=acts, prefix=f"layer{i}.att.")
            n2 = self.ln2[i](x)
            rec(f"layer{i}.ln2.out", n2[0])
            x = x + self.ffn[i](n2, record=acts, prefix=f"layer{i}.ffn.")

        x = self.ln_out(x)
        rec("final_norm.out", x[0])

        logits = self.head(x)
        rec("head.logits", logits[0])

        if acts is not None:
            return logits, acts
        return logits
