/**
 * TS port of the GPT forward pass in training/llmviz_train/gpt.py,
 * operation-for-operation, recording the same activation names. Gated by
 * gpt.golden.test.ts against PyTorch goldens at 1e-4.
 *
 * Architecture (pre-norm residual stack):
 *
 *     tokens -> tok_embedding + pos_embedding
 *            -> [x = x + attn_i(ln1_i(x)); x = x + mlp_i(ln2_i(x))] * n_layer
 *            -> ln_f -> lm_head (untied) -> logits
 *
 * CausalSelfAttention (head_dim = d_model / n_head):
 *
 *     qkv_proj -> split (q, k, v)               [torch.chunk order]
 *     per head: scores = q @ k^T / sqrt(head_dim),
 *               strict upper triangle masked to -inf BEFORE softmax,
 *               out_h = softmax(scores) @ v
 *     out = out_proj(concat heads)
 *
 * The batch dim of gpt.py is dropped (the goldens record batch item 0);
 * heads are processed as 2D (T, head_dim) / (T, T) slices and re-stacked
 * into (n_head, T, ·) tensors only for recording, matching the reference's
 * (B, n_head, T, ·) layout at B index 0.
 *
 * MLP: fc -> GELU (exact erf) -> proj.
 */

import { T } from "./tensor";
import { add, embedding, gelu, layernorm, linear, matmul, scale, softmax, transpose2d } from "./ops";
import { getTensor, type Manifest } from "./loader";
import type { Recorder } from "./recorder";
import { dimsFromManifest, sliceCols } from "./util";

/** GPT dims as exported in manifest.json (head_dim = d_model / n_head). */
export interface GptDims {
  n_layer: number;
  d_model: number;
  n_head: number;
  mlp_ratio: number;
  vocab_size: number;
  head_dim: number;
  max_seq_len: number;
}

const GPT_DIM_KEYS = [
  "n_layer",
  "d_model",
  "n_head",
  "mlp_ratio",
  "vocab_size",
  "head_dim",
  "max_seq_len",
] as const;

/** Validated GptDims from a manifest (throws on wrong arch / missing keys). */
export function gptDimsFrom(manifest: Manifest): GptDims {
  const dims = dimsFromManifest(manifest, "gpt", GPT_DIM_KEYS);
  if (dims.n_head * dims.head_dim !== dims.d_model) {
    throw new Error(
      `gpt dims inconsistent: n_head ${dims.n_head} * head_dim ${dims.head_dim} ` +
        `!= d_model ${dims.d_model}`,
    );
  }
  return dims;
}

/** Stack n same-shaped 2D tensors (r,c) into one (n,r,c) tensor (for recording). */
function stack(slices: T[]): T {
  const [r, c] = slices[0].shape;
  const out = T.zeros([slices.length, r, c]);
  for (let i = 0; i < slices.length; i++) {
    out.data.set(slices[i].data, i * r * c);
  }
  return out;
}

/** Minimal attention dims (a structural subset of GptDims and MoeDims). */
export interface AttnDims {
  d_model: number;
  n_head: number;
  head_dim: number;
}

/**
 * One CausalSelfAttention (gpt.py CausalSelfAttention.forward), no
 * residual — the caller adds it. x: (T, d_model) -> (T, d_model).
 * Records q/k/v/scores/weights/out under `prefix`. Exported so MoE (whose
 * attention is identical, same `attns.{i}.*` weight names) can reuse it.
 */
export function causalSelfAttention(
  weights: Map<string, T>,
  dims: AttnDims,
  layer: number,
  x: T,
  rec: Recorder | undefined,
): T {
  const { d_model: dModel, n_head: nHead, head_dim: headDim } = dims;
  const steps = x.shape[0];
  const prefix = `layer${layer}.attn.`;
  const w = (suffix: string): T => getTensor(weights, `attns.${layer}.${suffix}`);

  // qkv_proj -> split (q, k, v) in torch.chunk(3, dim=-1) order.
  const qkv = linear(x, w("qkv_proj.weight"), w("qkv_proj.bias")); // (T, 3*d_model)
  const q = sliceCols(qkv, 0, dModel);
  const k = sliceCols(qkv, dModel, 2 * dModel);
  const v = sliceCols(qkv, 2 * dModel, 3 * dModel);

  // view(T, n_head, head_dim).transpose(0, 1): head h owns columns
  // [h*head_dim, (h+1)*head_dim) of the (T, d_model) matrix.
  const qh: T[] = [];
  const kh: T[] = [];
  const vh: T[] = [];
  for (let h = 0; h < nHead; h++) {
    qh.push(sliceCols(q, h * headDim, (h + 1) * headDim));
    kh.push(sliceCols(k, h * headDim, (h + 1) * headDim));
    vh.push(sliceCols(v, h * headDim, (h + 1) * headDim));
  }
  rec?.record(`${prefix}q`, stack(qh)); // (n_head, T, head_dim)
  rec?.record(`${prefix}k`, stack(kh));
  rec?.record(`${prefix}v`, stack(vh));

  // Per head: scores = q @ k^T / sqrt(head_dim), strict upper triangle
  // masked to -inf (post-mask recorded), softmax, then weights @ v.
  const scoresH: T[] = [];
  const weightsH: T[] = [];
  const outH: T[] = [];
  for (let h = 0; h < nHead; h++) {
    const scores = scale(matmul(qh[h], transpose2d(kh[h])), 1 / Math.sqrt(headDim)); // (T, T)
    for (let i = 0; i < steps; i++) {
      for (let j = i + 1; j < steps; j++) {
        scores.data[i * steps + j] = -Infinity;
      }
    }
    scoresH.push(scores);
    const wts = softmax(scores); // (T, T)
    weightsH.push(wts);
    outH.push(matmul(wts, vh[h])); // (T, head_dim)
  }
  rec?.record(`${prefix}scores`, stack(scoresH)); // (n_head, T, T), post-mask
  rec?.record(`${prefix}weights`, stack(weightsH));

  // transpose(1, 2).reshape(T, d_model): re-interleave heads column-wise.
  const merged = T.zeros([steps, dModel]);
  for (let h = 0; h < nHead; h++) {
    for (let t = 0; t < steps; t++) {
      merged.data.set(outH[h].data.subarray(t * headDim, (t + 1) * headDim), t * dModel + h * headDim);
    }
  }

  const out = linear(merged, w("out_proj.weight"), w("out_proj.bias"));
  rec?.record(`${prefix}out`, out);
  return out;
}

/**
 * One MLP (gpt.py MLP.forward): fc -> GELU (exact erf) -> proj, no
 * residual. x: (T, d_model) -> (T, d_model). Records under `prefix`.
 */
function mlp(
  weights: Map<string, T>,
  layer: number,
  x: T,
  rec: Recorder | undefined,
): T {
  const prefix = `layer${layer}.mlp.`;
  const w = (suffix: string): T => getTensor(weights, `mlps.${layer}.${suffix}`);

  const pre = linear(x, w("fc.weight"), w("fc.bias")); // (T, d_mlp)
  rec?.record(`${prefix}fc`, pre);
  const post = gelu(pre);
  rec?.record(`${prefix}act`, post);
  const out = linear(post, w("proj.weight"), w("proj.bias"));
  rec?.record(`${prefix}proj`, out);
  return out;
}

/**
 * Full forward pass: tokens -> logits (T, vocab), recording the same
 * activation names as gpt.py when `rec` is given.
 */
export function gptForward(
  weights: Map<string, T>,
  dims: GptDims,
  tokens: number[],
  rec?: Recorder,
): T {
  if (tokens.length > dims.max_seq_len) {
    throw new Error(`T=${tokens.length} exceeds max_seq_len=${dims.max_seq_len}`);
  }

  const positions = tokens.map((_, i) => i);
  let x = add(
    embedding(tokens, getTensor(weights, "tok_embedding.weight")),
    embedding(positions, getTensor(weights, "pos_embedding.weight")),
  ); // (T, d_model)
  rec?.record("embed.out", x);

  for (let i = 0; i < dims.n_layer; i++) {
    let normed = layernorm(
      x,
      getTensor(weights, `ln1s.${i}.weight`),
      getTensor(weights, `ln1s.${i}.bias`),
    );
    rec?.record(`layer${i}.ln1.out`, normed);
    x = add(x, causalSelfAttention(weights, dims, i, normed, rec));

    normed = layernorm(
      x,
      getTensor(weights, `ln2s.${i}.weight`),
      getTensor(weights, `ln2s.${i}.bias`),
    );
    rec?.record(`layer${i}.ln2.out`, normed);
    x = add(x, mlp(weights, i, normed, rec));
  }

  x = layernorm(x, getTensor(weights, "ln_f.weight"), getTensor(weights, "ln_f.bias"));
  rec?.record("final_norm.out", x);

  const logits = linear(x, getTensor(weights, "lm_head.weight"));
  rec?.record("head.logits", logits);
  return logits;
}
