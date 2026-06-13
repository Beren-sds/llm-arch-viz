/**
 * TS port of the MoE forward pass in training/llmviz_train/moe.py,
 * operation-for-operation, recording the same activation names. Gated by
 * moe.golden.test.ts against PyTorch goldens at 1e-4.
 *
 * A GPT whose dense MLP is replaced by a top-k routed mixture of experts.
 * Attention is reused verbatim from gpt.ts (identical weights). The MoE
 * block scores experts with a router, keeps the top-k per token
 * (renormalized), and combines each expert's dense output by its gate.
 */

import { T } from "./tensor";
import { add, embedding, gelu, layernorm, linear, softmax } from "./ops";
import { getTensor, type Manifest } from "./loader";
import type { Recorder } from "./recorder";
import { dimsFromManifest } from "./util";
import { causalSelfAttention } from "./gpt";

export interface MoeDims {
  n_layer: number;
  d_model: number;
  n_head: number;
  d_expert: number;
  n_experts: number;
  top_k: number;
  vocab_size: number;
  head_dim: number;
  max_seq_len: number;
}

const MOE_DIM_KEYS = [
  "n_layer",
  "d_model",
  "n_head",
  "d_expert",
  "n_experts",
  "top_k",
  "vocab_size",
  "head_dim",
  "max_seq_len",
] as const;

export function moeDimsFrom(manifest: Manifest): MoeDims {
  const dims = dimsFromManifest(manifest, "moe", MOE_DIM_KEYS);
  if (dims.n_head * dims.head_dim !== dims.d_model) {
    throw new Error(`moe dims inconsistent: n_head·head_dim != d_model`);
  }
  return dims;
}

/** One expert MLP: fc → GELU → proj (matches gpt.MLP). */
function expert(weights: Map<string, T>, layer: number, e: number, x: T): T {
  const w = (s: string): T => getTensor(weights, `moes.${layer}.experts.${e}.${s}`);
  return linear(gelu(linear(x, w("fc.weight"), w("fc.bias"))), w("proj.weight"), w("proj.bias"));
}

/**
 * Top-k masked, renormalized gate matrix (T, E) from the softmax router
 * probabilities — gate[t,e] = probs[t,e] / Σ_top-k(probs[t]) when e is in
 * the token's top-k, else 0. Matches torch.topk + scatter in the reference.
 */
function topkGates(probs: T, topK: number): T {
  const [steps, e] = probs.shape;
  const out = new Float32Array(steps * e);
  for (let t = 0; t < steps; t++) {
    const base = t * e;
    // indices of this row's experts sorted by probability, descending
    const idx = Array.from({ length: e }, (_, j) => j).sort(
      (a, b) => probs.data[base + b] - probs.data[base + a],
    );
    let sum = 0;
    for (let r = 0; r < topK; r++) sum += probs.data[base + idx[r]];
    for (let r = 0; r < topK; r++) {
      const j = idx[r];
      out[base + j] = probs.data[base + j] / sum;
    }
  }
  return T.from(out, probs.shape);
}

/** One MoE block (no residual). x already LayerNorm'd: (T, d_model). */
function moeBlock(
  weights: Map<string, T>,
  dims: MoeDims,
  layer: number,
  x: T,
  rec: Recorder | undefined,
): T {
  const [steps, dM] = x.shape;
  const probs = softmax(linear(x, getTensor(weights, `moes.${layer}.router.weight`)));
  rec?.record(`layer${layer}.moe.router`, probs);
  const gates = topkGates(probs, dims.top_k);
  rec?.record(`layer${layer}.moe.gates`, gates);

  const out = new Float32Array(steps * dM);
  for (let e = 0; e < dims.n_experts; e++) {
    const he = expert(weights, layer, e, x); // dense: every token
    rec?.record(`layer${layer}.moe.expert${e}.out`, he);
    for (let t = 0; t < steps; t++) {
      const g = gates.data[t * dims.n_experts + e];
      if (g === 0) continue;
      for (let c = 0; c < dM; c++) out[t * dM + c] += g * he.data[t * dM + c];
    }
  }
  const combined = T.from(out, x.shape);
  rec?.record(`layer${layer}.moe.out`, combined);
  return combined;
}

/** Full forward: tokens → logits (T, vocab), recording the moe.py names. */
export function moeForward(
  weights: Map<string, T>,
  dims: MoeDims,
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
  );
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
    x = add(x, moeBlock(weights, dims, i, normed, rec));
  }

  x = layernorm(x, getTensor(weights, "ln_f.weight"), getTensor(weights, "ln_f.bias"));
  rec?.record("final_norm.out", x);

  const logits = linear(x, getTensor(weights, "lm_head.weight"));
  rec?.record("head.logits", logits);
  return logits;
}
