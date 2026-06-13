/**
 * TS port of the masked text-diffusion denoiser in
 * training/llmviz_train/diffusion.py, operation-for-operation, recording
 * the same activation names. Gated by diffusion.golden.test.ts at 1e-4.
 *
 * A GPT with the causal mask removed: every position attends to all
 * positions (causal=false), predicting the clean tokens at masked inputs.
 * Attention is the shared causalSelfAttention with causal=false; the MLP is
 * the same fc → GELU → proj.
 */

import { T } from "./tensor";
import { add, embedding, gelu, layernorm, linear } from "./ops";
import { getTensor, type Manifest } from "./loader";
import type { Recorder } from "./recorder";
import { dimsFromManifest } from "./util";
import { causalSelfAttention } from "./gpt";

export interface DiffusionDims {
  n_layer: number;
  d_model: number;
  n_head: number;
  mlp_ratio: number;
  mask_id: number;
  vocab_size: number;
  head_dim: number;
  max_seq_len: number;
}

const DIFFUSION_DIM_KEYS = [
  "n_layer",
  "d_model",
  "n_head",
  "mlp_ratio",
  "mask_id",
  "vocab_size",
  "head_dim",
  "max_seq_len",
] as const;

export function diffusionDimsFrom(manifest: Manifest): DiffusionDims {
  const dims = dimsFromManifest(manifest, "diffusion", DIFFUSION_DIM_KEYS);
  if (dims.n_head * dims.head_dim !== dims.d_model) {
    throw new Error("diffusion dims inconsistent: n_head·head_dim != d_model");
  }
  return dims;
}

function mlp(weights: Map<string, T>, layer: number, x: T, rec: Recorder | undefined): T {
  const w = (s: string): T => getTensor(weights, `mlps.${layer}.${s}`);
  const fc = linear(x, w("fc.weight"), w("fc.bias"));
  rec?.record(`layer${layer}.mlp.fc`, fc);
  const act = gelu(fc);
  rec?.record(`layer${layer}.mlp.act`, act);
  const proj = linear(act, w("proj.weight"), w("proj.bias"));
  rec?.record(`layer${layer}.mlp.proj`, proj);
  return proj;
}

/** Full forward: (masked) tokens → logits (T, vocab), recording diffusion.py names. */
export function diffusionForward(
  weights: Map<string, T>,
  dims: DiffusionDims,
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
    // causal=false — the denoiser sees the whole (corrupted) sequence.
    x = add(x, causalSelfAttention(weights, dims, i, normed, rec, false));

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

  const logits = linear(x, getTensor(weights, "head.weight"));
  rec?.record("head.logits", logits);
  return logits;
}
