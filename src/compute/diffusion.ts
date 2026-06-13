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

/** Trajectory of a confidence-ordered iterative denoise over the answer tail. */
export interface DiffusionSampler {
  /** Absolute answer-tail positions, in the order they are unmasked. */
  order: number[];
  /** The token id committed at each revealed position (aligned with order). */
  values: number[];
}

/** Argmax index + its softmax probability for logits row `p` of (T, vocab). */
function rowArgmaxProb(logits: T, p: number, vocab: number): { argmax: number; prob: number } {
  const base = p * vocab;
  let max = -Infinity;
  let argmax = 0;
  for (let j = 0; j < vocab; j++) {
    const v = logits.data[base + j];
    if (v > max) {
      max = v;
      argmax = j;
    }
  }
  let sum = 0;
  for (let j = 0; j < vocab; j++) sum += Math.exp(logits.data[base + j] - max);
  return { argmax, prob: 1 / sum }; // exp(max-max)=1, divided by the sum
}

/**
 * Confidence-ordered iterative denoise of the answer tail (the last
 * `answerLen` positions of `baseTokens`). Each step forwards the
 * partially-masked sequence, then commits the still-masked answer position
 * whose top prediction is most confident. Deterministic, and built only on
 * the golden-gated diffusionForward — it adds a sampling *policy*, not new
 * trained compute.
 */
export function diffusionSampleTrajectory(
  weights: Map<string, T>,
  dims: DiffusionDims,
  baseTokens: number[],
  answerLen: number,
): DiffusionSampler {
  const len = baseTokens.length;
  const ansStart = len - answerLen;
  const work = baseTokens.slice();
  for (let i = ansStart; i < len; i++) work[i] = dims.mask_id;
  const order: number[] = [];
  const values: number[] = [];
  const revealed = new Set<number>();
  for (let step = 0; step < answerLen; step++) {
    const logits = diffusionForward(weights, dims, work);
    let bestPos = -1;
    let bestConf = -Infinity;
    let bestVal = 0;
    for (let p = ansStart; p < len; p++) {
      if (revealed.has(p)) continue;
      const { argmax, prob } = rowArgmaxProb(logits, p, dims.vocab_size);
      if (prob > bestConf) {
        bestConf = prob;
        bestPos = p;
        bestVal = argmax;
      }
    }
    work[bestPos] = bestVal;
    revealed.add(bestPos);
    order.push(bestPos);
    values.push(bestVal);
  }
  return { order, values };
}

/** Masked input after revealing the first `k` steps of a trajectory. */
export function diffusionRevealInput(
  baseTokens: number[],
  dims: DiffusionDims,
  answerLen: number,
  sampler: DiffusionSampler,
  k: number,
): number[] {
  const len = baseTokens.length;
  const ansStart = len - answerLen;
  const work = baseTokens.slice();
  for (let i = ansStart; i < len; i++) work[i] = dims.mask_id;
  for (let j = 0; j < Math.min(k, sampler.order.length); j++) {
    work[sampler.order[j]] = sampler.values[j];
  }
  return work;
}
