/**
 * TS port of the KAN forward pass in training/llmviz_train/kan.py,
 * operation-for-operation, recording the same activation names. Gated by
 * kan.golden.test.ts against PyTorch goldens at 1e-4.
 *
 * A GPT whose feed-forward is a FastKAN: each edge function is a weighted
 * sum of fixed Gaussian bumps on a grid. Attention is reused verbatim from
 * gpt.ts. The KAN feed-forward over a LayerNorm'd input x̂:
 *   rbf  = exp(-((x̂ − grid)/denom)²)            (T, d_model, G), flattened
 *   out  = spline_linear(rbf) + base_linear(silu(x̂))
 */

import { T } from "./tensor";
import { add, embedding, layernorm, linear, silu } from "./ops";
import { getTensor, type Manifest } from "./loader";
import type { Recorder } from "./recorder";
import { dimsFromManifest } from "./util";
import { causalSelfAttention } from "./gpt";

export interface KanDims {
  n_layer: number;
  d_model: number;
  n_head: number;
  num_grids: number;
  grid_min: number;
  grid_max: number;
  vocab_size: number;
  head_dim: number;
  max_seq_len: number;
}

const KAN_DIM_KEYS = [
  "n_layer",
  "d_model",
  "n_head",
  "num_grids",
  "grid_min",
  "grid_max",
  "vocab_size",
  "head_dim",
  "max_seq_len",
] as const;

export function kanDimsFrom(manifest: Manifest): KanDims {
  const dims = dimsFromManifest(manifest, "kan", KAN_DIM_KEYS);
  if (dims.n_head * dims.head_dim !== dims.d_model) {
    throw new Error("kan dims inconsistent: n_head·head_dim != d_model");
  }
  return dims;
}

/**
 * Flatten the Gaussian-basis evaluation of x̂ to (T, d_model·G), in the
 * (channel, grid) row-major order the spline_linear weight expects.
 */
function rbfFeatures(x: T, dims: KanDims): T {
  const [steps, d] = x.shape;
  const g = dims.num_grids;
  const denom = (dims.grid_max - dims.grid_min) / (g - 1);
  const grid = new Float32Array(g);
  for (let j = 0; j < g; j++) grid[j] = dims.grid_min + j * denom;
  const out = new Float32Array(steps * d * g);
  for (let t = 0; t < steps; t++) {
    for (let i = 0; i < d; i++) {
      const xv = x.data[t * d + i];
      const base = (t * d + i) * g;
      for (let j = 0; j < g; j++) {
        const z = (xv - grid[j]) / denom;
        out[base + j] = Math.exp(-(z * z));
      }
    }
  }
  return T.from(out, [steps, d * g]);
}

/** One KAN feed-forward (no residual). x already LayerNorm'd: (T, d_model). */
function kanBlock(
  weights: Map<string, T>,
  dims: KanDims,
  layer: number,
  x: T,
  rec: Recorder | undefined,
): T {
  const rbf = rbfFeatures(x, dims);
  rec?.record(`layer${layer}.kan.rbf`, rbf);
  const spline = linear(rbf, getTensor(weights, `kans.${layer}.spline_linear.weight`));
  rec?.record(`layer${layer}.kan.spline`, spline);
  const base = linear(
    silu(x),
    getTensor(weights, `kans.${layer}.base_linear.weight`),
    getTensor(weights, `kans.${layer}.base_linear.bias`),
  );
  rec?.record(`layer${layer}.kan.base`, base);
  const out = add(spline, base);
  rec?.record(`layer${layer}.kan.out`, out);
  return out;
}

/** Full forward: tokens → logits (T, vocab), recording the kan.py names. */
export function kanForward(
  weights: Map<string, T>,
  dims: KanDims,
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
    x = add(x, kanBlock(weights, dims, i, normed, rec));
  }

  x = layernorm(x, getTensor(weights, "ln_f.weight"), getTensor(weights, "ln_f.bias"));
  rec?.record("final_norm.out", x);

  const logits = linear(x, getTensor(weights, "lm_head.weight"));
  rec?.record("head.logits", logits);
  return logits;
}
