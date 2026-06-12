/**
 * TS port of the Mamba forward pass in training/llmviz_train/mamba.py,
 * operation-for-operation, recording the same activation names. Gated by
 * mamba.golden.test.ts against PyTorch goldens at 1e-4.
 *
 * Architecture (pre-norm residual stack):
 *
 *     tokens -> embedding -> [x = x + block_i(rmsnorm_i(x))] * n_layer
 *            -> final_norm -> lm_head (untied) -> logits
 *
 * MambaBlock (d_inner = expand * d_model):
 *
 *     in_proj -> split (x_part, z)                       [x_part first]
 *     x_part -> depthwise causal conv1d -> SiLU
 *            -> x_proj -> split (dt, B, C)               [in that order]
 *            -> delta = softplus(dt_proj(dt))
 *            -> selective scan with A = -exp(A_log), D skip
 *     out = out_proj(y * silu(z))
 */

import { T } from "./tensor";
import {
  add,
  causalDepthwiseConv1d,
  embedding,
  exp,
  linear,
  mul,
  rmsnorm,
  scale,
  silu,
  softplus,
} from "./ops";
import { getTensor, type Manifest } from "./loader";
import type { Recorder } from "./recorder";
import { dimsFromManifest, sliceCols } from "./util";

/** Mamba dims as exported in manifest.json (dt_rank = ceil(d_model/16)). */
export interface MambaDims {
  n_layer: number;
  d_model: number;
  d_state: number;
  d_conv: number;
  expand: number;
  vocab_size: number;
  dt_rank: number;
}

const MAMBA_DIM_KEYS = [
  "n_layer",
  "d_model",
  "d_state",
  "d_conv",
  "expand",
  "vocab_size",
  "dt_rank",
] as const;

/** Validated MambaDims from a manifest (throws on wrong arch / missing keys). */
export function mambaDimsFrom(manifest: Manifest): MambaDims {
  return dimsFromManifest(manifest, "mamba", MAMBA_DIM_KEYS);
}

/**
 * One MambaBlock (mamba.py MambaBlock.forward), no residual — the caller
 * adds it. x: (T, d_model) -> (T, d_model). Records under `prefix`.
 */
function mambaBlock(
  weights: Map<string, T>,
  dims: MambaDims,
  layer: number,
  x: T,
  rec: Recorder | undefined,
): T {
  const { d_state: dState, d_conv: dConv, dt_rank: dtRank } = dims;
  const dInner = dims.expand * dims.d_model;
  const steps = x.shape[0];
  const prefix = `layer${layer}.`;
  const w = (suffix: string): T => getTensor(weights, `blocks.${layer}.${suffix}`);

  // in_proj -> split (x_part, z), x_part first (torch.chunk(2, dim=-1)).
  const xz = linear(x, w("in_proj.weight")); // (T, 2*d_inner)
  rec?.record(`${prefix}in_proj.out`, xz);
  const xPart = sliceCols(xz, 0, dInner);
  const z = sliceCols(xz, dInner, 2 * dInner);

  // Depthwise causal conv along time, then SiLU. conv1d.weight is PyTorch
  // (d_inner, 1, d_conv); squeeze the middle dim to (d_inner, d_conv).
  const convW3 = w("conv1d.weight");
  const convKernels = T.from(convW3.data, [convW3.shape[0], convW3.shape[2]]);
  const xConv = silu(causalDepthwiseConv1d(xPart, convKernels, w("conv1d.bias"), dConv));
  rec?.record(`${prefix}conv.out`, xConv);

  // x_proj -> split (dt, B, C) in that order (torch.split sizes).
  const dbc = linear(xConv, w("x_proj.weight")); // (T, dt_rank + 2*d_state)
  rec?.record(`${prefix}x_proj.out`, dbc);
  const dt = sliceCols(dbc, 0, dtRank);
  const bMat = sliceCols(dbc, dtRank, dtRank + dState);
  const cMat = sliceCols(dbc, dtRank + dState, dtRank + 2 * dState);

  const delta = softplus(linear(dt, w("dt_proj.weight"), w("dt_proj.bias"))); // (T, d_inner)
  rec?.record(`${prefix}delta.out`, delta);

  const A = scale(exp(w("A_log")), -1); // (d_inner, d_state), A = -exp(A_log)
  const D = w("D"); // (d_inner)

  // Selective scan, one token at a time. This is the one place raw indexed
  // loops are used: h is recurrent state, updated in place per step.
  //
  //   h[d,s] = exp(delta[t,d] * A[d,s]) * h[d,s] + delta[t,d] * B[t,s] * x[t,d]
  //   y[t,d] = sum_s h[d,s] * C[t,s] + D[d] * x[t,d]
  const h = T.zeros([dInner, dState]);
  const y = T.zeros([steps, dInner]);
  for (let t = 0; t < steps; t++) {
    for (let d = 0; d < dInner; d++) {
      const deltaTd = delta.data[t * dInner + d];
      const xTd = xConv.data[t * dInner + d];
      for (let s = 0; s < dState; s++) {
        h.data[d * dState + s] =
          Math.exp(deltaTd * A.data[d * dState + s]) * h.data[d * dState + s] +
          deltaTd * bMat.data[t * dState + s] * xTd;
      }
    }
    rec?.record(`${prefix}ssm.h.t${t}`, h); // MapRecorder clones the snapshot
    for (let d = 0; d < dInner; d++) {
      let acc = 0; // f64 accumulation, f32 round on store
      for (let s = 0; s < dState; s++) {
        acc += h.data[d * dState + s] * cMat.data[t * dState + s];
      }
      y.data[t * dInner + d] = acc + D.data[d] * xConv.data[t * dInner + d];
    }
  }
  rec?.record(`${prefix}ssm.out`, y);

  const gated = mul(y, silu(z));
  rec?.record(`${prefix}gate.out`, gated);

  const out = linear(gated, w("out_proj.weight"));
  rec?.record(`${prefix}out_proj.out`, out);
  return out;
}

/**
 * Full forward pass: tokens -> logits (T, vocab), recording the same
 * activation names as mamba.py when `rec` is given.
 */
export function mambaForward(
  weights: Map<string, T>,
  dims: MambaDims,
  tokens: number[],
  rec?: Recorder,
): T {
  let x = embedding(tokens, getTensor(weights, "embedding.weight")); // (T, d_model)
  rec?.record("embed.out", x);

  for (let i = 0; i < dims.n_layer; i++) {
    const normed = rmsnorm(x, getTensor(weights, `norms.${i}.weight`));
    rec?.record(`layer${i}.norm.out`, normed);
    x = add(x, mambaBlock(weights, dims, i, normed, rec));
  }

  x = rmsnorm(x, getTensor(weights, "final_norm.weight"));
  rec?.record("final_norm.out", x);

  const logits = linear(x, getTensor(weights, "lm_head.weight"));
  rec?.record("head.logits", logits);
  return logits;
}
