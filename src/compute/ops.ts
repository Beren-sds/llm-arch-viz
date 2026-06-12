/**
 * Tensor ops mirroring the exact forward-pass semantics of the tiny
 * PyTorch reference models (training/llmviz_train/{mamba,gpt}.py).
 *
 * Conventions:
 *  - every op returns a new T; inputs are never mutated;
 *  - inner loops accumulate in f64 (plain JS numbers) and round to f32
 *    only on store into the output Float32Array; this stays within the
 *    1e-4 golden-test gate of PyTorch's f32-accumulated kernels;
 *  - "last dim" ops (rmsnorm, layernorm, softmax, argmax) treat the
 *    tensor as (rows, lastDim) regardless of leading rank.
 */

import { T } from "./tensor";

function assert2D(x: T, name: string): void {
  if (x.shape.length !== 2) {
    throw new Error(`${name} must be 2D, got shape [${x.shape.join(", ")}]`);
  }
}

function assertSameShape(a: T, b: T, op: string): void {
  if (a.shape.length !== b.shape.length || a.shape.some((d, i) => d !== b.shape[i])) {
    throw new Error(
      `${op}: shape mismatch [${a.shape.join(", ")}] vs [${b.shape.join(", ")}]`,
    );
  }
}

function lastDim(x: T): number {
  return x.shape[x.shape.length - 1];
}

/** 2D x 2D matrix product: (m,k) @ (k,n) -> (m,n). */
export function matmul(a: T, b: T): T {
  assert2D(a, "matmul: a");
  assert2D(b, "matmul: b");
  const [m, k] = a.shape;
  const [k2, n] = b.shape;
  if (k !== k2) {
    throw new Error(`matmul: inner dims differ, (${m},${k}) @ (${k2},${n})`);
  }
  const out = T.zeros([m, n]);
  for (let i = 0; i < m; i++) {
    for (let j = 0; j < n; j++) {
      let acc = 0; // f64 accumulation
      for (let p = 0; p < k; p++) {
        acc += a.data[i * k + p] * b.data[p * n + j];
      }
      out.data[i * n + j] = acc; // f32 round on store
    }
  }
  return out;
}

/**
 * nn.Linear: x (T,in) @ w^T + b, with w in PyTorch (out,in) layout
 * and optional bias (out). Returns (T,out).
 */
export function linear(x: T, w: T, b?: T): T {
  assert2D(x, "linear: x");
  assert2D(w, "linear: w");
  const [t, dIn] = x.shape;
  const [dOut, dIn2] = w.shape;
  if (dIn !== dIn2) {
    throw new Error(`linear: x in-dim ${dIn} != w in-dim ${dIn2} (w is (out,in))`);
  }
  if (b !== undefined && (b.shape.length !== 1 || b.shape[0] !== dOut)) {
    throw new Error(`linear: bias shape [${b.shape.join(", ")}] != [${dOut}]`);
  }
  const out = T.zeros([t, dOut]);
  for (let i = 0; i < t; i++) {
    for (let o = 0; o < dOut; o++) {
      let acc = b !== undefined ? b.data[o] : 0;
      for (let j = 0; j < dIn; j++) {
        acc += x.data[i * dIn + j] * w.data[o * dIn + j];
      }
      out.data[i * dOut + o] = acc;
    }
  }
  return out;
}

/** nn.Embedding row gather: ids (n) into table (vocab,d) -> (n,d). */
export function embedding(ids: number[], table: T): T {
  assert2D(table, "embedding: table");
  const [vocab, d] = table.shape;
  const out = T.zeros([ids.length, d]);
  for (let i = 0; i < ids.length; i++) {
    const id = ids[i];
    if (!Number.isInteger(id) || id < 0 || id >= vocab) {
      throw new Error(`embedding: id ${id} out of range [0, ${vocab})`);
    }
    out.data.set(table.data.subarray(id * d, (id + 1) * d), i * d);
  }
  return out;
}

/**
 * RMSNorm over the last dim (mamba.py): x * rsqrt(mean(x^2) + eps) * weight.
 * No bias, no mean subtraction.
 */
export function rmsnorm(x: T, weight: T, eps = 1e-5): T {
  const d = lastDim(x);
  if (weight.shape.length !== 1 || weight.shape[0] !== d) {
    throw new Error(`rmsnorm: weight shape [${weight.shape.join(", ")}] != [${d}]`);
  }
  const out = T.zeros(x.shape);
  const rows = x.size / d;
  for (let r = 0; r < rows; r++) {
    const base = r * d;
    let sumSq = 0;
    for (let j = 0; j < d; j++) {
      const v = x.data[base + j];
      sumSq += v * v;
    }
    const inv = 1 / Math.sqrt(sumSq / d + eps);
    for (let j = 0; j < d; j++) {
      out.data[base + j] = x.data[base + j] * inv * weight.data[j];
    }
  }
  return out;
}

/**
 * nn.LayerNorm over the last dim (gpt.py): (x - mean) / sqrt(var + eps)
 * * weight + bias, with biased variance (divide by d, like PyTorch).
 */
export function layernorm(x: T, weight: T, bias: T, eps = 1e-5): T {
  const d = lastDim(x);
  if (weight.shape.length !== 1 || weight.shape[0] !== d) {
    throw new Error(`layernorm: weight shape [${weight.shape.join(", ")}] != [${d}]`);
  }
  if (bias.shape.length !== 1 || bias.shape[0] !== d) {
    throw new Error(`layernorm: bias shape [${bias.shape.join(", ")}] != [${d}]`);
  }
  const out = T.zeros(x.shape);
  const rows = x.size / d;
  for (let r = 0; r < rows; r++) {
    const base = r * d;
    let mean = 0;
    for (let j = 0; j < d; j++) mean += x.data[base + j];
    mean /= d;
    let varAcc = 0;
    for (let j = 0; j < d; j++) {
      const c = x.data[base + j] - mean;
      varAcc += c * c;
    }
    const inv = 1 / Math.sqrt(varAcc / d + eps);
    for (let j = 0; j < d; j++) {
      out.data[base + j] = (x.data[base + j] - mean) * inv * weight.data[j] + bias.data[j];
    }
  }
  return out;
}

/**
 * Softmax over the last dim, max-subtracted for stability.
 *
 * -inf entries (causal mask) map to exact 0: exp(-inf - finiteMax) === 0.
 * An all--inf row yields NaN (max is -inf, so -inf - -inf = NaN), which
 * matches PyTorch F.softmax; under a causal mask this never occurs
 * because row t always has >= 1 finite entry (positions 0..t).
 */
export function softmax(x: T): T {
  const d = lastDim(x);
  const out = T.zeros(x.shape);
  const rows = x.size / d;
  const exps = new Float64Array(d); // scratch row buffer, reused across rows
  for (let r = 0; r < rows; r++) {
    const base = r * d;
    let max = -Infinity;
    for (let j = 0; j < d; j++) {
      if (x.data[base + j] > max) max = x.data[base + j];
    }
    let sum = 0;
    for (let j = 0; j < d; j++) {
      const e = Math.exp(x.data[base + j] - max);
      exps[j] = e;
      sum += e;
    }
    for (let j = 0; j < d; j++) {
      out.data[base + j] = exps[j] / sum;
    }
  }
  return out;
}

/** SiLU: x * sigmoid(x) (F.silu in mamba.py). */
export function silu(x: T): T {
  const out = T.zeros(x.shape);
  for (let i = 0; i < x.size; i++) {
    const v = x.data[i];
    out.data[i] = v / (1 + Math.exp(-v));
  }
  return out;
}

/**
 * erf via Abramowitz & Stegun 7.1.26 polynomial (max abs error ~1.5e-7).
 * JS Math has no erf; this approximation is comfortably inside the
 * 1e-4 golden-test gate. Computed in f64.
 */
function erf(x: number): number {
  const sign = x < 0 ? -1 : 1;
  const ax = Math.abs(x);
  const t = 1 / (1 + 0.3275911 * ax);
  const poly =
    ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t +
      0.254829592) *
    t;
  return sign * (1 - poly * Math.exp(-ax * ax));
}

/**
 * GELU, exact erf variant (nn.GELU default in gpt.py):
 * gelu(x) = x * Phi(x) = x * 0.5 * (1 + erf(x / sqrt(2))).
 */
export function gelu(x: T): T {
  const out = T.zeros(x.shape);
  for (let i = 0; i < x.size; i++) {
    const v = x.data[i];
    out.data[i] = v * 0.5 * (1 + erf(v / Math.SQRT2));
  }
  return out;
}

/**
 * Softplus log(1 + exp(x)), numerically stable: returns x identically
 * for x > 20 (PyTorch F.softplus threshold), else log1p(exp(x)).
 */
export function softplus(x: T): T {
  const out = T.zeros(x.shape);
  for (let i = 0; i < x.size; i++) {
    const v = x.data[i];
    out.data[i] = v > 20 ? v : Math.log1p(Math.exp(v));
  }
  return out;
}

/**
 * Depthwise causal conv1d along time (mamba.py conv1d with groups=C,
 * padding=dConv-1, then truncated back to T steps):
 *
 *   out[t,c] = bias[c] + sum_k kernels[c,k] * x[t - (dConv-1) + k, c]
 *
 * with out-of-range x treated as 0 (the left zero-padding).
 * x: (T,C), kernels: (C,dConv), bias: (C) -> (T,C).
 */
export function causalDepthwiseConv1d(x: T, kernels: T, bias: T, dConv: number): T {
  assert2D(x, "causalDepthwiseConv1d: x");
  assert2D(kernels, "causalDepthwiseConv1d: kernels");
  const [steps, c] = x.shape;
  if (kernels.shape[0] !== c || kernels.shape[1] !== dConv) {
    throw new Error(
      `causalDepthwiseConv1d: kernels shape [${kernels.shape.join(", ")}] != [${c}, ${dConv}]`,
    );
  }
  if (bias.shape.length !== 1 || bias.shape[0] !== c) {
    throw new Error(`causalDepthwiseConv1d: bias shape [${bias.shape.join(", ")}] != [${c}]`);
  }
  const out = T.zeros([steps, c]);
  for (let t = 0; t < steps; t++) {
    for (let ch = 0; ch < c; ch++) {
      let acc = bias.data[ch];
      for (let k = 0; k < dConv; k++) {
        const src = t - (dConv - 1) + k;
        if (src >= 0) {
          acc += kernels.data[ch * dConv + k] * x.data[src * c + ch];
        }
      }
      out.data[t * c + ch] = acc;
    }
  }
  return out;
}

/** Elementwise exp (the mamba scan's exp(delta*A) and A = -exp(A_log)). */
export function exp(x: T): T {
  const out = T.zeros(x.shape);
  for (let i = 0; i < x.size; i++) out.data[i] = Math.exp(x.data[i]);
  return out;
}

/** Multiply every element by a scalar. */
export function scale(x: T, s: number): T {
  const out = T.zeros(x.shape);
  for (let i = 0; i < x.size; i++) out.data[i] = x.data[i] * s;
  return out;
}

/** Elementwise add; shapes must match exactly (no broadcasting). */
export function add(a: T, b: T): T {
  assertSameShape(a, b, "add");
  const out = T.zeros(a.shape);
  for (let i = 0; i < a.size; i++) out.data[i] = a.data[i] + b.data[i];
  return out;
}

/** Elementwise multiply; shapes must match exactly (no broadcasting). */
export function mul(a: T, b: T): T {
  assertSameShape(a, b, "mul");
  const out = T.zeros(a.shape);
  for (let i = 0; i < a.size; i++) out.data[i] = a.data[i] * b.data[i];
  return out;
}

/** Argmax along the last dim, one index per row (first occurrence wins). */
export function argmax(x: T): number[] {
  const d = lastDim(x);
  const rows = x.size / d;
  const out: number[] = new Array(rows);
  for (let r = 0; r < rows; r++) {
    const base = r * d;
    let best = 0;
    let bestVal = x.data[base];
    for (let j = 1; j < d; j++) {
      if (x.data[base + j] > bestVal) {
        bestVal = x.data[base + j];
        best = j;
      }
    }
    out[r] = best;
  }
  return out;
}
