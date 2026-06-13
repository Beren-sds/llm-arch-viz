/**
 * TS port of the RWKV-4 forward pass in training/llmviz_train/rwkv.py,
 * operation-for-operation, recording the same activation names. Gated by
 * rwkv.golden.test.ts against PyTorch goldens at 1e-4.
 *
 * Per layer, two residual sub-blocks over a pre-LayerNorm'd input:
 *   time-mix: token-shift → r,k,v → WKV recurrence → out = (σ(r)·wkv)·Wo
 *   channel-mix: token-shift → k = relu(·)² → out = σ(r) · (k·Wv)
 *
 * WKV is the numerically-stable max-shifted recurrence (one step per token),
 * matching the RWKV CUDA kernel and the Python reference loop.
 */

import { T } from "./tensor";
import { add, embedding, layernorm, linear, mul } from "./ops";
import { getTensor, type Manifest } from "./loader";
import type { Recorder } from "./recorder";
import { dimsFromManifest } from "./util";

export interface RwkvDims {
  n_layer: number;
  d_model: number;
  ffn_mult: number;
  vocab_size: number;
  d_ffn: number;
}

const RWKV_DIM_KEYS = ["n_layer", "d_model", "ffn_mult", "vocab_size", "d_ffn"] as const;

/** Validated RwkvDims from a manifest (throws on wrong arch / missing keys). */
export function rwkvDimsFrom(manifest: Manifest): RwkvDims {
  return dimsFromManifest(manifest, "rwkv", RWKV_DIM_KEYS);
}

function sigmoid(x: T): T {
  const out = new Float32Array(x.data.length);
  for (let i = 0; i < out.length; i++) out[i] = 1 / (1 + Math.exp(-x.data[i]));
  return T.from(out, x.shape);
}

/** relu then square (RWKV channel-mix hidden activation). */
function reluSquare(x: T): T {
  const out = new Float32Array(x.data.length);
  for (let i = 0; i < out.length; i++) {
    const v = x.data[i] > 0 ? x.data[i] : 0;
    out[i] = v * v;
  }
  return T.from(out, x.shape);
}

/** Each row (position) sees the previous row; row 0 is zeros. x: (T, C). */
function tokenShift(x: T): T {
  const [steps, c] = x.shape;
  const out = new Float32Array(steps * c); // zero-filled row 0
  for (let t = 1; t < steps; t++) out.set(x.data.subarray((t - 1) * c, t * c), t * c);
  return T.from(out, x.shape);
}

/** Per-channel mix: out[t,c] = x·mix[c] + xs·(1−mix[c]). x,xs: (T,C); mix: (C,). */
function timeMix(x: T, xs: T, mix: T): T {
  const [steps, c] = x.shape;
  const out = new Float32Array(steps * c);
  for (let t = 0; t < steps; t++) {
    for (let j = 0; j < c; j++) {
      const m = mix.data[j];
      const i = t * c + j;
      out[i] = x.data[i] * m + xs.data[i] * (1 - m);
    }
  }
  return T.from(out, x.shape);
}

/**
 * WKV linear attention: per channel, a numerator/denominator state with
 * decay w = −exp(time_decay) and current-token bonus u = time_first, in the
 * max-shifted stable form. k, v: (T, C); time_decay, time_first: (C,).
 */
function wkv(k: T, v: T, timeDecay: T, timeFirst: T): T {
  const [steps, c] = k.shape;
  const out = new Float32Array(steps * c);
  const aa = new Float32Array(c);
  const bb = new Float32Array(c);
  const pp = new Float32Array(c).fill(-1e38);
  const w = new Float32Array(c);
  for (let j = 0; j < c; j++) w[j] = -Math.exp(timeDecay.data[j]);
  const u = timeFirst.data;

  for (let t = 0; t < steps; t++) {
    const base = t * c;
    for (let j = 0; j < c; j++) {
      const kt = k.data[base + j];
      const vt = v.data[base + j];
      // output for the current token (includes the bonus u)
      let ww = u[j] + kt;
      let q = Math.max(pp[j], ww);
      let e1 = Math.exp(pp[j] - q);
      let e2 = Math.exp(ww - q);
      out[base + j] = (e1 * aa[j] + e2 * vt) / (e1 * bb[j] + e2);
      // advance the state by one step of decay
      ww = pp[j] + w[j];
      q = Math.max(ww, kt);
      e1 = Math.exp(ww - q);
      e2 = Math.exp(kt - q);
      aa[j] = e1 * aa[j] + e2 * vt;
      bb[j] = e1 * bb[j] + e2;
      pp[j] = q;
    }
  }
  return T.from(out, k.shape);
}

/** One time-mixing sub-block (no residual). x already LayerNorm'd: (T, d_model). */
function timeMixBlock(
  weights: Map<string, T>,
  layer: number,
  x: T,
  rec: Recorder | undefined,
): T {
  const p = `layer${layer}.att.`;
  const w = (s: string): T => getTensor(weights, `att.${layer}.${s}`);
  const xs = tokenShift(x);
  const r = sigmoid(linear(timeMix(x, xs, w("time_mix_r")), w("receptance.weight")));
  const k = linear(timeMix(x, xs, w("time_mix_k")), w("key.weight"));
  const v = linear(timeMix(x, xs, w("time_mix_v")), w("value.weight"));
  rec?.record(`${p}r`, r);
  rec?.record(`${p}k`, k);
  rec?.record(`${p}v`, v);
  const wk = wkv(k, v, w("time_decay"), w("time_first"));
  rec?.record(`${p}wkv`, wk);
  const out = linear(mul(r, wk), w("output.weight"));
  rec?.record(`${p}out`, out);
  return out;
}

/** One channel-mixing sub-block (no residual). x already LayerNorm'd: (T, d_model). */
function channelMixBlock(
  weights: Map<string, T>,
  layer: number,
  x: T,
  rec: Recorder | undefined,
): T {
  const p = `layer${layer}.ffn.`;
  const w = (s: string): T => getTensor(weights, `ffn.${layer}.${s}`);
  const xs = tokenShift(x);
  const k = reluSquare(linear(timeMix(x, xs, w("time_mix_k")), w("key.weight")));
  rec?.record(`${p}k`, k);
  const r = sigmoid(linear(timeMix(x, xs, w("time_mix_r")), w("receptance.weight")));
  const out = mul(r, linear(k, w("value.weight")));
  rec?.record(`${p}out`, out);
  return out;
}

/**
 * Full forward: tokens → logits (T, vocab), recording the same activation
 * names as rwkv.py when `rec` is given.
 */
export function rwkvForward(
  weights: Map<string, T>,
  dims: RwkvDims,
  tokens: number[],
  rec?: Recorder,
): T {
  let x = embedding(tokens, getTensor(weights, "embedding.weight"));
  rec?.record("embed.out", x);

  for (let i = 0; i < dims.n_layer; i++) {
    const n1 = layernorm(x, getTensor(weights, `ln1.${i}.weight`), getTensor(weights, `ln1.${i}.bias`));
    rec?.record(`layer${i}.ln1.out`, n1);
    x = add(x, timeMixBlock(weights, i, n1, rec));

    const n2 = layernorm(x, getTensor(weights, `ln2.${i}.weight`), getTensor(weights, `ln2.${i}.bias`));
    rec?.record(`layer${i}.ln2.out`, n2);
    x = add(x, channelMixBlock(weights, i, n2, rec));
  }

  x = layernorm(x, getTensor(weights, "ln_out.weight"), getTensor(weights, "ln_out.bias"));
  rec?.record("final_norm.out", x);

  const logits = linear(x, getTensor(weights, "head.weight"));
  rec?.record("head.logits", logits);
  return logits;
}
