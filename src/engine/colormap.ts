/**
 * Diverging value colormap for tensor cells.
 *
 * Finite values map blue -> near-white -> red over [-scale, +scale]
 * (linear in each half, clamped beyond). Special values get sentinel
 * colors so they are visually unmistakable:
 *   -Infinity  -> dark slate ("masked", e.g. causal-attention cells)
 *   +Infinity / NaN -> magenta (something is wrong upstream)
 */

import type { T } from "../compute/tensor";

export interface RGB {
  r: number;
  g: number;
  b: number;
}

/** v = -scale: deep blue. */
const NEG = { r: 0.13, g: 0.3, b: 0.85 } as const;
/** v = 0: near-white. */
const MID = { r: 0.96, g: 0.96, b: 0.97 } as const;
/** v = +scale: deep red. */
const POS = { r: 0.88, g: 0.18, b: 0.16 } as const;

/** -Infinity (masked cells, e.g. GPT attention above the diagonal). */
export const MASKED_COLOR: RGB = { r: 0.16, g: 0.18, b: 0.22 };
/** +Infinity / NaN — loud error signal, should never appear in healthy runs. */
export const ERROR_COLOR: RGB = { r: 1, g: 0, b: 1 };

/**
 * Write the color for value `v` under `scale` into `out`; returns `out`.
 * Throws if `scale` is not a finite positive number.
 */
export function valueColor(v: number, scale: number, out: RGB): RGB {
  if (!Number.isFinite(scale) || scale <= 0) {
    throw new Error(`valueColor: scale must be a finite positive number, got ${scale}`);
  }
  if (v === -Infinity) {
    out.r = MASKED_COLOR.r;
    out.g = MASKED_COLOR.g;
    out.b = MASKED_COLOR.b;
    return out;
  }
  if (!Number.isFinite(v)) {
    // +Infinity or NaN
    out.r = ERROR_COLOR.r;
    out.g = ERROR_COLOR.g;
    out.b = ERROR_COLOR.b;
    return out;
  }
  // t in [-1, 1], clamped.
  let t = v / scale;
  if (t > 1) t = 1;
  else if (t < -1) t = -1;
  const end = t < 0 ? NEG : POS;
  const a = t < 0 ? -t : t;
  // Two-product lerp form is exact at a=0 and a=1 (hits MID/end precisely).
  const ia = 1 - a;
  out.r = MID.r * ia + end.r * a;
  out.g = MID.g * ia + end.g * a;
  out.b = MID.b * ia + end.b * a;
  return out;
}

/**
 * Default per-tensor color scale: max |finite value| over the data.
 * Non-finite entries (masked -Infinity, NaN) are ignored; returns 1 when
 * no finite nonzero value exists so valueColor never divides by zero.
 */
export function tensorScale(t: T): number {
  let max = 0;
  const data = t.data;
  for (let i = 0; i < data.length; i++) {
    const a = Math.abs(data[i]);
    if (Number.isFinite(a) && a > max) max = a;
  }
  return max > 0 ? max : 1;
}
