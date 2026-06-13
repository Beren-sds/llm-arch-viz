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

// Muted diverging palette: a calm light-slate zero that recedes on the dark
// background, fanning to a soft blue and a warm coral rather than saturated
// primaries — the saturated blue/red speckle read as noise, not data.
/** v = -scale: soft blue. */
export const NEG_COLOR = { r: 0.31, g: 0.52, b: 0.82 } as const;
/** v = 0: soft light slate (not pure white — pure white glares on dark). */
export const MID_COLOR = { r: 0.81, g: 0.82, b: 0.86 } as const;
/** v = +scale: warm coral. */
export const POS_COLOR = { r: 0.88, g: 0.46, b: 0.38 } as const;

/**
 * Contrast curve exponent applied to |t| before the lerp (< 1 lifts
 * mid-magnitude cells toward their hue). Endpoints stay exact (0^γ=0,
 * 1^γ=1), so the colour at -scale/0/+scale is unchanged — only the spread
 * between them is enriched, so grids read as structure rather than washing
 * to the near-neutral zero colour. The exact value is always in the tooltip.
 */
export const CONTRAST = 0.7;

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
  const end = t < 0 ? NEG_COLOR : POS_COLOR;
  const a = t < 0 ? -t : t;
  // Contrast-curved blend; exact at a=0 and a=1 (hits MID/end precisely).
  const ca = a === 0 || a === 1 ? a : Math.pow(a, CONTRAST);
  const ia = 1 - ca;
  out.r = MID_COLOR.r * ia + end.r * ca;
  out.g = MID_COLOR.g * ia + end.g * ca;
  out.b = MID_COLOR.b * ia + end.b * ca;
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
