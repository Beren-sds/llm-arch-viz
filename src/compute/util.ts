/**
 * Small helpers shared by the per-arch forward passes (mamba.ts, gpt.ts):
 * column slicing for torch.chunk/split mirrors, and validated extraction
 * of typed dims from a manifest.
 */

import { T } from "./tensor";
import type { Manifest } from "./loader";

/** Column slice [start, end) of a 2D tensor: (rows, cols) -> (rows, end-start). */
export function sliceCols(x: T, start: number, end: number): T {
  if (x.shape.length !== 2) {
    throw new Error(`sliceCols: x must be 2D, got shape [${x.shape.join(", ")}]`);
  }
  const [rows, cols] = x.shape;
  if (!Number.isInteger(start) || !Number.isInteger(end) || start < 0 || end < start || end > cols) {
    throw new Error(`sliceCols: range [${start}, ${end}) invalid for ${cols} cols`);
  }
  const width = end - start;
  const out = T.zeros([rows, width]);
  for (let r = 0; r < rows; r++) {
    out.data.set(x.data.subarray(r * cols + start, r * cols + end), r * width);
  }
  return out;
}

/**
 * Validate manifest.arch and extract the named numeric dims. Throws on
 * arch mismatch or any missing/non-numeric key; returns a fully typed
 * record so callers need no cast.
 */
export function dimsFromManifest<K extends string>(
  manifest: Manifest,
  arch: string,
  keys: readonly K[],
): Record<K, number> {
  if (manifest.arch !== arch) {
    throw new Error(`manifest arch "${manifest.arch}" != expected "${arch}"`);
  }
  const out: Partial<Record<K, number>> = {};
  for (const k of keys) {
    const v = manifest.dims[k];
    if (typeof v !== "number") {
      throw new Error(`manifest dims missing numeric "${k}"`);
    }
    out[k] = v;
  }
  return out as Record<K, number>;
}
