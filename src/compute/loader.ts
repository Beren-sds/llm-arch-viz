/**
 * Weight loader for the export artifacts produced by
 * training/llmviz_train/export.py:
 *
 *   public/models/<arch>/manifest.json  — arch, dims, checkpoint provenance,
 *                                         and the tensors table; offsets and
 *                                         lengths are in FLOAT-COUNT units
 *                                         (float32 elements, NOT bytes)
 *   public/models/<arch>/weights.bin    — every tensor as little-endian
 *                                         float32, concatenated in manifest
 *                                         order (sorted by tensor name)
 *
 * Endianness: weights.bin is little-endian f32. JS Float32Array views are
 * platform-endian, but every target platform of this app (x86-64, arm64,
 * wasm) is little-endian, so a plain Float32Array view reads correctly.
 * We document rather than branch on this; a DataView fallback would only
 * matter on big-endian hosts none of our targets have.
 */

import { T } from "./tensor";

/** One entry of the manifest tensors table; offset/length in f32 elements. */
export interface TensorEntry {
  name: string;
  shape: number[];
  offset: number;
  length: number;
}

/** Checkpoint provenance recorded by export.py. */
export interface ManifestCheckpoint {
  run_dir: string;
  step: number;
  val_exact: number;
  golden_seed: number;
  golden_candidates_skipped: number;
}

export interface Manifest {
  arch: string;
  offset_unit: "float32";
  /** Arch config + vocab_size + derived dims (keys differ per arch). */
  dims: Record<string, number>;
  checkpoint: ManifestCheckpoint;
  tensors: TensorEntry[];
}

/**
 * Slice weights.bin into named tensors per the manifest.
 *
 * Pure and strict: validates offset_unit, per-entry contiguity (entries
 * must tile [0, totalFloats) in array order with no gaps or overlaps),
 * shape-product === length, and total byte length — any violation throws;
 * nothing is ever silently truncated or zero-filled. Returned tensors own
 * copies of their data (T.from copies), not views into `bin`.
 */
export function parseWeights(manifest: Manifest, bin: ArrayBuffer): Map<string, T> {
  if (manifest.offset_unit !== "float32") {
    throw new Error(
      `unsupported offset_unit "${String(manifest.offset_unit)}" (expected "float32")`,
    );
  }

  // Validate the tensor table before touching any bytes.
  let cursor = 0;
  for (const entry of manifest.tensors) {
    if (entry.offset !== cursor) {
      throw new Error(
        `tensor "${entry.name}": offset ${entry.offset} breaks contiguity ` +
          `(expected ${cursor}; entries must tile the buffer in order, no gaps/overlaps)`,
      );
    }
    const size = entry.shape.reduce((a, d) => a * d, 1);
    if (size !== entry.length) {
      throw new Error(
        `tensor "${entry.name}": shape [${entry.shape.join(", ")}] has ` +
          `${size} elements but length is ${entry.length}`,
      );
    }
    cursor += entry.length;
  }

  const expectedBytes = 4 * cursor;
  if (bin.byteLength !== expectedBytes) {
    throw new Error(
      `weights.bin byte length mismatch: manifest implies ${expectedBytes} bytes ` +
        `(${cursor} float32s), got ${bin.byteLength}`,
    );
  }

  const weights = new Map<string, T>();
  for (const entry of manifest.tensors) {
    if (weights.has(entry.name)) {
      throw new Error(`duplicate tensor name "${entry.name}" in manifest`);
    }
    // Platform-endian view == little-endian on all our targets (see header).
    const view = new Float32Array(bin, 4 * entry.offset, entry.length);
    weights.set(entry.name, T.from(view, entry.shape));
  }
  return weights;
}

/**
 * Fetch and parse one exported model.
 *
 * Any non-OK response throws with status + url; network errors from
 * fetchFn propagate. Never returns partial or fake data.
 */
export async function loadModel(
  arch: string,
  baseUrl: string,
  fetchFn: typeof fetch = fetch,
): Promise<{ manifest: Manifest; weights: Map<string, T> }> {
  const manifestUrl = `${baseUrl}/models/${arch}/manifest.json`;
  const manifestRes = await fetchFn(manifestUrl);
  if (!manifestRes.ok) {
    throw new Error(`failed to fetch ${manifestUrl}: HTTP ${manifestRes.status}`);
  }
  const manifest = (await manifestRes.json()) as Manifest;

  const binUrl = `${baseUrl}/models/${arch}/weights.bin`;
  const binRes = await fetchFn(binUrl);
  if (!binRes.ok) {
    throw new Error(`failed to fetch ${binUrl}: HTTP ${binRes.status}`);
  }
  const bin = await binRes.arrayBuffer();

  return { manifest, weights: parseWeights(manifest, bin) };
}
