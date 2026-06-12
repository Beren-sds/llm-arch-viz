/**
 * Golden-activation comparator shared by the Mamba and GPT forward gates.
 *
 * goldens/<arch>/goldens.json layout (written by export.py):
 *
 *   {
 *     "inputs":      [{ "tokens": number[], "answer": number[] }, ...],
 *     "activations": [{ "<name>": { "shape": number[], "data": [...] } }, ...]
 *   }
 *
 * with activations[i] belonging to inputs[i]. JSON cannot carry non-finite
 * floats, so data entries may be the strings "Infinity" / "-Infinity" /
 * "NaN" (the GPT goldens contain "-Infinity" from the causal mask);
 * decodeGoldenTensor maps them back to numbers and rejects anything else.
 */

import { T } from "./tensor";

/** One recorded activation as stored in goldens.json. */
export interface GoldenTensor {
  shape: number[];
  data: (number | string)[];
}

/** All activations for one golden input, keyed by recording name. */
export type GoldenActivations = Record<string, GoldenTensor>;

export interface GoldenFile {
  inputs: { tokens: number[]; answer: number[] }[];
  activations: GoldenActivations[];
}

function decodeValue(v: number | string, name: string, idx: number): number {
  if (typeof v === "number") return v;
  switch (v) {
    case "Infinity":
      return Infinity;
    case "-Infinity":
      return -Infinity;
    case "NaN":
      return NaN;
    default:
      throw new Error(`golden "${name}"[${idx}]: unrecognized value string "${v}"`);
  }
}

/** Decode one golden tensor into a T (throws on shape/data length mismatch). */
export function decodeGoldenTensor(g: GoldenTensor, name = "<golden>"): T {
  return T.from(
    g.data.map((v, i) => decodeValue(v, name, i)),
    g.shape,
  );
}

export interface WorstEntry {
  name: string;
  idx: number;
  got: number;
  want: number;
  diff: number;
}

export interface GoldenComparison {
  /** Largest-diff element across all common tensors (null if none compared). */
  worst: WorstEntry | null;
  maxAbsDiff: number;
  /** Names present in the golden but never recorded. */
  missing: string[];
  /** Names recorded but absent from the golden. */
  extra: string[];
}

/**
 * Per-element diff with non-finite semantics:
 *  - NaN vs NaN -> 0; NaN vs anything else -> Infinity
 *  - sign-matched infinities -> 0; any other finite/non-finite mix -> Infinity
 *  - finite vs finite -> |got - want|
 */
function valueDiff(got: number, want: number): number {
  const gotNaN = Number.isNaN(got);
  const wantNaN = Number.isNaN(want);
  if (gotNaN || wantNaN) return gotNaN && wantNaN ? 0 : Infinity;
  if (!Number.isFinite(got) || !Number.isFinite(want)) {
    return got === want ? 0 : Infinity;
  }
  return Math.abs(got - want);
}

/**
 * Diff a recorded activation map against one golden input's activations.
 *
 * Shape mismatch on a common name throws (that is a structural bug, not a
 * numeric divergence). Ties on diff keep the earliest element, so a fully
 * matching run reports the first compared element with diff 0.
 */
export function compareToGolden(
  recorded: Map<string, T>,
  golden: GoldenActivations,
): GoldenComparison {
  const missing: string[] = [];
  const extra: string[] = [];
  let worst: WorstEntry | null = null;

  for (const name of Object.keys(golden)) {
    const got = recorded.get(name);
    if (got === undefined) {
      missing.push(name);
      continue;
    }
    const want = decodeGoldenTensor(golden[name], name);
    if (
      got.shape.length !== want.shape.length ||
      got.shape.some((d, i) => d !== want.shape[i])
    ) {
      throw new Error(
        `golden "${name}": shape mismatch, recorded [${got.shape.join(", ")}] ` +
          `vs golden [${want.shape.join(", ")}]`,
      );
    }
    for (let i = 0; i < want.size; i++) {
      const diff = valueDiff(got.data[i], want.data[i]);
      if (worst === null || diff > worst.diff) {
        worst = { name, idx: i, got: got.data[i], want: want.data[i], diff };
      }
    }
  }

  for (const name of recorded.keys()) {
    if (!(name in golden)) extra.push(name);
  }

  return { worst, maxAbsDiff: worst === null ? 0 : worst.diff, missing, extra };
}
