/**
 * THE load-bearing gate of phase 1: the TS Mamba forward pass must match
 * the PyTorch reference (training/llmviz_train/mamba.py) on every recorded
 * activation, for both golden inputs, within 1e-4 — using the real
 * committed weights. Every downstream visualized value depends on this.
 */

import { describe, expect, it } from "vitest";
import { readFileSync } from "node:fs";
import { argmax } from "./ops";
import { compareToGolden, type GoldenFile } from "./golden";
import { parseWeights, type Manifest } from "./loader";
import { mambaForward, type MambaDims } from "./mamba";
import { MapRecorder } from "./recorder";

function loadArtifacts(): { manifest: Manifest; weights: ReturnType<typeof parseWeights> } {
  const dir = new URL("../../public/models/mamba/", import.meta.url);
  const manifest = JSON.parse(readFileSync(new URL("manifest.json", dir), "utf8")) as Manifest;
  const raw = readFileSync(new URL("weights.bin", dir));
  const bin = raw.buffer.slice(raw.byteOffset, raw.byteOffset + raw.byteLength);
  return { manifest, weights: parseWeights(manifest, bin) };
}

function loadGoldens(): GoldenFile {
  const url = new URL("../../goldens/mamba/goldens.json", import.meta.url);
  return JSON.parse(readFileSync(url, "utf8")) as GoldenFile;
}

function dimsOf(manifest: Manifest): MambaDims {
  const need = [
    "n_layer",
    "d_model",
    "d_state",
    "d_conv",
    "expand",
    "vocab_size",
    "dt_rank",
  ] as const;
  for (const k of need) {
    if (typeof manifest.dims[k] !== "number") {
      throw new Error(`manifest dims missing "${k}"`);
    }
  }
  return manifest.dims as unknown as MambaDims;
}

describe("mambaForward vs PyTorch goldens (real weights)", () => {
  const { manifest, weights } = loadArtifacts();
  const dims = dimsOf(manifest);
  const goldens = loadGoldens();

  it("golden file has 2 inputs with matching activations", () => {
    expect(goldens.inputs.length).toBe(2);
    expect(goldens.activations.length).toBe(2);
  });

  for (let g = 0; g < 2; g++) {
    describe(`golden input ${g}`, () => {
      const input = goldens.inputs[g];
      const rec = new MapRecorder();
      const logits = mambaForward(weights, dims, input.tokens, rec);
      const cmp = compareToGolden(rec.activations, goldens.activations[g]);

      it("records exactly the golden activation names (no missing, no extra)", () => {
        expect(cmp.missing).toEqual([]);
        expect(cmp.extra).toEqual([]);
      });

      it("matches every activation within 1e-4", () => {
        const w = cmp.worst;
        const msg =
          w === null
            ? "no activations compared"
            : `worst tensor "${w.name}" at flat index ${w.idx}: ` +
              `got ${w.got}, want ${w.want}, |diff| ${w.diff}`;
        expect(cmp.worst, msg).not.toBeNull();
        expect(cmp.maxAbsDiff, msg).toBeLessThan(1e-4);
      });

      it("predicts the stored answer at positions 16..19", () => {
        expect(logits.shape).toEqual([input.tokens.length, dims.vocab_size]);
        const preds = argmax(logits);
        expect(preds.slice(16, 20)).toEqual(input.answer);
      });
    });
  }
});
