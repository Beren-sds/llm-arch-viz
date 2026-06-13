/**
 * Golden gate: the TS RWKV forward pass must match the PyTorch reference
 * (training/llmviz_train/rwkv.py) on every recorded activation, for both
 * golden inputs, within 1e-4 — using the real committed weights. The WKV
 * recurrence is the most precision-sensitive piece, so this is the gate
 * that unblocks the RWKV scene.
 */

import { describe, expect, it } from "vitest";
import { readFileSync } from "node:fs";
import { argmax } from "./ops";
import { compareToGolden, type GoldenFile } from "./golden";
import { parseWeights, type Manifest } from "./loader";
import { rwkvDimsFrom, rwkvForward } from "./rwkv";
import { MapRecorder } from "./recorder";

function loadArtifacts(): { manifest: Manifest; weights: ReturnType<typeof parseWeights> } {
  const dir = new URL("../../public/models/rwkv/", import.meta.url);
  const manifest = JSON.parse(readFileSync(new URL("manifest.json", dir), "utf8")) as Manifest;
  const raw = readFileSync(new URL("weights.bin", dir));
  const bin = raw.buffer.slice(raw.byteOffset, raw.byteOffset + raw.byteLength);
  return { manifest, weights: parseWeights(manifest, bin) };
}

function loadGoldens(): GoldenFile {
  const url = new URL("../../goldens/rwkv/goldens.json", import.meta.url);
  return JSON.parse(readFileSync(url, "utf8")) as GoldenFile;
}

describe("rwkvForward vs PyTorch goldens (real weights)", () => {
  const { manifest, weights } = loadArtifacts();
  const dims = rwkvDimsFrom(manifest);
  const goldens = loadGoldens();

  it("golden file has 2 inputs with matching activations", () => {
    expect(goldens.inputs.length).toBe(2);
    expect(goldens.activations.length).toBe(goldens.inputs.length);
  });

  for (let g = 0; g < goldens.inputs.length; g++) {
    describe(`golden input ${g}`, () => {
      const input = goldens.inputs[g];
      const rec = new MapRecorder();
      const logits = rwkvForward(weights, dims, input.tokens, rec);
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

      it("predicts the stored answer at the final positions", () => {
        expect(logits.shape).toEqual([input.tokens.length, dims.vocab_size]);
        const preds = argmax(logits);
        const start = input.tokens.length - input.answer.length - 1;
        expect(preds.slice(start, start + input.answer.length)).toEqual(input.answer);
      });
    });
  }
});
