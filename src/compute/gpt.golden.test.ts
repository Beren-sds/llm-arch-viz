/**
 * The GPT half of the phase-1 gate: the TS GPT forward pass must match
 * the PyTorch reference (training/llmviz_train/gpt.py) on every recorded
 * activation, for both golden inputs, within 1e-4 — using the real
 * committed weights. The attn.scores tensors carry -Infinity from the
 * causal mask; the comparator requires sign-matched infinities exactly.
 */

import { describe, expect, it } from "vitest";
import { readFileSync } from "node:fs";
import { argmax } from "./ops";
import { compareToGolden, type GoldenFile } from "./golden";
import { parseWeights, type Manifest } from "./loader";
import { gptDimsFrom, gptForward } from "./gpt";
import { MapRecorder } from "./recorder";

function loadArtifacts(): { manifest: Manifest; weights: ReturnType<typeof parseWeights> } {
  const dir = new URL("../../public/models/gpt/", import.meta.url);
  const manifest = JSON.parse(readFileSync(new URL("manifest.json", dir), "utf8")) as Manifest;
  const raw = readFileSync(new URL("weights.bin", dir));
  const bin = raw.buffer.slice(raw.byteOffset, raw.byteOffset + raw.byteLength);
  return { manifest, weights: parseWeights(manifest, bin) };
}

function loadGoldens(): GoldenFile {
  const url = new URL("../../goldens/gpt/goldens.json", import.meta.url);
  return JSON.parse(readFileSync(url, "utf8")) as GoldenFile;
}

describe("gptForward vs PyTorch goldens (real weights)", () => {
  const { manifest, weights } = loadArtifacts();
  const dims = gptDimsFrom(manifest);
  const goldens = loadGoldens();

  it("golden file has 2 inputs with matching activations", () => {
    expect(goldens.inputs.length).toBe(2);
    expect(goldens.activations.length).toBe(goldens.inputs.length);
  });

  for (let g = 0; g < goldens.inputs.length; g++) {
    describe(`golden input ${g}`, () => {
      const input = goldens.inputs[g];
      const rec = new MapRecorder();
      const logits = gptForward(weights, dims, input.tokens, rec);
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

      it("masks the strict upper triangle of every scores tensor to -inf", () => {
        const steps = input.tokens.length;
        for (let layer = 0; layer < dims.n_layer; layer++) {
          const scores = rec.activations.get(`layer${layer}.attn.scores`);
          expect(scores).toBeDefined();
          expect(scores!.shape).toEqual([dims.n_head, steps, steps]);
          for (let h = 0; h < dims.n_head; h++) {
            for (let i = 0; i < steps; i++) {
              for (let j = 0; j < steps; j++) {
                const v = scores!.at(h, i, j);
                if (j > i) {
                  expect(v).toBe(-Infinity);
                } else {
                  expect(Number.isFinite(v)).toBe(true);
                }
              }
            }
          }
        }
      });

      it("predicts the stored answer at the final positions", () => {
        expect(logits.shape).toEqual([input.tokens.length, dims.vocab_size]);
        const preds = argmax(logits);
        // logits[t] predicts token t+1; the answer is the input's tail.
        const start = input.tokens.length - input.answer.length - 1;
        expect(preds.slice(start, start + input.answer.length)).toEqual(input.answer);
      });
    });
  }
});
