/**
 * The iterative denoising sampler is a confidence-ordered *policy* over the
 * already-golden-gated diffusionForward (diffusion.golden.test.ts gates the
 * forward at 1e-4). These checks pin the policy on the real weights:
 * deterministic, reveals each answer position exactly once, monotone reveal
 * count, and it reconstructs the true answer the single-pass denoiser also
 * recovers.
 */

import { describe, expect, it } from "vitest";
import { readFileSync } from "node:fs";
import { parseWeights, type Manifest } from "./loader";
import type { GoldenFile } from "./golden";
import {
  diffusionDimsFrom,
  diffusionRevealInput,
  diffusionSampleTrajectory,
} from "./diffusion";

function loadArtifacts(): { manifest: Manifest; weights: ReturnType<typeof parseWeights> } {
  const dir = new URL("../../public/models/diffusion/", import.meta.url);
  const manifest = JSON.parse(readFileSync(new URL("manifest.json", dir), "utf8")) as Manifest;
  const raw = readFileSync(new URL("weights.bin", dir));
  const bin = raw.buffer.slice(raw.byteOffset, raw.byteOffset + raw.byteLength);
  return { manifest, weights: parseWeights(manifest, bin) };
}

function golden(): { tokens: number[]; answer: number[] } {
  const url = new URL("../../goldens/diffusion/goldens.json", import.meta.url);
  return (JSON.parse(readFileSync(url, "utf8")) as GoldenFile).inputs[0];
}

describe("diffusion iterative sampler (real weights)", () => {
  const { manifest, weights } = loadArtifacts();
  const dims = diffusionDimsFrom(manifest);
  const g = golden();
  const answerLen = g.answer.length;
  const len = g.tokens.length;
  const ansStart = len - answerLen;
  // Clean base: golden tokens with the masked tail filled by the true answer.
  const base = g.tokens.slice();
  for (let i = 0; i < answerLen; i++) base[ansStart + i] = g.answer[i];

  const traj = diffusionSampleTrajectory(weights, dims, base, answerLen);

  it("reveals each answer position exactly once", () => {
    expect(traj.order).toHaveLength(answerLen);
    expect(new Set(traj.order).size).toBe(answerLen);
    for (const p of traj.order) {
      expect(p).toBeGreaterThanOrEqual(ansStart);
      expect(p).toBeLessThan(len);
    }
  });

  it("is deterministic", () => {
    const again = diffusionSampleTrajectory(weights, dims, base, answerLen);
    expect(again.order).toEqual(traj.order);
    expect(again.values).toEqual(traj.values);
  });

  it("reconstructs the true answer at the tail", () => {
    const tail = new Array<number>(answerLen);
    for (let j = 0; j < answerLen; j++) tail[traj.order[j] - ansStart] = traj.values[j];
    expect(tail).toEqual(g.answer);
  });

  it("reveal count is monotone: step k has exactly k committed answer tokens", () => {
    for (let k = 0; k <= answerLen; k++) {
      const inp = diffusionRevealInput(base, dims, answerLen, traj, k);
      let revealed = 0;
      for (let i = ansStart; i < len; i++) if (inp[i] !== dims.mask_id) revealed++;
      expect(revealed).toBe(k);
    }
  });
});
