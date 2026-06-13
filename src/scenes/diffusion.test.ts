/**
 * Smoke test for the diffusion scene against the real committed manifest +
 * weights: every recorded activation and every weight tensor has a view;
 * setTokens masks the answer tail and applies the denoiser's activations.
 */

import { describe, expect, it } from "vitest";
import { readFileSync } from "node:fs";
import * as THREE from "three";
import { parseWeights, type Manifest } from "../compute/loader";
import { diffusionDimsFrom, diffusionForward } from "../compute/diffusion";
import { MapRecorder } from "../compute/recorder";
import type { GoldenFile } from "../compute/golden";
import { I18n, type Locale } from "../i18n/i18n";
import en from "../i18n/en.json";
import zh from "../i18n/zh.json";
import {
  buildDiffusionChapters,
  buildDiffusionScene,
  DIFFUSION_ANCHOR_NAMES,
  DIFFUSION_SEQ_LEN,
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

const stubFactory = { label: () => new THREE.Group(), bracket: () => new THREE.Group() };

function build() {
  const { manifest, weights } = loadArtifacts();
  const scene = new THREE.Scene();
  const built = buildDiffusionScene({ scene, weights, manifest, labelFactory: stubFactory });
  return { manifest, weights, scene, built };
}

describe("buildDiffusionScene (real manifest + weights)", () => {
  const { manifest, weights, scene, built } = build();
  const dims = diffusionDimsFrom(manifest);
  const g = golden();
  const rec = new MapRecorder();
  // The golden tokens are already masked at the answer tail (denoiser input).
  diffusionForward(weights, dims, g.tokens, rec);
  const actNames = [...rec.activations.keys()];

  it("has a view for every recorded activation and every weight", () => {
    expect(g.tokens).toHaveLength(DIFFUSION_SEQ_LEN);
    for (const name of actNames) expect(built.views.has(name), name).toBe(true);
    for (const entry of manifest.tensors) expect(built.views.has(entry.name), entry.name).toBe(true);
  });

  it("has no views beyond activations + weights", () => {
    expect(built.views.size).toBe(actNames.length + manifest.tensors.length);
  });

  it("attention scores are bidirectional (no masked -inf triangle)", () => {
    const s = built.views.get("layer0.attn.scores")!;
    expect(s.shape).toEqual([dims.n_head, DIFFUSION_SEQ_LEN, DIFFUSION_SEQ_LEN]);
    // Build the clean sequence, let the scene mask + denoise, then check no -inf.
    const clean = g.tokens.slice();
    for (let i = 0; i < g.answer.length; i++) clean[clean.length - g.answer.length + i] = g.answer[i];
    built.setTokens(clean);
    expect(Array.from(s.lastValues).every(Number.isFinite)).toBe(true);
  });

  it("setTokens masks the answer tail and applies the denoiser's activations", () => {
    const clean = g.tokens.slice();
    for (let i = 0; i < g.answer.length; i++) clean[clean.length - g.answer.length + i] = g.answer[i];
    built.setTokens(clean); // scene masks the tail -> same input as the golden tokens
    for (const name of ["embed.out", "layer0.attn.weights", "head.logits"]) {
      expect(Array.from(built.views.get(name)!.lastValues), name).toEqual(
        Array.from(rec.activations.get(name)!.data),
      );
    }
  });

  it("exposes all named anchors plus cameraHome", () => {
    for (const name of DIFFUSION_ANCHOR_NAMES) {
      const kf = built.anchors.get(name);
      expect(kf, name).toBeDefined();
      expect([...kf!.pos, ...kf!.target].every(Number.isFinite)).toBe(true);
    }
    expect(built.cameraHome).toBe(built.anchors.get("home"));
  });

  it("dispose detaches the scene root", () => {
    expect(() => built.dispose()).not.toThrow();
    expect(scene.children).toHaveLength(0);
  });
});

describe("buildDiffusionChapters (real scene + real i18n dicts)", () => {
  const { built } = build();
  const i18n = new I18n({ en, zh } satisfies Record<Locale, Record<string, string>>);
  const reg = buildDiffusionChapters(built, i18n);
  const EXPECTED_IDS = ["intro", "embed", "attention", "mlp", "readout"];

  it("registers the chapters in order", () => {
    expect(reg.count).toBe(EXPECTED_IDS.length);
    expect(EXPECTED_IDS.map((_, i) => reg.get(i).id)).toEqual(EXPECTED_IDS);
  });

  it("every narration + title key exists in both locales", () => {
    for (let i = 0; i < reg.count; i++) {
      const ch = reg.get(i);
      expect(i18n.missingLocales(ch.narrationKey), ch.narrationKey).toEqual([]);
      expect(i18n.missingLocales(ch.narrationKey.replace(/\.body$/, ".title"))).toEqual([]);
    }
  });

  it("every focus-highlight names a real view", () => {
    for (let i = 0; i < reg.count; i++) {
      for (const name of reg.get(i).highlights) {
        expect(built.views.has(name), `chapter "${reg.get(i).id}" highlight "${name}"`).toBe(true);
      }
    }
  });
});
