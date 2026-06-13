/**
 * Smoke test for the MoE scene against the real committed manifest +
 * weights: every recorded activation (router, gates, all experts, combined)
 * and every weight tensor has a view; setTokens applies. Labels stubbed.
 */

import { describe, expect, it } from "vitest";
import { readFileSync } from "node:fs";
import * as THREE from "three";
import { parseWeights, type Manifest } from "../compute/loader";
import { moeDimsFrom, moeForward } from "../compute/moe";
import { MapRecorder } from "../compute/recorder";
import type { GoldenFile } from "../compute/golden";
import { I18n, type Locale } from "../i18n/i18n";
import en from "../i18n/en.json";
import zh from "../i18n/zh.json";
import { buildMoeChapters, buildMoeScene, MOE_ANCHOR_NAMES, MOE_SEQ_LEN } from "./moe";

function loadArtifacts(): { manifest: Manifest; weights: ReturnType<typeof parseWeights> } {
  const dir = new URL("../../public/models/moe/", import.meta.url);
  const manifest = JSON.parse(readFileSync(new URL("manifest.json", dir), "utf8")) as Manifest;
  const raw = readFileSync(new URL("weights.bin", dir));
  const bin = raw.buffer.slice(raw.byteOffset, raw.byteOffset + raw.byteLength);
  return { manifest, weights: parseWeights(manifest, bin) };
}

function goldenTokens(): number[] {
  const url = new URL("../../goldens/moe/goldens.json", import.meta.url);
  return (JSON.parse(readFileSync(url, "utf8")) as GoldenFile).inputs[0].tokens;
}

const stubFactory = { label: () => new THREE.Group(), bracket: () => new THREE.Group() };

function build() {
  const { manifest, weights } = loadArtifacts();
  const scene = new THREE.Scene();
  const built = buildMoeScene({ scene, weights, manifest, labelFactory: stubFactory });
  return { manifest, weights, scene, built };
}

describe("buildMoeScene (real manifest + weights)", () => {
  const { manifest, weights, scene, built } = build();
  const dims = moeDimsFrom(manifest);
  const tokens = goldenTokens();
  const rec = new MapRecorder();
  moeForward(weights, dims, tokens, rec);
  const actNames = [...rec.activations.keys()];

  it("has a view for every recorded activation and every weight", () => {
    expect(tokens).toHaveLength(MOE_SEQ_LEN);
    for (const name of actNames) expect(built.views.has(name), name).toBe(true);
    for (const entry of manifest.tensors) expect(built.views.has(entry.name), entry.name).toBe(true);
  });

  it("has no views beyond activations + weights", () => {
    expect(built.views.size).toBe(actNames.length + manifest.tensors.length);
  });

  it("routing + per-expert outputs are present with the right shapes", () => {
    expect(built.views.get("layer0.moe.router")!.shape).toEqual([MOE_SEQ_LEN, dims.n_experts]);
    expect(built.views.get("layer0.moe.gates")!.shape).toEqual([MOE_SEQ_LEN, dims.n_experts]);
    for (let e = 0; e < dims.n_experts; e++) {
      expect(built.views.get(`layer0.moe.expert${e}.out`)!.shape).toEqual([MOE_SEQ_LEN, dims.d_model]);
    }
  });

  it("setTokens applies the recorder's activations to the views", () => {
    built.setTokens(tokens);
    for (const name of ["embed.out", "layer0.moe.gates", "layer0.moe.out", "head.logits"]) {
      expect(Array.from(built.views.get(name)!.lastValues), name).toEqual(
        Array.from(rec.activations.get(name)!.data),
      );
    }
  });

  it("exposes all named anchors plus cameraHome", () => {
    for (const name of MOE_ANCHOR_NAMES) {
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

describe("buildMoeChapters (real scene + real i18n dicts)", () => {
  const { built } = build();
  const i18n = new I18n({ en, zh } satisfies Record<Locale, Record<string, string>>);
  const reg = buildMoeChapters(built, i18n);
  const EXPECTED_IDS = ["intro", "embed", "attention", "moe", "readout"];

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
