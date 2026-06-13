/**
 * Smoke test for the GPT scene against the REAL committed manifest +
 * weights (like the golden tests): every recorded activation and every
 * weight tensor has a view; setTokens pushes recorder values into the
 * views; the binding's runForward is an INCLUSIVE prefix. Labels stubbed
 * (troika needs a browser-like global; see LabelFactory).
 */

import { describe, expect, it } from "vitest";
import { readFileSync } from "node:fs";
import * as THREE from "three";
import { parseWeights, type Manifest } from "../compute/loader";
import { gptDimsFrom, gptForward } from "../compute/gpt";
import { MapRecorder } from "../compute/recorder";
import type { GoldenFile } from "../compute/golden";
import { I18n, type Locale } from "../i18n/i18n";
import en from "../i18n/en.json";
import zh from "../i18n/zh.json";
import { buildGptChapters, buildGptScene, GPT_ANCHOR_NAMES, GPT_SEQ_LEN } from "./gpt";

function loadArtifacts(): { manifest: Manifest; weights: ReturnType<typeof parseWeights> } {
  const dir = new URL("../../public/models/gpt/", import.meta.url);
  const manifest = JSON.parse(readFileSync(new URL("manifest.json", dir), "utf8")) as Manifest;
  const raw = readFileSync(new URL("weights.bin", dir));
  const bin = raw.buffer.slice(raw.byteOffset, raw.byteOffset + raw.byteLength);
  return { manifest, weights: parseWeights(manifest, bin) };
}

function goldenTokens(): number[] {
  const url = new URL("../../goldens/gpt/goldens.json", import.meta.url);
  const goldens = JSON.parse(readFileSync(url, "utf8")) as GoldenFile;
  return goldens.inputs[0].tokens;
}

const stubFactory = {
  label: () => new THREE.Group(),
  bracket: () => new THREE.Group(),
};

function build() {
  const { manifest, weights } = loadArtifacts();
  const scene = new THREE.Scene();
  const built = buildGptScene({ scene, weights, manifest, labelFactory: stubFactory });
  return { manifest, weights, scene, built };
}

describe("buildGptScene (real manifest + weights)", () => {
  const { manifest, weights, scene, built } = build();
  const dims = gptDimsFrom(manifest);
  const tokens = goldenTokens();

  const rec = new MapRecorder();
  gptForward(weights, dims, tokens, rec);
  const actNames = [...rec.activations.keys()];

  it("has a TensorView for every activation a real forward records", () => {
    expect(tokens).toHaveLength(GPT_SEQ_LEN);
    for (const name of actNames) {
      expect(built.views.has(name), `missing activation view "${name}"`).toBe(true);
    }
  });

  it("has a TensorView for every weight tensor in the manifest", () => {
    for (const entry of manifest.tensors) {
      expect(built.views.has(entry.name), `missing weight view "${entry.name}"`).toBe(true);
    }
  });

  it("has no views beyond activations + weights", () => {
    expect(built.views.size).toBe(actNames.length + manifest.tensors.length);
  });

  it("attention tensors are 3D (n_head, T, ·)", () => {
    expect(built.views.get("layer0.attn.weights")!.shape).toEqual([dims.n_head, GPT_SEQ_LEN, GPT_SEQ_LEN]);
    expect(built.views.get("layer0.attn.q")!.shape).toEqual([dims.n_head, GPT_SEQ_LEN, dims.head_dim]);
  });

  it("weight views carry their weight values", () => {
    const qkv = built.views.get("attns.0.qkv_proj.weight")!;
    expect(qkv.shape).toEqual([3 * dims.d_model, dims.d_model]);
    expect(Array.from(qkv.lastValues)).toEqual(
      Array.from(weights.get("attns.0.qkv_proj.weight")!.data),
    );
  });

  it("setTokens(golden tokens) applies the recorder's activations to the views", () => {
    built.setTokens(tokens);
    for (const name of ["embed.out", "layer0.attn.weights", "head.logits"]) {
      expect(Array.from(built.views.get(name)!.lastValues), name).toEqual(
        Array.from(rec.activations.get(name)!.data),
      );
    }
  });

  it("binding.runForward(10) is INCLUSIVE: an 11-token prefix with T=11 shapes", () => {
    const acts = built.binding.runForward(10);
    expect(acts.get("embed.out")!.shape).toEqual([11, dims.d_model]);
    expect(acts.get("layer0.attn.weights")!.shape).toEqual([dims.n_head, 11, 11]);
    expect(acts.get("head.logits")!.shape).toEqual([11, dims.vocab_size]);
  });

  it("runForward out of range / before setTokens throws", () => {
    expect(() => built.binding.runForward(GPT_SEQ_LEN)).toThrow(/out of range/);
    expect(() => built.binding.runForward(-1)).toThrow(/out of range/);
  });

  it("exposes all named anchors plus cameraHome", () => {
    for (const name of GPT_ANCHOR_NAMES) {
      const kf = built.anchors.get(name);
      expect(kf, `anchor "${name}"`).toBeDefined();
      expect([...kf!.pos, ...kf!.target].every(Number.isFinite)).toBe(true);
    }
    expect(built.cameraHome).toBe(built.anchors.get("home"));
  });

  it("setHighlight is idempotent; setDim(null) undims everything", () => {
    const view = built.views.get("embed.out")!;
    built.binding.setHighlight(["embed.out"], true);
    expect(view.material.uniforms.uHighlight.value).toBe(1);
    built.binding.setDim(["embed.out"]);
    expect(built.views.get("head.logits")!.material.uniforms.uDim.value).toBe(1);
    expect(view.material.uniforms.uDim.value).toBe(0);
    built.binding.setDim(null);
    expect(built.views.get("head.logits")!.material.uniforms.uDim.value).toBe(0);
  });

  it("dispose runs without throwing and detaches the scene root", () => {
    expect(() => built.dispose()).not.toThrow();
    expect(scene.children).toHaveLength(0);
    expect(() => built.dispose()).not.toThrow();
  });
});

describe("buildGptChapters (real scene + real i18n dicts)", () => {
  const { built } = build();
  const i18n = new I18n({ en, zh } satisfies Record<Locale, Record<string, string>>);
  const reg = buildGptChapters(built, i18n);

  const EXPECTED_IDS = ["intro", "embed", "attention", "mlp", "readout"];

  it("registers the 5 chapters in order", () => {
    expect(reg.count).toBe(EXPECTED_IDS.length);
    expect(EXPECTED_IDS.map((_, i) => reg.get(i).id)).toEqual(EXPECTED_IDS);
  });

  it("every chapter camera is a finite keyframe drawn from the scene anchors", () => {
    const anchorKfs = new Set([...built.anchors.values()]);
    for (let i = 0; i < reg.count; i++) {
      const ch = reg.get(i);
      expect([...ch.camera.pos, ...ch.camera.target].every(Number.isFinite), ch.id).toBe(true);
      expect(anchorKfs.has(ch.camera), ch.id).toBe(true);
    }
  });

  it("no GPT chapter has a timeline (attention is one-shot, not scanned)", () => {
    for (let i = 0; i < reg.count; i++) {
      expect(reg.get(i).timeline, reg.get(i).id).toBeUndefined();
    }
  });

  it("every sidebar title key exists in BOTH locales", () => {
    for (let i = 0; i < reg.count; i++) {
      const titleKey = reg.get(i).narrationKey.replace(/\.body$/, ".title");
      expect(i18n.missingLocales(titleKey), titleKey).toEqual([]);
    }
  });

  it("every focus-highlight names a real view", () => {
    for (let i = 0; i < reg.count; i++) {
      const ch = reg.get(i);
      for (const name of ch.highlights) {
        expect(built.views.has(name), `chapter "${ch.id}" highlight "${name}"`).toBe(true);
      }
    }
  });
});
