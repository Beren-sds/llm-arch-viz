/**
 * Smoke test for the Mamba scene layout, against the REAL committed
 * manifest + weights (like the golden tests): every recorded activation
 * and every weight tensor must have a view; setTokens pushes recorder
 * values into the views; the binding's runForward is an INCLUSIVE prefix.
 *
 * Runs in node — label/bracket creation is stubbed (troika's sync()
 * needs a browser-like global; see LabelFactory in mamba.ts).
 */

import { describe, expect, it } from "vitest";
import { readFileSync } from "node:fs";
import * as THREE from "three";
import { parseWeights, type Manifest } from "../compute/loader";
import { mambaDimsFrom, mambaForward } from "../compute/mamba";
import { MapRecorder } from "../compute/recorder";
import type { GoldenFile } from "../compute/golden";
import { buildMambaScene, MAMBA_ANCHOR_NAMES, MAMBA_SEQ_LEN } from "./mamba";

function loadArtifacts(): { manifest: Manifest; weights: ReturnType<typeof parseWeights> } {
  const dir = new URL("../../public/models/mamba/", import.meta.url);
  const manifest = JSON.parse(readFileSync(new URL("manifest.json", dir), "utf8")) as Manifest;
  const raw = readFileSync(new URL("weights.bin", dir));
  const bin = raw.buffer.slice(raw.byteOffset, raw.byteOffset + raw.byteLength);
  return { manifest, weights: parseWeights(manifest, bin) };
}

function goldenTokens(): number[] {
  const url = new URL("../../goldens/mamba/goldens.json", import.meta.url);
  const goldens = JSON.parse(readFileSync(url, "utf8")) as GoldenFile;
  return goldens.inputs[0].tokens;
}

/** No-GL label stubs: plain Object3D stand-ins for troika text/brackets. */
const stubFactory = {
  label: () => new THREE.Group(),
  bracket: () => new THREE.Group(),
};

function build() {
  const { manifest, weights } = loadArtifacts();
  const scene = new THREE.Scene();
  const built = buildMambaScene({ scene, weights, manifest, labelFactory: stubFactory });
  return { manifest, weights, scene, built };
}

describe("buildMambaScene (real manifest + weights)", () => {
  const { manifest, weights, scene, built } = build();
  const dims = mambaDimsFrom(manifest);
  const tokens = goldenTokens();

  // Reference forward at full length, recorded with the real recorder.
  const rec = new MapRecorder();
  mambaForward(weights, dims, tokens, rec);
  const actNames = [...rec.activations.keys()];

  it("has a TensorView for every activation a real forward records", () => {
    expect(tokens).toHaveLength(MAMBA_SEQ_LEN);
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
    // 61 activations (1 + 2·(8 + 21) + 2) + 23 weights = 84, counted from
    // the real run rather than hardcoded.
    expect(built.views.size).toBe(actNames.length + manifest.tensors.length);
  });

  it("weight views carry their weight values (conv kernels squeezed)", () => {
    const aLog = built.views.get("blocks.0.A_log")!;
    expect(aLog.shape).toEqual([96, 8]);
    expect(Array.from(aLog.lastValues)).toEqual(Array.from(weights.get("blocks.0.A_log")!.data));
    const conv = built.views.get("blocks.0.conv1d.weight")!;
    expect(conv.shape).toEqual([96, 4]); // (96, 1, 4) squeezed for display
    expect(Array.from(conv.lastValues)).toEqual(
      Array.from(weights.get("blocks.0.conv1d.weight")!.data),
    );
  });

  it("setTokens(golden tokens) applies the recorder's activations to the views", () => {
    built.setTokens(tokens);
    for (const name of ["embed.out", "layer1.gate.out", "head.logits"]) {
      const view = built.views.get(name)!;
      expect(Array.from(view.lastValues), name).toEqual(
        Array.from(rec.activations.get(name)!.data),
      );
    }
  });

  it("after a full setTokens, the visible h snapshot is the final token's", () => {
    expect(built.views.get("layer0.ssm.h.t20")!.mesh.visible).toBe(true);
    expect(built.views.get("layer0.ssm.h.t0")!.mesh.visible).toBe(false);
    expect(built.views.get("layer1.ssm.h.t20")!.mesh.visible).toBe(true);
  });

  it("binding.runForward(10) is INCLUSIVE: an 11-token prefix with T=11 shapes", () => {
    const acts = built.binding.runForward(10);
    expect(acts.get("embed.out")!.shape).toEqual([11, dims.d_model]);
    expect(acts.get("layer0.conv.out")!.shape).toEqual([11, dims.expand * dims.d_model]);
    expect(acts.get("head.logits")!.shape).toEqual([11, dims.vocab_size]);
    expect(acts.has("layer0.ssm.h.t10")).toBe(true);
    expect(acts.has("layer0.ssm.h.t11")).toBe(false);
  });

  it("applyActivations on a prefix zero-pads future rows and shows the latest h", () => {
    built.binding.applyActivations(built.binding.runForward(10));
    const embed = built.views.get("embed.out")!;
    const full = rec.activations.get("embed.out")!;
    // First 11 rows match the full run (embedding is per-token), rest are 0.
    const cols = dims.d_model;
    expect(Array.from(embed.lastValues.slice(0, 11 * cols))).toEqual(
      Array.from(full.data.slice(0, 11 * cols)),
    );
    expect(Array.from(embed.lastValues.slice(11 * cols)).every((v) => v === 0)).toBe(true);
    expect(built.views.get("layer0.ssm.h.t10")!.mesh.visible).toBe(true);
    expect(built.views.get("layer0.ssm.h.t20")!.mesh.visible).toBe(false);
    // restore full state for any later assertions
    built.setTokens(tokens);
  });

  it("runForward out of range / before setTokens throws", () => {
    expect(() => built.binding.runForward(MAMBA_SEQ_LEN)).toThrow(/out of range/);
    expect(() => built.binding.runForward(-1)).toThrow(/out of range/);
  });

  it("exposes all named anchors plus cameraHome", () => {
    for (const name of MAMBA_ANCHOR_NAMES) {
      const kf = built.anchors.get(name);
      expect(kf, `anchor "${name}"`).toBeDefined();
      expect([...kf!.pos, ...kf!.target].every(Number.isFinite)).toBe(true);
    }
    expect(built.cameraHome).toBe(built.anchors.get("home"));
  });

  it("setHighlight is idempotent; setDim(null) undims everything", () => {
    const view = built.views.get("embed.out")!;
    built.binding.setHighlight(["embed.out"], true);
    built.binding.setHighlight(["embed.out"], true);
    expect(view.material.uniforms.uHighlight.value).toBe(1);
    built.binding.setHighlight(["embed.out"], false);
    expect(view.material.uniforms.uHighlight.value).toBe(0);

    built.binding.setDim(["embed.out"]);
    expect(built.views.get("head.logits")!.material.uniforms.uDim.value).toBe(1);
    expect(view.material.uniforms.uDim.value).toBe(0);
    built.binding.setDim(null);
    expect(built.views.get("head.logits")!.material.uniforms.uDim.value).toBe(0);
  });

  it("dispose runs without throwing and detaches the scene root", () => {
    expect(() => built.dispose()).not.toThrow();
    expect(scene.children).toHaveLength(0);
    expect(() => built.dispose()).not.toThrow(); // idempotent
  });
});
