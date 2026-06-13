/**
 * Instance-budget guard (Task 23). The design doc caps a scene at ~2M cube
 * instances for 60 fps on Apple Silicon. This builds each real scene and
 * sums InstancedMesh.count across every TensorView (one mesh = one draw
 * call), asserting we stay well under budget and logging the measured
 * numbers that docs/plans/perf-notes.md cites. (Frame RATE can't be
 * measured here — headless runs use a software rasterizer; see perf-notes.)
 */

import { describe, expect, it } from "vitest";
import { readFileSync } from "node:fs";
import * as THREE from "three";
import { parseWeights, type Manifest } from "../compute/loader";
import { buildMambaScene } from "./mamba";
import { buildGptScene } from "./gpt";

const INSTANCE_BUDGET = 2_000_000;

const stubFactory = {
  label: () => new THREE.Group(),
  bracket: () => new THREE.Group(),
};

function load(arch: string): { manifest: Manifest; weights: ReturnType<typeof parseWeights> } {
  const dir = new URL(`../../public/models/${arch}/`, import.meta.url);
  const manifest = JSON.parse(readFileSync(new URL("manifest.json", dir), "utf8")) as Manifest;
  const raw = readFileSync(new URL("weights.bin", dir));
  const bin = raw.buffer.slice(raw.byteOffset, raw.byteOffset + raw.byteLength);
  return { manifest, weights: parseWeights(manifest, bin) };
}

function measure(name: string, built: { views: Map<string, { mesh: THREE.InstancedMesh }> }): number {
  let instances = 0;
  for (const view of built.views.values()) instances += view.mesh.count;
  // Logged so perf-notes.md numbers stay honest (re-run to refresh).
  console.log(`[perf] ${name}: ${built.views.size} draw calls, ${instances} cube instances`);
  return instances;
}

describe("scene instance budget", () => {
  it("mamba stays well under the 2M-instance budget", () => {
    const { manifest, weights } = load("mamba");
    const scene = new THREE.Scene();
    const built = buildMambaScene({ scene, weights, manifest, labelFactory: stubFactory });
    expect(measure("mamba", built)).toBeLessThan(INSTANCE_BUDGET);
    built.dispose();
  });

  it("gpt stays well under the 2M-instance budget", () => {
    const { manifest, weights } = load("gpt");
    const scene = new THREE.Scene();
    const built = buildGptScene({ scene, weights, manifest, labelFactory: stubFactory });
    expect(measure("gpt", built)).toBeLessThan(INSTANCE_BUDGET);
    built.dispose();
  });
});
