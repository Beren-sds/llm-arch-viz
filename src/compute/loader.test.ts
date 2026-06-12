import { describe, expect, it } from "vitest";
import { readFileSync } from "node:fs";
import { loadModel, parseWeights, type Manifest, type TensorEntry } from "./loader";

/**
 * Hand-built fixture: 3 tiny tensors written little-endian via DataView,
 * mirroring the export contract (offsets/lengths in float32 elements,
 * tensors contiguous in array order starting at 0).
 */
const FIXTURE_VALUES: Record<string, number[]> = {
  a: [1, 2.5, -3, 0.5],
  b: [10, 20, 30],
  c: [-1.5, 7],
};

function makeFixtureManifest(): Manifest {
  return {
    arch: "test",
    offset_unit: "float32",
    dims: { d_model: 2 },
    checkpoint: {
      run_dir: "outputs/test/fake",
      step: 1,
      val_exact: 1.0,
      golden_seed: 0,
      golden_candidates_skipped: 0,
    },
    tensors: [
      { name: "a", shape: [2, 2], offset: 0, length: 4 },
      { name: "b", shape: [3], offset: 4, length: 3 },
      { name: "c", shape: [2, 1], offset: 7, length: 2 },
    ],
  };
}

function makeFixtureBin(): ArrayBuffer {
  const floats = [...FIXTURE_VALUES.a, ...FIXTURE_VALUES.b, ...FIXTURE_VALUES.c];
  const buf = new ArrayBuffer(4 * floats.length);
  const view = new DataView(buf);
  for (let i = 0; i < floats.length; i++) {
    view.setFloat32(4 * i, floats[i], true); // explicit little-endian
  }
  return buf;
}

describe("parseWeights", () => {
  it("parses a hand-built LE fixture into correct shapes and values", () => {
    const weights = parseWeights(makeFixtureManifest(), makeFixtureBin());
    expect(weights.size).toBe(3);
    const a = weights.get("a")!;
    expect(a.shape).toEqual([2, 2]);
    expect(Array.from(a.data)).toEqual(FIXTURE_VALUES.a);
    expect(a.at(0, 1)).toBe(2.5);
    const b = weights.get("b")!;
    expect(b.shape).toEqual([3]);
    expect(Array.from(b.data)).toEqual(FIXTURE_VALUES.b);
    const c = weights.get("c")!;
    expect(c.shape).toEqual([2, 1]);
    expect(Array.from(c.data)).toEqual(FIXTURE_VALUES.c);
  });

  it("returns tensors backed by copies, not views of the input buffer", () => {
    const bin = makeFixtureBin();
    const weights = parseWeights(makeFixtureManifest(), bin);
    new DataView(bin).setFloat32(0, 999, true);
    expect(weights.get("a")!.data[0]).toBe(1);
  });

  it("throws on byte-length mismatch with both sizes in the message", () => {
    const manifest = makeFixtureManifest();
    const shortBin = makeFixtureBin().slice(0, 20);
    expect(() => parseWeights(manifest, shortBin)).toThrow(/36/);
    expect(() => parseWeights(manifest, shortBin)).toThrow(/20/);
    const longBin = new ArrayBuffer(40);
    expect(() => parseWeights(manifest, longBin)).toThrow(/36/);
    expect(() => parseWeights(manifest, longBin)).toThrow(/40/);
  });

  it("throws on an offset gap between consecutive tensors", () => {
    const manifest = makeFixtureManifest();
    // b should start at 4; a gap of 1 float leaves [4,5) unclaimed.
    manifest.tensors[1] = { ...manifest.tensors[1], offset: 5 };
    expect(() => parseWeights(manifest, makeFixtureBin())).toThrow(/offset/);
  });

  it("throws on overlapping tensors", () => {
    const manifest = makeFixtureManifest();
    manifest.tensors[1] = { ...manifest.tensors[1], offset: 3 };
    expect(() => parseWeights(manifest, makeFixtureBin())).toThrow(/offset/);
  });

  it("throws when the first tensor does not start at offset 0", () => {
    const manifest = makeFixtureManifest();
    manifest.tensors[0] = { ...manifest.tensors[0], offset: 1 };
    expect(() => parseWeights(manifest, makeFixtureBin())).toThrow(/offset/);
  });

  it("throws when shape product != length", () => {
    const manifest = makeFixtureManifest();
    // Keep offsets/lengths contiguous so only the shape check can fire.
    manifest.tensors[1] = { ...manifest.tensors[1], shape: [4] };
    expect(() => parseWeights(manifest, makeFixtureBin())).toThrow(/shape/);
  });

  it("throws on an unknown offset_unit instead of guessing", () => {
    const manifest = { ...makeFixtureManifest(), offset_unit: "bytes" } as unknown as Manifest;
    expect(() => parseWeights(manifest, makeFixtureBin())).toThrow(/offset_unit/);
  });
});

describe("loadModel", () => {
  function fakeFetchOk(): typeof fetch {
    const manifest = makeFixtureManifest();
    const bin = makeFixtureBin();
    return (async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url.endsWith("manifest.json")) {
        return { ok: true, status: 200, json: async () => manifest } as Response;
      }
      if (url.endsWith("weights.bin")) {
        return { ok: true, status: 200, arrayBuffer: async () => bin } as Response;
      }
      return { ok: false, status: 404 } as Response;
    }) as typeof fetch;
  }

  it("fetches manifest + bin and returns parsed weights (happy path)", async () => {
    const { manifest, weights } = await loadModel("test", "/base", fakeFetchOk());
    expect(manifest.arch).toBe("test");
    expect(weights.size).toBe(3);
    expect(Array.from(weights.get("b")!.data)).toEqual(FIXTURE_VALUES.b);
  });

  it("requests the documented URL layout", async () => {
    const seen: string[] = [];
    const manifest = makeFixtureManifest();
    const bin = makeFixtureBin();
    const fetchFn = (async (input: RequestInfo | URL) => {
      seen.push(String(input));
      return {
        ok: true,
        status: 200,
        json: async () => manifest,
        arrayBuffer: async () => bin,
      } as Response;
    }) as typeof fetch;
    await loadModel("mamba", "https://example.test", fetchFn);
    expect(seen).toEqual([
      "https://example.test/models/mamba/manifest.json",
      "https://example.test/models/mamba/weights.bin",
    ]);
  });

  it("throws on non-OK response with status and url in the message", async () => {
    const fetchFn = (async () => ({ ok: false, status: 404 }) as Response) as typeof fetch;
    await expect(loadModel("nope", "/base", fetchFn)).rejects.toThrow(
      /404.*\/base\/models\/nope\/manifest\.json|\/base\/models\/nope\/manifest\.json.*404/,
    );
  });

  it("propagates network errors from fetch", async () => {
    const fetchFn = (async () => {
      throw new TypeError("network down");
    }) as typeof fetch;
    await expect(loadModel("test", "/base", fetchFn)).rejects.toThrow("network down");
  });
});

describe("real export artifacts (read from disk)", () => {
  function loadArtifacts(arch: string): { manifest: Manifest; bin: ArrayBuffer } {
    const dir = new URL(`../../public/models/${arch}/`, import.meta.url);
    const manifest = JSON.parse(
      readFileSync(new URL("manifest.json", dir), "utf8"),
    ) as Manifest;
    const raw = readFileSync(new URL("weights.bin", dir));
    // Node Buffers come from a shared pool; slice to the exact byte range.
    const bin = raw.buffer.slice(raw.byteOffset, raw.byteOffset + raw.byteLength);
    return { manifest, bin };
  }

  // Expected values below were derived by reading the manifests and
  // unpacking weights.bin with python struct ('<f' at 4*offset), e.g.:
  //   struct.unpack_from('<f', bin, 4*tensor['offset'])[0]
  // for the named tensor/element. They are the exact f64 renderings of
  // the stored f32 values, so toBe (exact) comparison is valid.

  it("parses public/models/mamba: 23 tensors, spot-checked values", () => {
    const { manifest, bin } = loadArtifacts("mamba");
    const weights = parseWeights(manifest, bin);
    expect(manifest.arch).toBe("mamba");
    expect(weights.size).toBe(23);
    // blocks.0.A_log is the first tensor (offset 0); element [0,0].
    expect(weights.get("blocks.0.A_log")!.shape).toEqual([96, 8]);
    expect(weights.get("blocks.0.A_log")!.at(0, 0)).toBe(-0.05882740393280983);
    // embedding.weight[0,0]
    expect(weights.get("embedding.weight")!.at(0, 0)).toBe(0.07403270155191422);
    // norms.1.weight is the last tensor; first and last elements.
    const n1 = weights.get("norms.1.weight")!;
    expect(n1.shape).toEqual([48]);
    expect(n1.at(0)).toBe(1.0419590473175049);
    expect(n1.at(47)).toBe(1.0794402360916138);
  });

  it("parses public/models/gpt: 29 tensors, spot-checked values", () => {
    const { manifest, bin } = loadArtifacts("gpt");
    const weights = parseWeights(manifest, bin);
    expect(manifest.arch).toBe("gpt");
    expect(weights.size).toBe(29);
    // attns.0.out_proj.bias is the first tensor (offset 0); element [0].
    expect(weights.get("attns.0.out_proj.bias")!.at(0)).toBe(0.09712594002485275);
    // pos_embedding.weight[0,0]
    expect(weights.get("pos_embedding.weight")!.at(0, 0)).toBe(-0.5447865724563599);
    // tok_embedding.weight is the last tensor; first and last elements.
    const tok = weights.get("tok_embedding.weight")!;
    expect(tok.shape).toEqual([16, 48]);
    expect(tok.at(0, 0)).toBe(0.07384829968214035);
    expect(tok.at(15, 47)).toBe(-1.0171412229537964);
  });

  it("rejects a truncated real weights.bin instead of silently parsing", () => {
    const { manifest, bin } = loadArtifacts("mamba");
    expect(() => parseWeights(manifest, bin.slice(0, bin.byteLength - 4))).toThrow(
      /byte/i,
    );
  });
});

// Compile-time check: TensorEntry stays structurally usable on its own.
const _entryCheck: TensorEntry = { name: "x", shape: [1], offset: 0, length: 1 };
void _entryCheck;
