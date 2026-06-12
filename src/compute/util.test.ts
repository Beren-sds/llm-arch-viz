import { describe, expect, it } from "vitest";
import { T } from "./tensor";
import { dimsFromManifest, sliceCols } from "./util";
import type { Manifest } from "./loader";

describe("sliceCols", () => {
  it("slices a column range of a 2D tensor", () => {
    const x = T.from([1, 2, 3, 4, 5, 6], [2, 3]);
    const out = sliceCols(x, 1, 3);
    expect(out.shape).toEqual([2, 2]);
    expect(Array.from(out.data)).toEqual([2, 3, 5, 6]);
  });

  it("throws on non-2D input", () => {
    expect(() => sliceCols(T.zeros([6]), 0, 2)).toThrow(/2D/);
    expect(() => sliceCols(T.zeros([2, 3, 4]), 0, 2)).toThrow(/2D/);
  });

  it("throws on out-of-range or inverted ranges", () => {
    const x = T.zeros([2, 3]);
    expect(() => sliceCols(x, -1, 2)).toThrow(/range/);
    expect(() => sliceCols(x, 0, 4)).toThrow(/range/);
    expect(() => sliceCols(x, 2, 1)).toThrow(/range/);
  });
});

describe("dimsFromManifest", () => {
  const manifest = {
    arch: "gpt",
    offset_unit: "float32",
    dims: { a: 1, b: 2 },
    checkpoint: {} as Manifest["checkpoint"],
    tensors: [],
  } as Manifest;

  it("returns the requested numeric dims", () => {
    expect(dimsFromManifest(manifest, "gpt", ["a", "b"])).toEqual({ a: 1, b: 2 });
  });

  it("throws on arch mismatch", () => {
    expect(() => dimsFromManifest(manifest, "mamba", ["a"])).toThrow(/arch/);
  });

  it("throws on a missing dim key", () => {
    expect(() => dimsFromManifest(manifest, "gpt", ["a", "c"])).toThrow(/"c"/);
  });
});
