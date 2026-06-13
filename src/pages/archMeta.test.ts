import { describe, expect, it } from "vitest";
import { ARCH_VISUALS, archAccent, archGlyph } from "./archMeta";

const KNOWN = ["mamba", "gpt", "rwkv", "moe", "kan", "diffusion"];

describe("archMeta", () => {
  it("has a visual for every live architecture", () => {
    for (const id of KNOWN) expect(ARCH_VISUALS[id], id).toBeDefined();
  });

  it("accents are distinct valid hex colours", () => {
    const accents = KNOWN.map((id) => archAccent(id));
    for (const a of accents) expect(a).toMatch(/^#[0-9a-f]{6}$/i);
    expect(new Set(accents).size).toBe(KNOWN.length); // all distinct
  });

  it("glyphs are non-empty inline SVG using currentColor", () => {
    for (const id of KNOWN) {
      const g = archGlyph(id);
      expect(g, id).toMatch(/^<svg[\s\S]*<\/svg>$/);
      expect(g).toContain("currentColor");
    }
  });

  it("falls back gracefully for an unknown id", () => {
    expect(archAccent("nope")).toMatch(/^#[0-9a-f]{6}$/i);
    expect(archGlyph("nope")).toBe("");
  });
});
