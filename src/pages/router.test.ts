import { describe, expect, it } from "vitest";
import { resolveRoute } from "./router";

const ARCHS = ["mamba", "gpt"];

describe("resolveRoute", () => {
  it("maps a known arch", () => {
    expect(resolveRoute("#/mamba", ARCHS)).toEqual({ kind: "arch", arch: "mamba" });
    expect(resolveRoute("#/gpt", ARCHS)).toEqual({ kind: "arch", arch: "gpt" });
  });

  it("maps a known arch + chapter deep link", () => {
    expect(resolveRoute("#/gpt/attention", ARCHS)).toEqual({
      kind: "arch",
      arch: "gpt",
      chapterId: "attention",
    });
    expect(resolveRoute("#/mamba/scan", ARCHS)).toEqual({
      kind: "arch",
      arch: "mamba",
      chapterId: "scan",
    });
  });

  it("maps the compare route", () => {
    expect(resolveRoute("#/compare", ARCHS)).toEqual({ kind: "compare" });
  });

  it("does not treat #/compare/x as compare (no sub-route)", () => {
    expect(resolveRoute("#/compare/foo", ARCHS)).toEqual({ kind: "landing" });
  });

  it("lands for the root hash and empty string", () => {
    expect(resolveRoute("#/", ARCHS)).toEqual({ kind: "landing" });
    expect(resolveRoute("", ARCHS)).toEqual({ kind: "landing" });
    expect(resolveRoute("#", ARCHS)).toEqual({ kind: "landing" });
  });

  it("lands for an unimplemented arch (no dead routes)", () => {
    expect(resolveRoute("#/rwkv", ARCHS)).toEqual({ kind: "landing" });
    expect(resolveRoute("#/rwkv/intro", ARCHS)).toEqual({ kind: "landing" });
  });

  it("lands for malformed hashes", () => {
    expect(resolveRoute("#garbage", ARCHS)).toEqual({ kind: "landing" });
    expect(resolveRoute("#/mamba/scan/extra", ARCHS)).toEqual({ kind: "landing" });
    expect(resolveRoute("#/Mamba", ARCHS)).toEqual({ kind: "landing" }); // case-sensitive
  });
});
