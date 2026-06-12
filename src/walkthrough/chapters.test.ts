import { describe, expect, it } from "vitest";
import { I18n, type Locale } from "../i18n/i18n";
import { type Chapter, ChapterRegistry, parseHash, toHash } from "./chapters";
import type { TimelineSpec } from "./timeline";

function makeI18n(
  en: Record<string, string> = {
    "ch.embed": "Embedding",
    "ch.attn": "Attention",
    "ch.scan": "SSM scan",
  },
  zh: Record<string, string> = {
    "ch.embed": "嵌入",
    "ch.attn": "注意力",
    "ch.scan": "SSM 扫描",
  },
): I18n {
  return new I18n({ en, zh } satisfies Record<Locale, Record<string, string>>);
}

function kf(x: number) {
  return {
    pos: [x, 0, 0] as [number, number, number],
    target: [0, 0, 0] as [number, number, number],
  };
}

function makeChapters(): Chapter[] {
  return [
    { id: "embed", camera: kf(1), highlights: ["tok"], narrationKey: "ch.embed" },
    { id: "attn", camera: kf(2), highlights: ["qk", "v"], narrationKey: "ch.attn" },
    { id: "scan", camera: kf(3), highlights: [], narrationKey: "ch.scan" },
  ];
}

describe("ChapterRegistry construction", () => {
  it("accepts a valid chapter list", () => {
    expect(() => new ChapterRegistry("mamba", makeChapters(), makeI18n())).not.toThrow();
  });

  it("throws on duplicate chapter ids", () => {
    const chapters = makeChapters();
    chapters[2] = { ...chapters[2], id: "embed" };
    expect(() => new ChapterRegistry("mamba", chapters, makeI18n())).toThrow(/embed/);
  });

  it("throws on an empty chapter list", () => {
    expect(() => new ChapterRegistry("mamba", [], makeI18n())).toThrow(/empty|no chapters/i);
  });

  it("throws when a narrationKey is missing from BOTH locales", () => {
    const chapters = makeChapters();
    chapters[0] = { ...chapters[0], narrationKey: "ch.ghost" };
    expect(() => new ChapterRegistry("mamba", chapters, makeI18n())).toThrow(/ch\.ghost/);
  });

  it("throws when a narrationKey exists in en but not zh", () => {
    const i18n = makeI18n(
      { "ch.embed": "Embedding", "ch.attn": "Attention", "ch.scan": "SSM scan" },
      { "ch.embed": "嵌入", "ch.attn": "注意力" }, // ch.scan untranslated
    );
    expect(() => new ChapterRegistry("mamba", makeChapters(), i18n)).toThrow(/ch\.scan.*zh|zh.*ch\.scan/s);
  });

  it("throws when a narrationKey exists in zh but not en", () => {
    const i18n = makeI18n(
      { "ch.embed": "Embedding", "ch.attn": "Attention" }, // ch.scan missing
      { "ch.embed": "嵌入", "ch.attn": "注意力", "ch.scan": "SSM 扫描" },
    );
    expect(() => new ChapterRegistry("mamba", makeChapters(), i18n)).toThrow(/ch\.scan.*en|en.*ch\.scan/s);
  });
});

describe("ChapterRegistry access", () => {
  const reg = new ChapterRegistry("mamba", makeChapters(), makeI18n());

  it("count reflects the chapter list", () => {
    expect(reg.count).toBe(3);
  });

  it("get(idx) returns chapters in order", () => {
    expect(reg.get(0).id).toBe("embed");
    expect(reg.get(2).id).toBe("scan");
  });

  it("get(idx) throws on out-of-range or non-integer indices", () => {
    expect(() => reg.get(-1)).toThrow(RangeError);
    expect(() => reg.get(3)).toThrow(RangeError);
    expect(() => reg.get(1.5)).toThrow(RangeError);
  });

  it("byId finds chapters; unknown id → undefined", () => {
    expect(reg.byId("attn")?.narrationKey).toBe("ch.attn");
    expect(reg.byId("nope")).toBeUndefined();
  });

  it("indexOf maps ids to positions; unknown id → -1", () => {
    expect(reg.indexOf("embed")).toBe(0);
    expect(reg.indexOf("scan")).toBe(2);
    expect(reg.indexOf("nope")).toBe(-1);
  });

  it("exposes the scene name", () => {
    expect(reg.scene).toBe("mamba");
  });

  it("carries an optional typed TimelineSpec per chapter", () => {
    const timeline: TimelineSpec = {
      steps: [{ kind: "stepToken", token: 0, durationMs: 500 }],
      loop: true,
    };
    const chapters = makeChapters();
    chapters[0] = { ...chapters[0], timeline };
    const withTl = new ChapterRegistry("mamba", chapters, makeI18n());
    expect(withTl.get(0).timeline).toEqual(timeline);
    expect(withTl.get(1).timeline).toBeUndefined();
  });
});

describe("parseHash", () => {
  it("parses '#/scene'", () => {
    expect(parseHash("#/mamba")).toEqual({ scene: "mamba" });
  });

  it("parses '#/scene/chapter'", () => {
    expect(parseHash("#/mamba/ssm-scan")).toEqual({ scene: "mamba", chapterId: "ssm-scan" });
  });

  it("rejects garbage", () => {
    for (const bad of [
      "",
      "#",
      "#/",
      "#//",
      "#//scan",
      "#/mamba/",
      "#/mamba/a/b",
      "mamba/scan",
      "/mamba/scan",
      "#mamba",
      "#/ma mba",
      "#/mamba/sc an",
    ]) {
      expect(parseHash(bad), `should reject ${JSON.stringify(bad)}`).toBeNull();
    }
  });
});

describe("toHash", () => {
  it("builds '#/scene' and '#/scene/chapter'", () => {
    expect(toHash("mamba")).toBe("#/mamba");
    expect(toHash("mamba", "ssm-scan")).toBe("#/mamba/ssm-scan");
  });

  it("round-trips through parseHash (both forms)", () => {
    const a = parseHash(toHash("gpt2", "attn-heads"));
    expect(a).toEqual({ scene: "gpt2", chapterId: "attn-heads" });
    const b = parseHash(toHash("gpt2"));
    expect(b).toEqual({ scene: "gpt2" });
    // and the reverse direction: parse → rebuild → same string
    expect(toHash(a!.scene, a!.chapterId)).toBe("#/gpt2/attn-heads");
    expect(toHash(b!.scene, b!.chapterId)).toBe("#/gpt2");
  });
});
