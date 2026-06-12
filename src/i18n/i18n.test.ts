// @vitest-environment happy-dom
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { I18n, LOCALE_STORAGE_KEY, type Locale, validateDicts } from "./i18n";
import en from "./en.json";
import zh from "./zh.json";

function makeDicts(): Record<Locale, Record<string, string>> {
  return {
    en: { "ui.play": "Play", "ui.pause": "Pause", "en.only": "EN only" },
    zh: { "ui.play": "播放", "ui.pause": "暂停" },
  };
}

beforeEach(() => {
  localStorage.clear();
});

afterEach(() => {
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

describe("I18n.t", () => {
  it("returns the current-locale string", () => {
    const i = new I18n(makeDicts());
    expect(i.t("ui.play")).toBe("Play");
    i.setLocale("zh");
    expect(i.t("ui.play")).toBe("播放");
  });

  it("falls back to EN when the key is missing in the current locale, without console.error", () => {
    const err = vi.spyOn(console, "error").mockImplementation(() => {});
    const i = new I18n(makeDicts());
    i.setLocale("zh");
    expect(i.t("en.only")).toBe("EN only");
    expect(err).not.toHaveBeenCalled();
  });

  it("falls back to the key itself with console.error when missing everywhere", () => {
    const err = vi.spyOn(console, "error").mockImplementation(() => {});
    const i = new I18n(makeDicts());
    expect(i.t("no.such.key")).toBe("no.such.key");
    expect(err).toHaveBeenCalledTimes(1);
    expect(String(err.mock.calls[0])).toContain("no.such.key");
  });

  it("does not resolve Object.prototype keys as translations", () => {
    const err = vi.spyOn(console, "error").mockImplementation(() => {});
    const i = new I18n(makeDicts());
    // 'toString'/'constructor' exist on the prototype of a plain object
    // literal; only own keys are translations.
    expect(i.t("toString")).toBe("toString");
    expect(i.t("constructor")).toBe("constructor");
    expect(err).toHaveBeenCalledTimes(2);
  });
});

describe("I18n locale + persistence", () => {
  it("defaults to 'en' with nothing stored", () => {
    expect(new I18n(makeDicts()).locale).toBe("en");
  });

  it("reads a persisted locale from localStorage", () => {
    localStorage.setItem(LOCALE_STORAGE_KEY, "zh");
    expect(new I18n(makeDicts()).locale).toBe("zh");
  });

  it("treats an invalid stored value as 'en'", () => {
    localStorage.setItem(LOCALE_STORAGE_KEY, "fr");
    expect(new I18n(makeDicts()).locale).toBe("en");
  });

  it("setLocale persists to localStorage", () => {
    const i = new I18n(makeDicts());
    i.setLocale("zh");
    expect(localStorage.getItem(LOCALE_STORAGE_KEY)).toBe("zh");
    expect(i.locale).toBe("zh");
  });

  it("survives a non-browser environment (no localStorage)", () => {
    vi.stubGlobal("localStorage", undefined);
    const i = new I18n(makeDicts());
    expect(i.locale).toBe("en");
    expect(() => i.setLocale("zh")).not.toThrow();
    expect(i.locale).toBe("zh");
  });
});

describe("I18n.onChange", () => {
  it("fires with the new locale on change", () => {
    const i = new I18n(makeDicts());
    const cb = vi.fn();
    i.onChange(cb);
    i.setLocale("zh");
    expect(cb).toHaveBeenCalledTimes(1);
    expect(cb).toHaveBeenCalledWith("zh");
  });

  it("does not fire when setLocale is a no-op (same locale)", () => {
    const i = new I18n(makeDicts());
    const cb = vi.fn();
    i.onChange(cb);
    i.setLocale("en");
    expect(cb).not.toHaveBeenCalled();
  });

  it("unsubscribe stops notifications (other listeners unaffected)", () => {
    const i = new I18n(makeDicts());
    const a = vi.fn();
    const b = vi.fn();
    const offA = i.onChange(a);
    i.onChange(b);
    offA();
    i.setLocale("zh");
    expect(a).not.toHaveBeenCalled();
    expect(b).toHaveBeenCalledTimes(1);
  });
});

describe("I18n.missingLocales", () => {
  it("treats Object.prototype keys as missing from every locale", () => {
    const i = new I18n(makeDicts());
    expect(i.missingLocales("toString")).toEqual(["en", "zh"]);
    expect(i.missingLocales("ui.play")).toEqual([]);
  });
});

describe("validateDicts", () => {
  it("returns [] when both locales have the same keys", () => {
    expect(
      validateDicts({ en: { a: "1", b: "2" }, zh: { a: "一", b: "二" } }),
    ).toEqual([]);
  });

  it("catches a key present only in en", () => {
    expect(validateDicts({ en: { a: "1", b: "2" }, zh: { a: "一" } })).toEqual(["b"]);
  });

  it("catches a key present only in zh", () => {
    expect(validateDicts({ en: { a: "1" }, zh: { a: "一", c: "三" } })).toEqual(["c"]);
  });

  it("reports one-sided keys from both directions, sorted", () => {
    expect(
      validateDicts({ en: { z: "1", a: "2" }, zh: { a: "一", m: "中" } }),
    ).toEqual(["m", "z"]);
  });

  it("is not fooled by Object.prototype keys (own-key check, not lookup)", () => {
    // 'toString' exists on zh's prototype; a plain `zh['toString']` lookup
    // would wrongly count it as translated.
    expect(validateDicts({ en: { toString: "To string" }, zh: {} })).toEqual(["toString"]);
  });
});

describe("real en.json / zh.json", () => {
  it("key sets are identical", () => {
    expect(validateDicts({ en, zh })).toEqual([]);
    expect(Object.keys(zh).sort()).toEqual(Object.keys(en).sort());
  });

  it("contains the seeded UI keys", () => {
    const required = [
      "ui.play",
      "ui.pause",
      "ui.next",
      "ui.prev",
      "ui.chapters",
      "ui.langToggle",
      "ui.tokenStep",
      "ui.loading",
      "ui.loadError",
      "ui.retry",
      "ui.webglError",
    ];
    for (const key of required) {
      expect(en, `en.json missing ${key}`).toHaveProperty([key]);
      expect(zh, `zh.json missing ${key}`).toHaveProperty([key]);
    }
  });

  it("has no empty translations", () => {
    for (const [k, v] of [...Object.entries(en), ...Object.entries(zh)]) {
      expect(v.trim(), `empty translation for ${k}`).not.toBe("");
    }
  });
});
