// @vitest-environment happy-dom

import { describe, expect, it, vi } from "vitest";
import { I18n, type Locale } from "../i18n/i18n";
import en from "../i18n/en.json";
import zh from "../i18n/zh.json";
import { createComparePage, formatParams, totalParams } from "./compare";

function i18n(): I18n {
  return new I18n({ en, zh } satisfies Record<Locale, Record<string, string>>);
}

const fakeManifest = (params: number) => ({ tensors: [{ name: "w", shape: [params] }] });

/** A fetch stub mapping arch id → manifest param count (or a thrown error). */
function fetchStub(counts: Record<string, number | "fail">): typeof fetch {
  return (async (url: string) => {
    const id = String(url).match(/models\/([^/]+)\/manifest\.json/)?.[1] ?? "";
    const c = counts[id];
    if (c === undefined || c === "fail") return { ok: false, status: 404 } as Response;
    return { ok: true, status: 200, json: async () => fakeManifest(c) } as unknown as Response;
  }) as unknown as typeof fetch;
}

describe("compare pure helpers", () => {
  it("totalParams sums shape products across tensors", () => {
    expect(totalParams({ tensors: [{ shape: [4, 6] }, { shape: [10] }] })).toBe(34);
  });

  it("formatParams is compact", () => {
    expect(formatParams(512)).toBe("512");
    expect(formatParams(61234)).toBe("61K");
    expect(formatParams(1250)).toBe("1.3K");
    expect(formatParams(1_250_000)).toBe("1.3M");
  });
});

describe("createComparePage (happy-dom)", () => {
  it("renders a row per arch, grouping recurrences (mamba/rwkv) first", () => {
    const container = document.createElement("div");
    createComparePage({
      container,
      i18n: i18n(),
      baseUrl: "/base/",
      archs: ["gpt", "mamba"],
      onOpen: () => {},
      onHome: () => {},
      fetchFn: fetchStub({ gpt: 1000, mamba: 2000 }),
    });
    const rows = container.querySelectorAll(".compare-row");
    expect(rows).toHaveLength(2);
    // Recurrence (mamba) is grouped ahead of attention (gpt).
    const opens = [...container.querySelectorAll(".compare-open")].map((e) => e.textContent);
    expect(opens).toEqual([en["card.mamba.title"], en["card.gpt.title"]]);
    expect(container.textContent).toContain(en["compare.gpt.cost"]);
  });

  it("fills live parameter counts (grouped order) and scales bars", async () => {
    const container = document.createElement("div");
    const page = createComparePage({
      container,
      i18n: i18n(),
      baseUrl: "/base/",
      archs: ["gpt", "mamba"],
      onOpen: () => {},
      onHome: () => {},
      fetchFn: fetchStub({ gpt: 61234, mamba: 2000 }),
    });
    await page.ready;
    // Grouped order is mamba, then gpt.
    const vals = [...container.querySelectorAll(".compare-param-val")].map((e) => e.textContent);
    expect(vals).toEqual(["2.0K", "61K"]);
    // Bars are scaled to the largest model (gpt = 100%, mamba ≈ 3%).
    const bars = [...container.querySelectorAll(".compare-bar-fill")].map(
      (e) => (e as HTMLElement).style.width,
    );
    expect(bars[1]).toBe("100%");
    expect(bars[0]).toBe("3%");
  });

  it("shows — for a failed manifest fetch (never a fake number)", async () => {
    const container = document.createElement("div");
    const errSpy = vi.spyOn(console, "error").mockImplementation(() => {});
    const page = createComparePage({
      container,
      i18n: i18n(),
      baseUrl: "/base/",
      archs: ["gpt"],
      onOpen: () => {},
      onHome: () => {},
      fetchFn: fetchStub({ gpt: "fail" }),
    });
    await page.ready;
    expect(container.querySelector(".compare-param-val")?.textContent).toBe("—");
    errSpy.mockRestore();
  });

  it("open + home cells fire their callbacks", () => {
    const container = document.createElement("div");
    const onOpen = vi.fn();
    const onHome = vi.fn();
    createComparePage({
      container,
      i18n: i18n(),
      baseUrl: "/base/",
      archs: ["gpt", "mamba"],
      onOpen,
      onHome,
      fetchFn: fetchStub({ gpt: 1000, mamba: 2000 }),
    });
    // First row is the grouped-first recurrence (mamba).
    container.querySelector<HTMLButtonElement>(".compare-open")!.click();
    expect(onOpen).toHaveBeenCalledWith("mamba");
    container.querySelector<HTMLButtonElement>(".viz-home")!.click();
    expect(onHome).toHaveBeenCalled();
  });
});
