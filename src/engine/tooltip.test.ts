// @vitest-environment happy-dom

import { afterEach, beforeEach, describe, expect, it } from "vitest";
import type { PickResult } from "./picking";
import { clampToViewport, formatValue, Tooltip, TOOLTIP_OFFSET } from "./tooltip";
import { TensorView } from "./tensorView";

function makeResult(overrides: Partial<PickResult> = {}): PickResult {
  const view = new TensorView("q_proj", [4, 6], { cellSize: 1, gap: 0, origin: [0, 0, 0] });
  return { view, name: "q_proj", indices: [2, 3], value: 1.23456789, ...overrides };
}

describe("clampToViewport", () => {
  it("offsets from the cursor when there is room", () => {
    expect(clampToViewport(100, 100, 200, 80, 1280, 720, 12)).toEqual({ x: 112, y: 112 });
  });

  it("flips to the left of the cursor at the right edge", () => {
    // 1250 + 12 + 200 > 1280 -> flip: 1250 - 12 - 200 = 1038
    expect(clampToViewport(1250, 100, 200, 80, 1280, 720, 12)).toEqual({ x: 1038, y: 112 });
  });

  it("flips above the cursor at the bottom edge", () => {
    // 700 + 12 + 80 > 720 -> flip: 700 - 12 - 80 = 608
    expect(clampToViewport(100, 700, 200, 80, 1280, 720, 12)).toEqual({ x: 112, y: 608 });
  });

  it("clamps to 0 when the flipped position would be negative", () => {
    // viewport 310 wide, tooltip 300: neither side fits -> pin at 0
    expect(clampToViewport(5, 100, 300, 80, 310, 720, 12).x).toBe(0);
  });

  it("never exceeds viewport - size even for oversized tooltips", () => {
    const { x, y } = clampToViewport(400, 300, 900, 800, 800, 600, 12);
    expect(x).toBe(0);
    expect(y).toBe(0);
  });
});

describe("formatValue", () => {
  it("formats finite values with toPrecision(4)", () => {
    expect(formatValue(1.23456789)).toBe("1.235");
    expect(formatValue(-0.000123456)).toBe("-0.0001235");
    expect(formatValue(0)).toBe("0.000");
  });

  it("formats -Infinity as masked", () => {
    expect(formatValue(-Infinity)).toBe("−∞ (masked)");
  });

  it("formats NaN as the literal string", () => {
    expect(formatValue(NaN)).toBe("NaN");
  });
});

describe("Tooltip DOM", () => {
  let container: HTMLDivElement;
  let tooltip: Tooltip;

  beforeEach(() => {
    container = document.createElement("div");
    document.body.appendChild(container);
    tooltip = new Tooltip(container);
  });

  afterEach(() => {
    tooltip.dispose();
    container.remove();
  });

  function root(): HTMLElement {
    const el = container.querySelector<HTMLElement>(".tensor-tooltip");
    expect(el).not.toBeNull();
    return el!;
  }

  it("mounts hidden", () => {
    expect(root().classList.contains("visible")).toBe(false);
  });

  it("show() populates name, indices, value and becomes visible", () => {
    tooltip.show(makeResult(), 100, 100);
    const el = root();
    expect(el.classList.contains("visible")).toBe(true);
    expect(el.querySelector(".tensor-tooltip-name")!.textContent).toBe("q_proj");
    expect(el.querySelector(".tensor-tooltip-indices")!.textContent).toBe("[2, 3]");
    expect(el.querySelector(".tensor-tooltip-value")!.textContent).toBe("1.235");
  });

  it("show() renders the formula line when present, hides it when absent", () => {
    tooltip.show(makeResult({ formula: "h = exp(Δ·A)·h + Δ·B·x" }), 100, 100);
    const formula = root().querySelector<HTMLElement>(".tensor-tooltip-formula")!;
    expect(formula.hidden).toBe(false);
    expect(formula.textContent).toBe("h = exp(Δ·A)·h + Δ·B·x");

    tooltip.show(makeResult(), 100, 100);
    expect(formula.hidden).toBe(true);
  });

  it("formats -Infinity as masked text", () => {
    tooltip.show(makeResult({ value: -Infinity }), 100, 100);
    expect(root().querySelector(".tensor-tooltip-value")!.textContent).toBe(
      "−∞ (masked)",
    );
  });

  it("marks NaN values with the nan class (magenta styling hook)", () => {
    tooltip.show(makeResult({ value: NaN }), 100, 100);
    const value = root().querySelector<HTMLElement>(".tensor-tooltip-value")!;
    expect(value.textContent).toBe("NaN");
    expect(value.classList.contains("is-nan")).toBe(true);

    tooltip.show(makeResult({ value: 1 }), 100, 100);
    expect(value.classList.contains("is-nan")).toBe(false);
  });

  it("positions near the cursor (offset applied)", () => {
    // happy-dom reports offsetWidth/Height = 0, so the clamp keeps x+offset.
    tooltip.show(makeResult(), 100, 200);
    const el = root();
    expect(el.style.left).toBe(`${100 + TOOLTIP_OFFSET}px`);
    expect(el.style.top).toBe(`${200 + TOOLTIP_OFFSET}px`);
  });

  it("hide() hides; dispose() unmounts", () => {
    tooltip.show(makeResult(), 100, 100);
    tooltip.hide();
    expect(root().classList.contains("visible")).toBe(false);
    tooltip.dispose();
    expect(container.querySelector(".tensor-tooltip")).toBeNull();
  });
});
