import { describe, expect, it } from "vitest";
import { T } from "../compute/tensor";
import {
  MASKED_COLOR,
  ERROR_COLOR,
  NEG_COLOR,
  MID_COLOR,
  POS_COLOR,
  tensorScale,
  valueColor,
} from "./colormap";

function rgb(v: number, scale: number): [number, number, number] {
  const out = { r: 0, g: 0, b: 0 };
  valueColor(v, scale, out);
  return [out.r, out.g, out.b];
}

describe("valueColor", () => {
  it("hits the exact endpoint colors at -scale, 0, +scale", () => {
    expect(rgb(-2, 2)).toEqual([NEG_COLOR.r, NEG_COLOR.g, NEG_COLOR.b]);
    expect(rgb(0, 2)).toEqual([MID_COLOR.r, MID_COLOR.g, MID_COLOR.b]);
    expect(rgb(2, 2)).toEqual([POS_COLOR.r, POS_COLOR.g, POS_COLOR.b]);
  });

  it("interpolates linearly at the positive midpoint (t = 0.5 toward POS)", () => {
    const [r, g, b] = rgb(1, 2);
    expect(r).toBeCloseTo(0.5 * MID_COLOR.r + 0.5 * POS_COLOR.r, 10);
    expect(g).toBeCloseTo(0.5 * MID_COLOR.g + 0.5 * POS_COLOR.g, 10);
    expect(b).toBeCloseTo(0.5 * MID_COLOR.b + 0.5 * POS_COLOR.b, 10);
  });

  it("interpolates linearly at the negative midpoint (t = 0.5 toward NEG)", () => {
    const [r, g, b] = rgb(-1, 2);
    expect(r).toBeCloseTo(0.5 * MID_COLOR.r + 0.5 * NEG_COLOR.r, 10);
    expect(g).toBeCloseTo(0.5 * MID_COLOR.g + 0.5 * NEG_COLOR.g, 10);
    expect(b).toBeCloseTo(0.5 * MID_COLOR.b + 0.5 * NEG_COLOR.b, 10);
  });

  it("clamps values beyond +/- scale to the endpoint colors", () => {
    expect(rgb(100, 1)).toEqual(rgb(1, 1));
    expect(rgb(-100, 1)).toEqual(rgb(-1, 1));
  });

  it("maps -Infinity to the masked dark-slate color", () => {
    expect(rgb(-Infinity, 1)).toEqual([MASKED_COLOR.r, MASKED_COLOR.g, MASKED_COLOR.b]);
    expect(rgb(-Infinity, 1)).toEqual([0.16, 0.18, 0.22]);
  });

  it("maps +Infinity and NaN to magenta (error signal)", () => {
    expect(rgb(Infinity, 1)).toEqual([1, 0, 1]);
    expect(rgb(NaN, 1)).toEqual([1, 0, 1]);
    expect(ERROR_COLOR).toEqual({ r: 1, g: 0, b: 1 });
  });

  it("returns the out object for chaining", () => {
    const out = { r: 0, g: 0, b: 0 };
    expect(valueColor(0, 1, out)).toBe(out);
  });

  it("throws on scale <= 0 or non-finite scale", () => {
    const out = { r: 0, g: 0, b: 0 };
    expect(() => valueColor(1, 0, out)).toThrow();
    expect(() => valueColor(1, -1, out)).toThrow();
    expect(() => valueColor(1, NaN, out)).toThrow();
    expect(() => valueColor(1, Infinity, out)).toThrow();
  });
});

describe("tensorScale", () => {
  it("returns the max |finite value| over data", () => {
    expect(tensorScale(T.from([1, -3, 2, 0.5], [4]))).toBe(3);
  });

  it("ignores non-finite entries (-inf masked cells, NaN)", () => {
    expect(tensorScale(T.from([-Infinity, 0.25, -0.5, NaN], [2, 2]))).toBe(0.5);
  });

  it("returns 1 for an all-masked tensor", () => {
    expect(tensorScale(T.from([-Infinity, -Infinity], [2]))).toBe(1);
  });

  it("returns 1 for an all-zero tensor (no finite nonzero values)", () => {
    expect(tensorScale(T.zeros([3]))).toBe(1);
  });
});
