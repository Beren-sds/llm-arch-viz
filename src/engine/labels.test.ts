// Pure-math tests for bracketGeometry — no GL, no font loading. The troika
// Text side of labels.ts is exercised by the screenshot harness instead
// (SDF text needs a real GL context + async font fetch).

import { describe, expect, it } from "vitest";
import { bracketGeometry } from "./labels";

describe("bracketGeometry — horizontal span (along x)", () => {
  // from→to along +x, offset -2 → bracket sits BELOW the segment
  // (perp = (-dy, dx)/len = (0,1,0); A = from + perp·offset).
  // tick = |offset|/2 = 1, pointing further down (away from the tensor).
  it("offset below: span at y-2, ticks extend to y-3, label at y-4, rotation 0", () => {
    const g = bracketGeometry([0, 0, 0], [10, 0, 0], -2);
    expect(g.points).toEqual([
      [0, -3, 0], // left tick end (away from tensor)
      [0, -2, 0], // span start
      [10, -2, 0], // span end
      [10, -3, 0], // right tick end
    ]);
    expect(g.labelPos).toEqual([5, -4, 0]); // midpoint + 2·offset along perp
    expect(g.labelRotationZ).toBe(0);
  });

  it("offset above (+2): ticks flip sign and point up, label above", () => {
    const g = bracketGeometry([0, 0, 0], [10, 0, 0], 2);
    expect(g.points).toEqual([
      [0, 3, 0],
      [0, 2, 0],
      [10, 2, 0],
      [10, 3, 0],
    ]);
    expect(g.labelPos).toEqual([5, 4, 0]);
    expect(g.labelRotationZ).toBe(0);
  });

  it("tick direction tracks the offset side: ticks are OUTSIDE the span", () => {
    const below = bracketGeometry([0, 0, 0], [10, 0, 0], -2);
    const above = bracketGeometry([0, 0, 0], [10, 0, 0], 2);
    // tick end is farther from the segment (y=0) than the span line
    expect(Math.abs(below.points[0][1])).toBeGreaterThan(Math.abs(below.points[1][1]));
    expect(Math.abs(above.points[0][1])).toBeGreaterThan(Math.abs(above.points[1][1]));
    expect(below.points[0][1]).toBeLessThan(0);
    expect(above.points[0][1]).toBeGreaterThan(0);
  });

  it("reversed span (-x direction) still yields rotation 0 (label never upside-down)", () => {
    const g = bracketGeometry([10, 0, 0], [0, 0, 0], -2);
    expect(g.labelRotationZ).toBe(0);
  });

  it("carries z through unchanged", () => {
    const g = bracketGeometry([0, 0, 5], [10, 0, 5], -2);
    for (const p of g.points) expect(p[2]).toBe(5);
    expect(g.labelPos[2]).toBe(5);
  });
});

describe("bracketGeometry — vertical span (along y)", () => {
  // Top→bottom along -y: dir = (0,-1,0), perp = (-dy, dx)/len = (1,0,0).
  // offset -2 → bracket sits LEFT of the segment; ticks extend further left.
  it("left-side bracket: span at x-2, ticks to x-3, label at x-4, rotation π/2", () => {
    const g = bracketGeometry([0, 0, 0], [0, -10, 0], -2);
    expect(g.points).toEqual([
      [-3, 0, 0], // top tick end
      [-2, 0, 0], // span start
      [-2, -10, 0], // span end
      [-3, -10, 0], // bottom tick end
    ]);
    expect(g.labelPos).toEqual([-4, -5, 0]);
    expect(g.labelRotationZ).toBeCloseTo(Math.PI / 2, 12);
  });

  it("upward span (+y) also normalizes to π/2 (reads bottom-to-top)", () => {
    const g = bracketGeometry([0, -10, 0], [0, 0, 0], 2);
    expect(g.labelRotationZ).toBeCloseTo(Math.PI / 2, 12);
  });

  it("vertical ticks point away from the tensor toward the offset side", () => {
    const left = bracketGeometry([0, 0, 0], [0, -10, 0], -2);
    expect(left.points[0][0]).toBeLessThan(left.points[1][0]); // tick further -x than span
  });
});

describe("bracketGeometry — degenerate input", () => {
  it("throws when from and to coincide in the XY plane", () => {
    expect(() => bracketGeometry([1, 2, 0], [1, 2, 0], -2)).toThrow(/XY/);
    // differs only in z: perpendicular in XY is undefined
    expect(() => bracketGeometry([1, 2, 0], [1, 2, 5], -2)).toThrow(/XY/);
  });
});
