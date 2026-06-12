import { describe, expect, it } from "vitest";
import { T } from "./tensor";

describe("T.from", () => {
  it("builds a tensor from a number[] and shape", () => {
    const t = T.from([1, 2, 3, 4, 5, 6], [2, 3]);
    expect(t.shape).toEqual([2, 3]);
    expect(t.data).toBeInstanceOf(Float32Array);
    expect(Array.from(t.data)).toEqual([1, 2, 3, 4, 5, 6]);
  });

  it("accepts a Float32Array (copied, not aliased)", () => {
    const src = new Float32Array([1, 2]);
    const t = T.from(src, [2]);
    src[0] = 99;
    expect(t.data[0]).toBe(1);
  });

  it("stores values rounded to float32", () => {
    const t = T.from([0.1], [1]);
    expect(t.data[0]).toBe(Math.fround(0.1));
  });

  it("throws on length/shape mismatch", () => {
    expect(() => T.from([1, 2, 3], [2, 2])).toThrow();
    expect(() => T.from([1, 2, 3, 4, 5], [2, 2])).toThrow();
  });

  it("throws on non-integer or negative dims", () => {
    expect(() => T.from([1, 2], [1.5, 2])).toThrow();
    expect(() => T.from([1, 2], [-2, -1])).toThrow();
  });
});

describe("T.zeros", () => {
  it("creates an all-zero tensor of the given shape", () => {
    const t = T.zeros([2, 2]);
    expect(t.shape).toEqual([2, 2]);
    expect(Array.from(t.data)).toEqual([0, 0, 0, 0]);
  });
});

describe("T.size", () => {
  it("is the product of dims", () => {
    expect(T.zeros([2, 3, 4]).size).toBe(24);
    expect(T.zeros([5]).size).toBe(5);
  });
});

describe("T.at", () => {
  it("indexes row-major", () => {
    const t = T.from([1, 2, 3, 4, 5, 6], [2, 3]);
    expect(t.at(0, 0)).toBe(1);
    expect(t.at(0, 2)).toBe(3);
    expect(t.at(1, 0)).toBe(4);
    expect(t.at(1, 2)).toBe(6);
  });

  it("indexes 3D row-major", () => {
    const t = T.from([0, 1, 2, 3, 4, 5, 6, 7], [2, 2, 2]);
    expect(t.at(1, 0, 1)).toBe(5);
    expect(t.at(0, 1, 0)).toBe(2);
  });

  it("throws on wrong arity", () => {
    const t = T.from([1, 2, 3, 4], [2, 2]);
    expect(() => t.at(0)).toThrow();
    expect(() => t.at(0, 0, 0)).toThrow();
  });

  it("throws on out-of-range / negative / non-integer index", () => {
    const t = T.from([1, 2, 3, 4], [2, 2]);
    expect(() => t.at(2, 0)).toThrow();
    expect(() => t.at(0, -1)).toThrow();
    expect(() => t.at(0.5, 0)).toThrow();
  });
});

describe("T.clone", () => {
  it("returns an independent deep copy", () => {
    const a = T.from([1, 2], [2]);
    const b = a.clone();
    b.data[0] = 42;
    expect(a.data[0]).toBe(1);
    expect(b.data[1]).toBe(2);
    // shape is `readonly number[]` (mutation is a compile error); still
    // assert the clone holds its own array instance, not a shared one.
    expect(b.shape).toEqual(a.shape);
    expect(b.shape).not.toBe(a.shape);
  });
});
