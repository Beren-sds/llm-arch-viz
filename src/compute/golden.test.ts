import { describe, expect, it } from "vitest";
import { T } from "./tensor";
import {
  compareToGolden,
  decodeGoldenTensor,
  type GoldenActivations,
} from "./golden";

function goldenOf(entries: Record<string, { shape: number[]; data: (number | string)[] }>) {
  return entries as GoldenActivations;
}

describe("decodeGoldenTensor", () => {
  it("decodes plain finite values into a T of the stated shape", () => {
    const t = decodeGoldenTensor({ shape: [2, 2], data: [1, -2.5, 0, 3] });
    expect(t.shape).toEqual([2, 2]);
    expect(Array.from(t.data)).toEqual([1, -2.5, 0, 3]);
  });

  it('decodes "Infinity" / "-Infinity" / "NaN" strings into numbers', () => {
    const t = decodeGoldenTensor({
      shape: [4],
      data: ["Infinity", "-Infinity", "NaN", 1.5],
    });
    expect(t.data[0]).toBe(Infinity);
    expect(t.data[1]).toBe(-Infinity);
    expect(Number.isNaN(t.data[2])).toBe(true);
    expect(t.data[3]).toBe(1.5);
  });

  it("throws on an unrecognized string instead of guessing", () => {
    expect(() => decodeGoldenTensor({ shape: [1], data: ["nan"] })).toThrow(/nan/);
  });

  it("throws on shape/data length mismatch", () => {
    expect(() => decodeGoldenTensor({ shape: [3], data: [1, 2] })).toThrow();
  });
});

describe("compareToGolden", () => {
  it("reports zero diff and the first element as worst when everything matches", () => {
    const recorded = new Map([["a", T.from([1, 2], [2])]]);
    const res = compareToGolden(recorded, goldenOf({ a: { shape: [2], data: [1, 2] } }));
    expect(res.maxAbsDiff).toBe(0);
    expect(res.missing).toEqual([]);
    expect(res.extra).toEqual([]);
    expect(res.worst).toEqual({ name: "a", idx: 0, got: 1, want: 1, diff: 0 });
  });

  it("finds the worst element across multiple tensors", () => {
    const recorded = new Map([
      ["a", T.from([1, 2], [2])],
      ["b", T.from([5, 5.5], [2])],
    ]);
    const res = compareToGolden(
      recorded,
      goldenOf({
        a: { shape: [2], data: [1, 2.25] },
        b: { shape: [2], data: [5, 5] },
      }),
    );
    expect(res.maxAbsDiff).toBeCloseTo(0.5, 10);
    expect(res.worst).toMatchObject({ name: "b", idx: 1, got: 5.5, want: 5 });
  });

  it("treats sign-matched infinities and NaN-vs-NaN as equal (diff 0)", () => {
    const recorded = new Map([
      ["m", T.from([Infinity, -Infinity, NaN, 1], [4])],
    ]);
    const res = compareToGolden(
      recorded,
      goldenOf({ m: { shape: [4], data: ["Infinity", "-Infinity", "NaN", 1] } }),
    );
    expect(res.maxAbsDiff).toBe(0);
  });

  it("treats finite-vs-nonfinite and sign-flipped infinities as Infinity diff", () => {
    const recordedA = new Map([["m", T.from([1], [1])]]);
    const resA = compareToGolden(
      recordedA,
      goldenOf({ m: { shape: [1], data: ["-Infinity"] } }),
    );
    expect(resA.maxAbsDiff).toBe(Infinity);
    expect(resA.worst).toMatchObject({ name: "m", idx: 0, got: 1, want: -Infinity });

    const recordedB = new Map([["m", T.from([Infinity], [1])]]);
    const resB = compareToGolden(
      recordedB,
      goldenOf({ m: { shape: [1], data: ["-Infinity"] } }),
    );
    expect(resB.maxAbsDiff).toBe(Infinity);

    const recordedC = new Map([["m", T.from([NaN], [1])]]);
    const resC = compareToGolden(recordedC, goldenOf({ m: { shape: [1], data: [0] } }));
    expect(resC.maxAbsDiff).toBe(Infinity);
  });

  it("lists missing (in golden, not recorded) and extra (recorded, not golden) keys", () => {
    const recorded = new Map([
      ["shared", T.from([1], [1])],
      ["only.recorded", T.from([1], [1])],
    ]);
    const res = compareToGolden(
      recorded,
      goldenOf({
        shared: { shape: [1], data: [1] },
        "only.golden": { shape: [1], data: [2] },
      }),
    );
    expect(res.missing).toEqual(["only.golden"]);
    expect(res.extra).toEqual(["only.recorded"]);
    // shared keys are still compared
    expect(res.maxAbsDiff).toBe(0);
  });

  it("throws on shape mismatch, naming the tensor", () => {
    const recorded = new Map([["a", T.from([1, 2], [2, 1])]]);
    expect(() =>
      compareToGolden(recorded, goldenOf({ a: { shape: [2], data: [1, 2] } })),
    ).toThrow(/"a"/);
  });

  it("returns worst null when there are no common keys", () => {
    const res = compareToGolden(new Map(), goldenOf({ a: { shape: [1], data: [1] } }));
    expect(res.worst).toBeNull();
    expect(res.maxAbsDiff).toBe(0);
    expect(res.missing).toEqual(["a"]);
  });
});
