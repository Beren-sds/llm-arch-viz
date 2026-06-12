import { describe, expect, it } from "vitest";
import { T } from "./tensor";
import { MapRecorder, type Recorder } from "./recorder";

describe("MapRecorder", () => {
  it("stores tensors under their names", () => {
    const rec = new MapRecorder();
    rec.record("a", T.from([1, 2], [2]));
    rec.record("b", T.from([3], [1]));
    expect(rec.activations.size).toBe(2);
    expect(Array.from(rec.activations.get("a")!.data)).toEqual([1, 2]);
    expect(rec.activations.get("b")!.shape).toEqual([1]);
  });

  it("clones on record: later mutation of the source does not leak in", () => {
    const rec = new MapRecorder();
    const src = T.from([1, 2, 3], [3]);
    rec.record("x", src);
    src.data[0] = 999;
    expect(Array.from(rec.activations.get("x")!.data)).toEqual([1, 2, 3]);
  });

  it("throws on duplicate names instead of silently overwriting", () => {
    const rec = new MapRecorder();
    rec.record("x", T.zeros([1]));
    expect(() => rec.record("x", T.zeros([1]))).toThrow(/"x"/);
  });

  it("satisfies the Recorder interface", () => {
    const rec: Recorder = new MapRecorder();
    rec.record("y", T.zeros([2]));
  });
});
