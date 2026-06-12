import { describe, expect, it, vi } from "vitest";
import {
  type CameraKeyframe,
  TourPlayer,
  cubicEase,
  interpolateView,
} from "./cameraTour";

describe("cubicEase", () => {
  it("maps the endpoints exactly", () => {
    expect(cubicEase(0)).toBe(0);
    expect(cubicEase(1)).toBe(1);
  });

  it("maps the midpoint to 0.5 (odd symmetry of 3t^2-2t^3)", () => {
    expect(cubicEase(0.5)).toBeCloseTo(0.5, 12);
  });

  it("is monotonically non-decreasing on [0,1]", () => {
    let prev = cubicEase(0);
    for (let i = 1; i <= 100; i++) {
      const cur = cubicEase(i / 100);
      expect(cur).toBeGreaterThanOrEqual(prev);
      prev = cur;
    }
  });

  it("clamps outside [0,1]", () => {
    expect(cubicEase(-0.5)).toBe(0);
    expect(cubicEase(-1e9)).toBe(0);
    expect(cubicEase(1.5)).toBe(1);
    expect(cubicEase(1e9)).toBe(1);
  });

  it("eases: slower than linear near 0, faster than linear near 1", () => {
    expect(cubicEase(0.1)).toBeLessThan(0.1);
    expect(cubicEase(0.9)).toBeGreaterThan(0.9);
  });
});

const A: CameraKeyframe = { pos: [0, 0, 0], target: [1, 2, 3] };
const B: CameraKeyframe = { pos: [10, 20, 30], target: [-1, -2, -3] };

describe("interpolateView", () => {
  it("returns a exactly at t=0", () => {
    const v = interpolateView(A, B, 0);
    expect(v.pos).toEqual(A.pos);
    expect(v.target).toEqual(A.target);
  });

  it("returns b exactly at t=1", () => {
    const v = interpolateView(A, B, 1);
    expect(v.pos).toEqual(B.pos);
    expect(v.target).toEqual(B.target);
  });

  it("returns the componentwise midpoint at t=0.5 (eased 0.5 = 0.5)", () => {
    const v = interpolateView(A, B, 0.5);
    expect(v.pos[0]).toBeCloseTo(5, 12);
    expect(v.pos[1]).toBeCloseTo(10, 12);
    expect(v.pos[2]).toBeCloseTo(15, 12);
    expect(v.target[0]).toBeCloseTo(0, 12);
    expect(v.target[1]).toBeCloseTo(0, 12);
    expect(v.target[2]).toBeCloseTo(0, 12);
  });

  it("applies the ease to t (t=0.25 is NOT the linear quarter point)", () => {
    const v = interpolateView(A, B, 0.25);
    // cubicEase(0.25) = 3*0.0625 - 2*0.015625 = 0.15625
    expect(v.pos[0]).toBeCloseTo(10 * 0.15625, 12);
  });

  it("writes into and returns the provided out param (no allocation)", () => {
    const out: CameraKeyframe = { pos: [9, 9, 9], target: [9, 9, 9] };
    const v = interpolateView(A, B, 1, out);
    expect(v).toBe(out);
    expect(out.pos).toEqual(B.pos);
    expect(out.target).toEqual(B.target);
  });

  it("does not mutate its inputs", () => {
    interpolateView(A, B, 0.7);
    expect(A.pos).toEqual([0, 0, 0]);
    expect(B.pos).toEqual([10, 20, 30]);
  });
});

/** applyView spy that records copies (the player may reuse its arrays). */
function makeSpy() {
  const calls: { pos: number[]; target: number[] }[] = [];
  const applyView = (pos: readonly number[], target: readonly number[]): void => {
    calls.push({ pos: [...pos], target: [...target] });
  };
  const last = () => {
    const l = calls[calls.length - 1];
    if (!l) throw new Error("applyView was never called");
    return l;
  };
  return { calls, applyView, last };
}

describe("TourPlayer", () => {
  it("first goTo with no prior view applies instantly and stays idle", () => {
    const spy = makeSpy();
    const p = new TourPlayer({ applyView: spy.applyView });
    p.goTo(A);
    expect(spy.calls.length).toBe(1);
    expect(spy.last().pos).toEqual(A.pos);
    expect(spy.last().target).toEqual(A.target);
    expect(p.state).toBe("idle");
  });

  it("goTo with instant:true snaps even when a prior view exists", () => {
    const spy = makeSpy();
    const p = new TourPlayer({ applyView: spy.applyView });
    p.goTo(A);
    p.goTo(B, { instant: true });
    expect(spy.calls.length).toBe(2);
    expect(spy.last().pos).toEqual(B.pos);
    expect(p.state).toBe("idle");
  });

  it("animates from current view and hits the endpoint exactly at duration", () => {
    const spy = makeSpy();
    const p = new TourPlayer({ applyView: spy.applyView, durationMs: 1000 });
    p.goTo(A);
    p.goTo(B);
    expect(p.state).toBe("animating");
    p.update(5000); // first update opens the clock
    p.update(6000); // exactly t = duration
    expect(p.state).toBe("idle");
    expect(spy.last().pos).toEqual(B.pos);
    expect(spy.last().target).toEqual(B.target);
  });

  it("is at the exact midpoint halfway through", () => {
    const spy = makeSpy();
    const p = new TourPlayer({ applyView: spy.applyView, durationMs: 1000 });
    p.goTo(A);
    p.goTo(B);
    p.update(0);
    p.update(500);
    expect(spy.last().pos[0]).toBeCloseTo(5, 9);
    expect(spy.last().pos[1]).toBeCloseTo(10, 9);
    expect(spy.last().pos[2]).toBeCloseTo(15, 9);
    expect(spy.last().target[0]).toBeCloseTo(0, 9);
    expect(p.state).toBe("animating");
  });

  it("overshooting the duration clamps to the endpoint (no extrapolation)", () => {
    const spy = makeSpy();
    const p = new TourPlayer({ applyView: spy.applyView, durationMs: 1000 });
    p.goTo(A);
    p.goTo(B);
    p.update(0);
    p.update(99999);
    expect(spy.last().pos).toEqual(B.pos);
    expect(p.state).toBe("idle");
  });

  it("uses the default 1200 ms duration when none is given", () => {
    const spy = makeSpy();
    const p = new TourPlayer({ applyView: spy.applyView });
    p.goTo(A);
    p.goTo(B);
    p.update(0);
    p.update(1199);
    expect(p.state).toBe("animating");
    p.update(1200);
    expect(p.state).toBe("idle");
    expect(spy.last().pos).toEqual(B.pos);
  });

  it("retargets mid-flight from the interpolated view — no teleport", () => {
    const spy = makeSpy();
    const p = new TourPlayer({ applyView: spy.applyView, durationMs: 1000 });
    const C: CameraKeyframe = { pos: [-50, 0, 7], target: [4, 4, 4] };
    p.goTo(A);
    p.goTo(B);
    p.update(0);
    p.update(600);
    const before = spy.last();
    p.goTo(C); // retarget mid-flight
    expect(p.state).toBe("animating");
    p.update(600.001); // immediately after: must (re)start from `before`
    const after = spy.last();
    for (let i = 0; i < 3; i++) {
      expect(after.pos[i]).toBeCloseTo(before.pos[i], 6);
      expect(after.target[i]).toBeCloseTo(before.target[i], 6);
    }
    // ...and it still arrives at C
    p.update(600.001 + 1000);
    expect(spy.last().pos).toEqual(C.pos);
    expect(spy.last().target).toEqual(C.target);
  });

  it("fires onArrive exactly once on completion", () => {
    const spy = makeSpy();
    const onArrive = vi.fn();
    const p = new TourPlayer({ applyView: spy.applyView, durationMs: 1000 });
    p.goTo(A);
    p.goTo(B, { onArrive });
    p.update(0);
    p.update(500);
    expect(onArrive).not.toHaveBeenCalled();
    p.update(1000);
    expect(onArrive).toHaveBeenCalledTimes(1);
    p.update(2000); // idle updates must not re-fire
    expect(onArrive).toHaveBeenCalledTimes(1);
  });

  it("fires onArrive for an instant goTo too (it completes immediately)", () => {
    const spy = makeSpy();
    const onArrive = vi.fn();
    const p = new TourPlayer({ applyView: spy.applyView });
    p.goTo(A, { onArrive }); // first call = instant
    expect(onArrive).toHaveBeenCalledTimes(1);
  });

  it("does NOT fire the superseded onArrive on retarget", () => {
    const spy = makeSpy();
    const first = vi.fn();
    const second = vi.fn();
    const p = new TourPlayer({ applyView: spy.applyView, durationMs: 1000 });
    p.goTo(A);
    p.goTo(B, { onArrive: first });
    p.update(0);
    p.update(500);
    p.goTo({ pos: [7, 7, 7], target: [0, 0, 0] }, { onArrive: second });
    p.update(500);
    p.update(1500);
    expect(first).not.toHaveBeenCalled();
    expect(second).toHaveBeenCalledTimes(1);
  });

  it("cancel freezes at the current interpolated view and skips onArrive", () => {
    const spy = makeSpy();
    const onArrive = vi.fn();
    const p = new TourPlayer({ applyView: spy.applyView, durationMs: 1000 });
    p.goTo(A);
    p.goTo(B, { onArrive });
    p.update(0);
    p.update(500);
    const frozen = spy.last();
    const n = spy.calls.length;
    p.cancel();
    expect(p.state).toBe("idle");
    p.update(1000);
    p.update(5000);
    expect(spy.calls.length).toBe(n); // no further applies after cancel
    expect(spy.last()).toEqual(frozen);
    expect(onArrive).not.toHaveBeenCalled();
  });

  it("a fresh goTo after cancel animates from the frozen view", () => {
    const spy = makeSpy();
    const p = new TourPlayer({ applyView: spy.applyView, durationMs: 1000 });
    p.goTo(A);
    p.goTo(B);
    p.update(0);
    p.update(500);
    const frozen = spy.last();
    p.cancel();
    p.goTo(B);
    p.update(1000); // t=0 of the new animation: must equal the frozen view
    const resumed = spy.last();
    for (let i = 0; i < 3; i++) {
      expect(resumed.pos[i]).toBeCloseTo(frozen.pos[i], 9);
    }
  });

  it("update while idle is a no-op", () => {
    const spy = makeSpy();
    const p = new TourPlayer({ applyView: spy.applyView });
    p.update(123);
    expect(spy.calls.length).toBe(0);
    p.goTo(A);
    const n = spy.calls.length;
    p.update(456);
    p.update(789);
    expect(spy.calls.length).toBe(n);
  });

  it("does not alias the caller's keyframe arrays (later mutation is ignored)", () => {
    const spy = makeSpy();
    const p = new TourPlayer({ applyView: spy.applyView, durationMs: 1000 });
    const kf: CameraKeyframe = { pos: [10, 0, 0], target: [0, 0, 0] };
    p.goTo({ pos: [0, 0, 0], target: [0, 0, 0] });
    p.goTo(kf);
    kf.pos[0] = -999; // caller mutates after the fact
    p.update(0);
    p.update(1000);
    expect(spy.last().pos[0]).toBe(10);
  });

  it("state is a read-only getter (assignment throws)", () => {
    const p = new TourPlayer({ applyView: makeSpy().applyView });
    expect(() => {
      (p as unknown as { state: string }).state = "animating";
    }).toThrow(TypeError);
    expect(p.state).toBe("idle");
  });

  it("clamps durationMs to >= 1 (zero/negative never divide by zero)", () => {
    for (const durationMs of [0, -500]) {
      const spy = makeSpy();
      const p = new TourPlayer({ applyView: spy.applyView, durationMs });
      p.goTo(A);
      p.goTo(B);
      p.update(0); // opens the clock at the start view
      p.update(0.5); // t = 0.5 of the clamped 1 ms — still animating, finite
      expect(p.state).toBe("animating");
      expect(Number.isFinite(spy.last().pos[0])).toBe(true);
      p.update(1); // t = 1 → arrive exactly
      expect(p.state).toBe("idle");
      expect(spy.last().pos).toEqual(B.pos);
    }
  });
});

describe("TourPlayer.syncCurrent", () => {
  it("next goTo animates from the externally synced view (no snap-back)", () => {
    const spy = makeSpy();
    const p = new TourPlayer({ applyView: spy.applyView, durationMs: 1000 });
    p.goTo(A); // player thinks we're at A
    // OrbitControls dragged the camera elsewhere:
    p.syncCurrent([100, 200, 300], [4, 5, 6]);
    p.goTo(B);
    p.update(0); // t=0 must start from the synced view, not from A
    expect(spy.last().pos).toEqual([100, 200, 300]);
    expect(spy.last().target).toEqual([4, 5, 6]);
    p.update(1000);
    expect(spy.last().pos).toEqual(B.pos);
  });

  it("syncCurrent before any goTo seeds the view: first goTo animates instead of snapping", () => {
    const spy = makeSpy();
    const p = new TourPlayer({ applyView: spy.applyView, durationMs: 1000 });
    p.syncCurrent([100, 0, 0], [0, 0, 0]);
    p.goTo(B);
    expect(p.state).toBe("animating");
    p.update(0);
    expect(spy.last().pos).toEqual([100, 0, 0]);
  });

  it("does not call applyView (the external writer already moved the camera)", () => {
    const spy = makeSpy();
    const p = new TourPlayer({ applyView: spy.applyView });
    p.goTo(A);
    const n = spy.calls.length;
    p.syncCurrent([1, 2, 3], [4, 5, 6]);
    expect(spy.calls.length).toBe(n);
  });

  it("copies its arguments — later caller mutation is ignored", () => {
    const spy = makeSpy();
    const p = new TourPlayer({ applyView: spy.applyView, durationMs: 1000 });
    p.goTo(A);
    const pos: [number, number, number] = [100, 0, 0];
    p.syncCurrent(pos, [0, 0, 0]);
    pos[0] = -999;
    p.goTo(B);
    p.update(0);
    expect(spy.last().pos[0]).toBe(100);
  });
});
