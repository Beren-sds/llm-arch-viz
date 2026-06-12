import { describe, expect, it } from "vitest";
import { FpsCounter } from "./scene";

describe("FpsCounter", () => {
  it("returns null until a full window elapses", () => {
    const c = new FpsCounter();
    expect(c.frame(0)).toBeNull(); // first frame only opens the window
    expect(c.frame(16)).toBeNull();
    expect(c.frame(500)).toBeNull();
    expect(c.frame(999)).toBeNull();
  });

  it("reports ~60 fps for 60 frames over one second", () => {
    const c = new FpsCounter();
    c.frame(0);
    let fps: number | null = null;
    for (let i = 1; i <= 60; i++) {
      fps = c.frame(i * (1000 / 60));
    }
    expect(fps).not.toBeNull();
    expect(fps!).toBeCloseTo(60, 5);
  });

  it("scales with the actual elapsed time, not the nominal window", () => {
    const c = new FpsCounter();
    c.frame(0);
    // 30 frames over 2 seconds -> 15 fps
    let fps: number | null = null;
    for (let i = 1; i <= 30; i++) {
      const r = c.frame(i * (2000 / 30));
      if (r !== null) fps = r;
    }
    expect(fps).toBeCloseTo(15, 5);
  });

  it("resets the window after each report", () => {
    const c = new FpsCounter();
    c.frame(0);
    expect(c.frame(1000)).toBeCloseTo(1, 5); // 1 frame over 1s
    expect(c.frame(1500)).toBeNull(); // new window, not yet full
    expect(c.frame(2000)).toBeCloseTo(2, 5); // 2 frames over the next 1s
  });

  it("uses a custom window length when given one", () => {
    const c = new FpsCounter(500);
    c.frame(0);
    expect(c.frame(250)).toBeNull();
    expect(c.frame(500)).toBeCloseTo(4, 5); // 2 frames / 0.5 s
  });
});
