// @vitest-environment happy-dom
//
// happy-dom: the Picker attaches pointer listeners to a DOM element and the
// tests dispatch synthetic pointer events. The raycast itself is pure math
// (no GL context needed), so the whole pick path is testable headlessly.

import * as THREE from "three";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { T } from "../compute/tensor";
import { Picker, resolvePick, type PickResult } from "./picking";
import { TensorView, type TensorLayout } from "./tensorView";

const LAYOUT: TensorLayout = { cellSize: 1, gap: 0.25, origin: [0, 0, 0] };

describe("resolvePick (pure math, no raycaster)", () => {
  it("1D shape [5]: indices = [i], value passthrough", () => {
    const view = new TensorView("vec", [5], LAYOUT);
    view.setValues(T.from([10, 11, 12, 13, 14], [5]));
    expect(resolvePick(view, 0)).toEqual({ name: "vec", indices: [0], value: 10 });
    expect(resolvePick(view, 3)).toEqual({ name: "vec", indices: [3], value: 13 });
  });

  it("2D shape [4,6]: indices = [row, col]", () => {
    const view = new TensorView("mat", [4, 6], LAYOUT);
    const values = new Float32Array(24);
    for (let k = 0; k < 24; k++) values[k] = k * 0.5;
    view.setValues(T.from(values, [4, 6]));
    // instanceId 15 -> row 2, col 3 (row-major)
    expect(resolvePick(view, 15)).toEqual({ name: "mat", indices: [2, 3], value: 7.5 });
    expect(resolvePick(view, 0)).toEqual({ name: "mat", indices: [0, 0], value: 0 });
    expect(resolvePick(view, 23)).toEqual({ name: "mat", indices: [3, 5], value: 11.5 });
  });

  it("3D shape [2,3,4]: indices = [slab, row, col]", () => {
    const view = new TensorView("attn", [2, 3, 4], LAYOUT);
    const values = new Float32Array(24);
    for (let k = 0; k < 24; k++) values[k] = k;
    view.setValues(T.from(values, [2, 3, 4]));
    // instanceId 17 = 1*12 + 1*4 + 1 -> [1, 1, 1]
    expect(resolvePick(view, 17)).toEqual({ name: "attn", indices: [1, 1, 1], value: 17 });
    expect(resolvePick(view, 23)).toEqual({ name: "attn", indices: [1, 2, 3], value: 23 });
  });

  it("passes NaN and -Infinity through unchanged at known instanceIds", () => {
    const view = new TensorView("masked", [2, 2], LAYOUT);
    view.setValues(T.from([1, NaN, -Infinity, 4], [2, 2]));
    expect(resolvePick(view, 1).value).toBeNaN();
    expect(resolvePick(view, 1).indices).toEqual([0, 1]);
    expect(resolvePick(view, 2).value).toBe(-Infinity);
    expect(resolvePick(view, 2).indices).toEqual([1, 0]);
  });

  it("throws on out-of-range instanceId", () => {
    const view = new TensorView("v", [2, 2], LAYOUT);
    expect(() => resolvePick(view, 4)).toThrow();
    expect(() => resolvePick(view, -1)).toThrow();
  });
});

// --- Picker (raycast + event throttling) --------------------------------

const W = 800;
const H = 600;

/** A camera at z=10 looking at the origin, matching an 800x600 element. */
function makeCamera(): THREE.PerspectiveCamera {
  const camera = new THREE.PerspectiveCamera(45, W / H, 0.1, 100);
  camera.position.set(0, 0, 10);
  camera.lookAt(0, 0, 0);
  camera.updateMatrixWorld();
  return camera;
}

/** DOM element whose bounding rect is a fixed 800x600 at the page origin. */
function makeElement(): HTMLElement {
  const el = document.createElement("div");
  el.getBoundingClientRect = () =>
    ({ left: 0, top: 0, width: W, height: H, right: W, bottom: H, x: 0, y: 0 }) as DOMRect;
  return el;
}

/** Single unit cube at the world origin, value 42. */
function makeUnitView(): TensorView {
  const view = new TensorView("cell", [1], { cellSize: 1, gap: 0, origin: [0, 0, 0] });
  view.setValues(T.from([42], [1]));
  return view;
}

function pointerMove(el: HTMLElement, clientX: number, clientY: number): void {
  el.dispatchEvent(new MouseEvent("pointermove", { clientX, clientY }));
}

describe("Picker.pick", () => {
  let camera: THREE.PerspectiveCamera;
  let el: HTMLElement;

  beforeEach(() => {
    camera = makeCamera();
    el = makeElement();
  });

  it("hits a registered view at the element center and resolves the result", () => {
    const view = makeUnitView();
    const picker = new Picker(camera, el);
    picker.add({ view, formula: "y = W·x" });
    const result = picker.pick(W / 2, H / 2);
    expect(result).not.toBeNull();
    expect(result!.view).toBe(view);
    expect(result!.name).toBe("cell");
    expect(result!.indices).toEqual([0]);
    expect(result!.value).toBe(42);
    expect(result!.formula).toBe("y = W·x");
    picker.dispose();
  });

  it("returns null when the ray misses everything", () => {
    const picker = new Picker(camera, el);
    picker.add({ view: makeUnitView() });
    expect(picker.pick(0, 0)).toBeNull(); // top-left corner: empty space
    picker.dispose();
  });

  it("omits formula when the target has none", () => {
    const picker = new Picker(camera, el);
    picker.add({ view: makeUnitView() });
    expect(picker.pick(W / 2, H / 2)!.formula).toBeUndefined();
    picker.dispose();
  });

  it("no longer hits a removed view", () => {
    const view = makeUnitView();
    const picker = new Picker(camera, el);
    picker.add({ view });
    expect(picker.pick(W / 2, H / 2)).not.toBeNull();
    picker.remove(view);
    expect(picker.pick(W / 2, H / 2)).toBeNull();
    picker.dispose();
  });
});

describe("Picker pointermove throttling (latest event, consumed per update)", () => {
  let camera: THREE.PerspectiveCamera;
  let el: HTMLElement;

  beforeEach(() => {
    camera = makeCamera();
    el = makeElement();
  });

  it("does not pick on pointermove alone; update() consumes the LATEST event once", () => {
    const picker = new Picker(camera, el);
    picker.add({ view: makeUnitView() });
    const onPick = vi.fn<(r: PickResult | null, x: number, y: number) => void>();
    picker.onPick = onPick;

    pointerMove(el, 0, 0); // would miss
    pointerMove(el, 10, 10); // would miss
    pointerMove(el, W / 2, H / 2); // hit — the latest wins
    expect(onPick).not.toHaveBeenCalled();

    picker.update();
    expect(onPick).toHaveBeenCalledTimes(1);
    const [result, x, y] = onPick.mock.calls[0];
    expect(result).not.toBeNull();
    expect(result!.value).toBe(42);
    expect(x).toBe(W / 2);
    expect(y).toBe(H / 2);

    picker.update(); // nothing pending: no second callback
    expect(onPick).toHaveBeenCalledTimes(1);
    picker.dispose();
  });

  it("pointerleave reports null on the next update", () => {
    const picker = new Picker(camera, el);
    picker.add({ view: makeUnitView() });
    const onPick = vi.fn<(r: PickResult | null, x: number, y: number) => void>();
    picker.onPick = onPick;

    pointerMove(el, W / 2, H / 2);
    picker.update();
    expect(onPick.mock.calls[0][0]).not.toBeNull();

    el.dispatchEvent(new MouseEvent("pointerleave"));
    picker.update();
    expect(onPick).toHaveBeenCalledTimes(2);
    expect(onPick.mock.calls[1][0]).toBeNull();
    picker.dispose();
  });

  it("dispose() detaches listeners: later moves are ignored", () => {
    const picker = new Picker(camera, el);
    picker.add({ view: makeUnitView() });
    const onPick = vi.fn();
    picker.onPick = onPick;
    picker.dispose();

    pointerMove(el, W / 2, H / 2);
    picker.update();
    expect(onPick).not.toHaveBeenCalled();
  });
});
