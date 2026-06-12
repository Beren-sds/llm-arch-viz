import * as THREE from "three";
import { describe, expect, it, vi } from "vitest";
import { T } from "../compute/tensor";
import { TensorView, unflattenIndex, type TensorLayout } from "./tensorView";

const LAYOUT: TensorLayout = { cellSize: 1, gap: 0.25, origin: [2, 3, 4] };
const PITCH = 1.25; // cellSize + gap

/** Translation component of instance matrix i (column-major 4x4: elements 12..14). */
function posOf(view: TensorView, i: number): [number, number, number] {
  const a = view.mesh.instanceMatrix.array;
  return [a[16 * i + 12], a[16 * i + 13], a[16 * i + 14]];
}

function colorOf(view: TensorView, i: number): [number, number, number] {
  const a = view.mesh.instanceColor!.array;
  return [a[3 * i], a[3 * i + 1], a[3 * i + 2]];
}

/** The color buffer is a Float32Array, so expectations must be f32-rounded. */
function f32(rgb: [number, number, number]): [number, number, number] {
  return [Math.fround(rgb[0]), Math.fround(rgb[1]), Math.fround(rgb[2])];
}

describe("TensorView construction", () => {
  it("instance count equals product(shape) for 1D/2D/3D", () => {
    expect(new TensorView("v", [5], LAYOUT).mesh.count).toBe(5);
    expect(new TensorView("m", [3, 4], LAYOUT).mesh.count).toBe(12);
    expect(new TensorView("a", [3, 21, 21], LAYOUT).mesh.count).toBe(3 * 21 * 21);
  });

  it("places instance 0 at the origin", () => {
    for (const shape of [[5], [3, 4], [2, 3, 4]]) {
      expect(posOf(new TensorView("t", shape, LAYOUT), 0)).toEqual([2, 3, 4]);
    }
  });

  it("lays a 1D tensor out as a row along +x", () => {
    const view = new TensorView("v", [4], LAYOUT);
    expect(posOf(view, 3)).toEqual([2 + 3 * PITCH, 3, 4]);
  });

  it("lays a 2D tensor out with row i along -y, col j along +x", () => {
    const view = new TensorView("m", [3, 4], LAYOUT);
    // instance (i=2, j=3) -> flat index 2*4+3 = 11
    expect(posOf(view, 11)).toEqual([2 + 3 * PITCH, 3 - 2 * PITCH, 4]);
  });

  it("stacks 3D slabs along -z with default slabGap = 4*cellSize", () => {
    const view = new TensorView("a", [2, 2, 2], LAYOUT);
    // slab 1, row 1, col 0 -> flat index 1*4 + 1*2 + 0 = 6
    expect(posOf(view, 6)).toEqual([2, 3 - PITCH, 4 - 4 * 1]);
  });

  it("honours an explicit slabGap", () => {
    const view = new TensorView("a", [3, 1, 1], { ...LAYOUT, slabGap: 10 });
    expect(posOf(view, 2)).toEqual([2, 3, 4 - 20]);
  });

  it("throws on non-positive cellSize or negative gap", () => {
    expect(() => new TensorView("v", [2], { cellSize: 0, gap: 0.25, origin: [0, 0, 0] })).toThrow(
      /cellSize/,
    );
    expect(() => new TensorView("v", [2], { cellSize: -1, gap: 0.25, origin: [0, 0, 0] })).toThrow(
      /cellSize/,
    );
    expect(() => new TensorView("v", [2], { cellSize: 1, gap: -0.1, origin: [0, 0, 0] })).toThrow(
      /gap/,
    );
    expect(() => new TensorView("v", [2], { cellSize: NaN, gap: 0, origin: [0, 0, 0] })).toThrow(
      /cellSize/,
    );
  });

  it("marks the instance color attribute as dynamic draw usage", () => {
    const view = new TensorView("v", [2], LAYOUT);
    expect(view.mesh.instanceColor!.usage).toBe(THREE.DynamicDrawUsage);
  });

  it("uses cellSize-sized box geometry", () => {
    const view = new TensorView("v", [1], { cellSize: 0.5, gap: 0, origin: [0, 0, 0] });
    view.mesh.geometry.computeBoundingBox();
    const bb = view.mesh.geometry.boundingBox!;
    expect(bb.max.x - bb.min.x).toBeCloseTo(0.5, 10);
  });
});

describe("setValues", () => {
  it("writes colormap colors into the instance color attribute (default scale)", () => {
    const view = new TensorView("v", [3], LAYOUT);
    view.setValues(T.from([-2, 0, 2], [3])); // tensorScale = 2
    expect(colorOf(view, 0)).toEqual(f32([0.13, 0.3, 0.85])); // -scale -> deep blue
    expect(colorOf(view, 1)).toEqual(f32([0.96, 0.96, 0.97])); // 0 -> near-white
    expect(colorOf(view, 2)).toEqual(f32([0.88, 0.18, 0.16])); // +scale -> deep red
  });

  it("respects an explicit scale (clamping beyond it)", () => {
    const view = new TensorView("v", [2], LAYOUT);
    view.setValues(T.from([5, -5], [2]), 1);
    expect(colorOf(view, 0)).toEqual(f32([0.88, 0.18, 0.16]));
    expect(colorOf(view, 1)).toEqual(f32([0.13, 0.3, 0.85]));
    expect(view.scale).toBe(1);
  });

  it("renders -Infinity masked cells with the masked color", () => {
    const view = new TensorView("s", [2], LAYOUT);
    view.setValues(T.from([-Infinity, 1], [2]));
    expect(colorOf(view, 0)).toEqual(f32([0.16, 0.18, 0.22]));
  });

  it("flags the color attribute for upload (version bump)", () => {
    const view = new TensorView("v", [1], LAYOUT);
    const before = view.mesh.instanceColor!.version;
    view.setValues(T.from([1], [1]));
    expect(view.mesh.instanceColor!.version).toBeGreaterThan(before);
  });

  it("throws on shape mismatch, including same-size different-shape", () => {
    const view = new TensorView("m", [2, 2], LAYOUT);
    expect(() => view.setValues(T.from([1, 2, 3], [3]))).toThrow(/shape/);
    expect(() => view.setValues(T.from([1, 2, 3, 4], [4]))).toThrow(/shape/);
  });

  it("stores scale and a defensive copy of the values", () => {
    const view = new TensorView("v", [2], LAYOUT);
    const t = T.from([1, -4], [2]);
    view.setValues(t);
    expect(view.scale).toBe(4);
    expect(Array.from(view.lastValues!)).toEqual([1, -4]);
    t.data[0] = 99;
    expect(view.lastValues![0]).toBe(1); // copy, not a reference
  });
});

describe("index helpers", () => {
  it("unflattenIndex round-trips for 1D/2D/3D shapes", () => {
    for (const shape of [[5], [3, 4], [2, 3, 4]]) {
      const size = shape.reduce((a, b) => a * b, 1);
      for (let i = 0; i < size; i++) {
        const coord = unflattenIndex(i, shape);
        expect(coord).toHaveLength(shape.length);
        let flat = 0;
        for (let d = 0; d < shape.length; d++) flat = flat * shape[d] + coord[d];
        expect(flat).toBe(i);
      }
    }
  });

  it("unflattenIndex throws out of range", () => {
    expect(() => unflattenIndex(12, [3, 4])).toThrow();
    expect(() => unflattenIndex(-1, [3, 4])).toThrow();
  });

  it("indexToCoord matches the view shape rank", () => {
    const view = new TensorView("a", [2, 3, 4], LAYOUT);
    expect(view.indexToCoord(23)).toEqual([1, 2, 3]);
  });

  it("coordOf normalizes 1D/2D/3D to {slab,row,col}", () => {
    expect(new TensorView("v", [5], LAYOUT).coordOf(3)).toEqual({ slab: 0, row: 0, col: 3 });
    expect(new TensorView("m", [3, 4], LAYOUT).coordOf(11)).toEqual({ slab: 0, row: 2, col: 3 });
    expect(new TensorView("a", [2, 3, 4], LAYOUT).coordOf(23)).toEqual({ slab: 1, row: 2, col: 3 });
  });
});

describe("highlight/dim uniforms + dispose", () => {
  it("setHighlight and setDim flip the shader uniforms", () => {
    const view = new TensorView("v", [1], LAYOUT);
    const u = view.material.uniforms;
    expect(u.uHighlight.value).toBe(0);
    expect(u.uDim.value).toBe(0);
    view.setHighlight(true);
    view.setDim(true);
    expect(u.uHighlight.value).toBe(1);
    expect(u.uDim.value).toBe(1);
    view.setHighlight(false);
    view.setDim(false);
    expect(u.uHighlight.value).toBe(0);
    expect(u.uDim.value).toBe(0);
  });

  it("dispose releases geometry and material without throwing", () => {
    const view = new TensorView("v", [2, 2], LAYOUT);
    expect(() => view.dispose()).not.toThrow();
  });

  it("dispose disposes the mesh itself (releases instance GL buffers)", () => {
    const view = new TensorView("v", [2, 2], LAYOUT);
    const meshDispose = vi.spyOn(view.mesh, "dispose");
    let disposeEventFired = false;
    view.mesh.addEventListener("dispose", () => {
      disposeEventFired = true;
    });
    view.dispose();
    expect(meshDispose).toHaveBeenCalledTimes(1);
    expect(disposeEventFired).toBe(true);
  });
});
