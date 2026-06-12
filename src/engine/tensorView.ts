/**
 * TensorView: one InstancedMesh per tensor — every cell of a 1D/2D/3D
 * tensor is a box instance, colored per-value through the diverging
 * colormap (bbycroft.net/llm look: flat-shaded grids of cells).
 *
 * Layout convention (row-major, matching T's memory order):
 *   1D (n)                -> row along +x
 *   2D (rows, cols)       -> col j along +x, row i along -y (row 0 on top)
 *   3D (slabs, rows, cols) -> 2D grids stacked along -z, slabGap apart
 */

import * as THREE from "three";
import type { T } from "../compute/tensor";
import { tensorScale, valueColor, type RGB } from "./colormap";

export interface TensorLayout {
  cellSize: number;
  gap: number;
  origin: [number, number, number];
  /** z spacing between 3D slabs; defaults to 4 * cellSize. */
  slabGap?: number;
}

/** Row-major unflatten; shared by picking/tooltip code. Throws out of range. */
export function unflattenIndex(i: number, shape: readonly number[]): number[] {
  let size = 1;
  for (const d of shape) size *= d;
  if (!Number.isInteger(i) || i < 0 || i >= size) {
    throw new Error(`unflattenIndex: index ${i} out of range for shape [${shape.join(", ")}]`);
  }
  const coord: number[] = new Array(shape.length).fill(0);
  let rem = i;
  for (let d = shape.length - 1; d >= 0; d--) {
    coord[d] = rem % shape[d];
    rem = Math.floor(rem / shape[d]);
  }
  return coord;
}

// Vertex: instance transform + per-instance color, with a cheap fake-light
// touch — top (+y) and front (+z) faces slightly brighter so cube edges
// read without real lighting. No normals needed in the fragment stage.
const VERTEX = /* glsl */ `
varying vec3 vColor;
void main() {
  #ifdef USE_INSTANCING_COLOR
    vColor = instanceColor;
  #else
    vColor = vec3(1.0);
  #endif
  vColor *= 0.84 + 0.16 * max(normal.y, 0.0) + 0.08 * max(normal.z, 0.0);
  vec4 p = vec4(position, 1.0);
  #ifdef USE_INSTANCING
    p = instanceMatrix * p;
  #endif
  gl_Position = projectionMatrix * modelViewMatrix * p;
}
`;

// Fragment: flat color, dimmed to 35% when uDim=1, additive pulse when
// uHighlight=1. The engine animates these floats between 0 and 1 later.
const FRAGMENT = /* glsl */ `
uniform float uDim;
uniform float uHighlight;
varying vec3 vColor;
void main() {
  vec3 c = vColor * mix(1.0, 0.35, uDim) + vec3(uHighlight * 0.25);
  gl_FragColor = vec4(c, 1.0);
}
`;

export class TensorView {
  readonly name: string;
  readonly shape: readonly number[];
  readonly layout: TensorLayout;
  readonly mesh: THREE.InstancedMesh;
  readonly material: THREE.ShaderMaterial;
  /** Copy of the most recent values (for tooltip lookup); null before setValues. */
  lastValues: Float32Array | null = null;
  /** Color scale used by the most recent setValues. */
  scale = 1;

  constructor(name: string, shape: readonly number[], layout: TensorLayout) {
    if (shape.length < 1 || shape.length > 3 || shape.some((d) => !Number.isInteger(d) || d < 1)) {
      throw new Error(`TensorView ${name}: shape [${shape.join(", ")}] must be 1-3 positive dims`);
    }
    this.name = name;
    this.shape = shape.slice();
    this.layout = layout;

    const [slabs, rows, cols] = normalizeDims(shape);
    const count = slabs * rows * cols;
    const { cellSize, gap, origin } = layout;
    const slabGap = layout.slabGap ?? 4 * cellSize;
    const pitch = cellSize + gap;

    const geometry = new THREE.BoxGeometry(cellSize, cellSize, cellSize);
    this.material = new THREE.ShaderMaterial({
      uniforms: { uDim: { value: 0 }, uHighlight: { value: 0 } },
      vertexShader: VERTEX,
      fragmentShader: FRAGMENT,
    });
    this.mesh = new THREE.InstancedMesh(geometry, this.material, count);
    this.mesh.name = name;
    // Positions live in instance matrices; geometry bounds alone would cull wrongly.
    this.mesh.frustumCulled = false;

    const m = new THREE.Matrix4();
    let i = 0;
    for (let s = 0; s < slabs; s++) {
      for (let r = 0; r < rows; r++) {
        for (let c = 0; c < cols; c++) {
          m.makeTranslation(origin[0] + c * pitch, origin[1] - r * pitch, origin[2] - s * slabGap);
          this.mesh.setMatrixAt(i++, m);
        }
      }
    }

    // Per-instance color, initialized to the colormap's zero (near-white).
    const colors = new Float32Array(count * 3);
    const zero: RGB = { r: 0, g: 0, b: 0 };
    valueColor(0, 1, zero);
    for (let k = 0; k < count; k++) {
      colors[3 * k] = zero.r;
      colors[3 * k + 1] = zero.g;
      colors[3 * k + 2] = zero.b;
    }
    this.mesh.instanceColor = new THREE.InstancedBufferAttribute(colors, 3);
  }

  /**
   * Map tensor values to instance colors. Throws on shape mismatch.
   * Default scale is tensorScale(t) (max |finite| value).
   */
  setValues(t: T, scale?: number): void {
    if (t.shape.length !== this.shape.length || t.shape.some((d, k) => d !== this.shape[k])) {
      throw new Error(
        `TensorView ${this.name}: shape mismatch — view [${this.shape.join(", ")}], ` +
          `tensor [${t.shape.join(", ")}]`,
      );
    }
    const s = scale ?? tensorScale(t);
    const attr = this.mesh.instanceColor!;
    const out = attr.array as Float32Array;
    const rgb: RGB = { r: 0, g: 0, b: 0 };
    for (let k = 0; k < t.data.length; k++) {
      valueColor(t.data[k], s, rgb);
      out[3 * k] = rgb.r;
      out[3 * k + 1] = rgb.g;
      out[3 * k + 2] = rgb.b;
    }
    attr.needsUpdate = true;
    this.lastValues = new Float32Array(t.data);
    this.scale = s;
  }

  /** Instance index -> coordinate in this view's shape (rank-matched). */
  indexToCoord(i: number): number[] {
    return unflattenIndex(i, this.shape);
  }

  /** Instance index -> normalized {slab, row, col} (missing leading dims = 0). */
  coordOf(i: number): { slab: number; row: number; col: number } {
    const coord = this.indexToCoord(i);
    const pad = 3 - coord.length;
    return {
      slab: pad > 0 ? 0 : coord[0],
      row: pad > 1 ? 0 : coord[coord.length - 2],
      col: coord[coord.length - 1],
    };
  }

  setHighlight(on: boolean): void {
    this.material.uniforms.uHighlight.value = on ? 1 : 0;
  }

  setDim(on: boolean): void {
    this.material.uniforms.uDim.value = on ? 1 : 0;
  }

  dispose(): void {
    this.mesh.geometry.dispose();
    this.material.dispose();
  }
}

/** (slabs, rows, cols) view of a 1-3D shape: 1D -> (1,1,n), 2D -> (1,r,c). */
function normalizeDims(shape: readonly number[]): [number, number, number] {
  if (shape.length === 1) return [1, 1, shape[0]];
  if (shape.length === 2) return [1, shape[0], shape[1]];
  return [shape[0], shape[1], shape[2]];
}
