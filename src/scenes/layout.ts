/**
 * Shared, pure layout primitives for the 3D scenes (mamba, gpt, …): cell
 * geometry constants, axis-aligned rectangles in the world XY plane, and a
 * camera keyframe that frames a rectangle head-on. No three.js, no scene
 * state — just geometry, so both scenes share one tested implementation.
 */

import type { CameraKeyframe } from "../engine/cameraTour";

/** Cube edge length (world units). */
export const CELL = 1;
/** Gap between adjacent cells. */
export const GAP = 0.25;
/** Center-to-center spacing of adjacent cells. */
export const PITCH = CELL + GAP;

/** Vertical field of view of the scene camera — keep in sync with createSceneShell. */
const VFOV_RAD = (45 * Math.PI) / 180;
/** Anchor distances assume at least this viewport aspect for width fits. */
const ASSUMED_ASPECT = 16 / 9;

/** Axis-aligned rect in world XY (top > bottom, right > left). */
export interface Rect {
  left: number;
  right: number;
  top: number;
  bottom: number;
}

/** Smallest rect covering all inputs. */
export function union(...rects: Rect[]): Rect {
  const r = { ...rects[0] };
  for (const x of rects) {
    r.left = Math.min(r.left, x.left);
    r.right = Math.max(r.right, x.right);
    r.top = Math.max(r.top, x.top);
    r.bottom = Math.min(r.bottom, x.bottom);
  }
  return r;
}

/** Grow a rect outward by `by` on every side. */
export function pad(r: Rect, by: number): Rect {
  return { left: r.left - by, right: r.right + by, top: r.top + by, bottom: r.bottom - by };
}

/**
 * XY footprint of a grid: cols across (+x), rows down (−y). Uses the LAST
 * two dims, so a 3D (slabs, rows, cols) tensor reports its in-plane face
 * (the slabs recede in −z and don't widen the footprint). 1D → a single row.
 */
export function gridSize(shape: readonly number[]): { w: number; h: number } {
  const cols = shape[shape.length - 1];
  const rows = shape.length >= 2 ? shape[shape.length - 2] : 1;
  return { w: cols * PITCH - GAP, h: rows * PITCH - GAP };
}

/**
 * Camera keyframe framing `r`: pulled back along +z far enough that the rect
 * fits vertically (45° vfov) and horizontally (assuming a 16:9-ish
 * viewport), with a little breathing room. `angled` offsets the position
 * slightly up-right for a hint of depth on the cube grids.
 */
export function frameRect(r: Rect, angled: boolean): CameraKeyframe {
  const w = r.right - r.left;
  const h = r.top - r.bottom;
  const cx = (r.left + r.right) / 2;
  const cy = (r.top + r.bottom) / 2;
  const tan = Math.tan(VFOV_RAD / 2);
  // +CELL: the cells are unit cubes around z=0; keep the front faces inside.
  const dist = Math.max(h / 2 / tan, w / 2 / (tan * ASSUMED_ASPECT)) * 1.12 + CELL;
  const pos: [number, number, number] = angled
    ? [cx + dist * 0.16, cy + dist * 0.08, dist]
    : [cx, cy, dist];
  return { pos, target: [cx, cy, 0] };
}
