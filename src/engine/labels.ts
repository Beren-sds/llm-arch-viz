/**
 * 3D text labels + dimension brackets for tensor views.
 *
 * Text rendering is troika-three-text (SDF glyphs — crisp at any zoom).
 * This module is the ONLY place allowed to import troika; the rest of the
 * engine talks to createTensorLabel / createDimBracket / disposeLabel /
 * labelsReady so the text backend stays swappable.
 *
 * ASYNC CAVEAT: troika typesets glyphs (and fetches its default font) off
 * the main thread. A freshly created label renders as NOTHING for a few
 * frames until its sync completes — await labelsReady(...) when a caller
 * (e.g. the screenshot harness) must know text is actually on screen.
 */

import * as THREE from "three";
import { Text } from "troika-three-text";

export const LABEL_COLOR = "#9fb3d9";
export const LABEL_FONT_SIZE = 3;
const DEFAULT_BRACKET_OFFSET = 2;

export interface LabelOptions {
  /** Font size in world units. Default LABEL_FONT_SIZE (3). */
  size?: number;
  /** Fill color. Default LABEL_COLOR (#9fb3d9). */
  color?: string | number;
}

/**
 * A floating text label (e.g. a tensor name) anchored centered-top: the
 * object's position is the top-center of the text block and glyphs extend
 * downward (-y) from it. Caller positions it. Font and sdfGlyphSize are
 * troika defaults. Pair with makeBillboard() if it should face the camera.
 */
export function createTensorLabel(text: string, opts?: LabelOptions): THREE.Object3D {
  const t = new Text();
  t.text = text;
  t.fontSize = opts?.size ?? LABEL_FONT_SIZE;
  t.color = opts?.color ?? LABEL_COLOR;
  t.anchorX = "center";
  t.anchorY = "top";
  t.sync(); // kick off async glyph generation immediately
  return t;
}

type Vec3 = readonly [number, number, number];

export interface BracketGeometry {
  /** Polyline tick → span → tick (4 points), like ⌐___¬ rotated to the span. */
  points: [number, number, number][];
  /** Label anchor: span midpoint pushed past the tick ends. */
  labelPos: [number, number, number];
  /** 0 for horizontal spans, π/2 for vertical — never upside-down. */
  labelRotationZ: number;
}

/**
 * Pure math for a dimension bracket along the from→to segment.
 *
 * perp is the +90° rotation of the span direction IN THE XY PLANE
 * ((dx,dy) → (-dy,dx), z untouched) — brackets annotate tensor grids that
 * live in XY, so the offset side is fully determined by the offset SIGN
 * relative to that perp. The span line sits at `from/to + perp·offset`;
 * ticks of length |offset|/2 extend OUTWARD (away from the tensor, toward
 * the offset side); labelPos sits at midpoint + perp·2·offset, i.e. one
 * tick-length beyond the tick ends.
 *
 * labelRotationZ = atan2(dy,dx) normalized into (-π/2, π/2] so the label
 * text never renders upside-down: horizontal spans (either direction) → 0,
 * vertical spans (either direction) → π/2 (reads bottom-to-top).
 *
 * Throws when from and to coincide in XY (perpendicular undefined).
 */
export function bracketGeometry(from: Vec3, to: Vec3, offset: number): BracketGeometry {
  const dx = to[0] - from[0];
  const dy = to[1] - from[1];
  const len = Math.hypot(dx, dy);
  if (len === 0) {
    throw new Error("bracketGeometry: from and to must differ in the XY plane");
  }
  // Unit perpendicular (in XY) and the outward direction on the offset side.
  const px = -dy / len;
  const py = dx / len;
  const sign = offset < 0 ? -1 : 1;
  const tick = Math.abs(offset) / 2;
  const ox = px * sign * tick; // tick vector, pointing away from the tensor
  const oy = py * sign * tick;

  const ax = from[0] + px * offset;
  const ay = from[1] + py * offset;
  const bx = to[0] + px * offset;
  const by = to[1] + py * offset;

  let rot = Math.atan2(dy, dx);
  if (rot > Math.PI / 2) rot -= Math.PI;
  else if (rot <= -Math.PI / 2) rot += Math.PI;

  return {
    points: [
      [ax + ox, ay + oy, from[2]],
      [ax, ay, from[2]],
      [bx, by, to[2]],
      [bx + ox, by + oy, to[2]],
    ],
    labelPos: [
      (from[0] + to[0]) / 2 + px * 2 * offset,
      (from[1] + to[1]) / 2 + py * 2 * offset,
      (from[2] + to[2]) / 2,
    ],
    labelRotationZ: rot,
  };
}

export interface DimBracketOptions {
  from: [number, number, number];
  to: [number, number, number];
  /** Dimension annotation, e.g. "d_model = 48". */
  label: string;
  /**
   * Signed perpendicular offset of the bracket from the segment; the sign
   * picks the side (see bracketGeometry). Default 2.
   */
  offset?: number;
  /** Line + label color. Default LABEL_COLOR. */
  color?: string | number;
}

/**
 * A dimension bracket: thin ⌐___¬ line along from→to plus a Text label
 * centered outside it, rotated to match the span (0 / π/2). Brackets are
 * FIXED in the tensor's plane — do NOT billboard them (the whole point is
 * that they measure an edge of the grid; a camera-facing bracket would
 * detach from what it annotates). Dispose via disposeLabel().
 */
export function createDimBracket(opts: DimBracketOptions): THREE.Object3D {
  const { points, labelPos, labelRotationZ } = bracketGeometry(
    opts.from,
    opts.to,
    opts.offset ?? DEFAULT_BRACKET_OFFSET,
  );
  const color = opts.color ?? LABEL_COLOR;

  const geometry = new THREE.BufferGeometry().setFromPoints(
    points.map(([x, y, z]) => new THREE.Vector3(x, y, z)),
  );
  const line = new THREE.Line(geometry, new THREE.LineBasicMaterial({ color }));

  const label = new Text();
  label.text = opts.label;
  label.fontSize = LABEL_FONT_SIZE;
  label.color = color;
  label.anchorX = "center";
  label.anchorY = "middle";
  label.position.set(labelPos[0], labelPos[1], labelPos[2]);
  label.rotation.z = labelRotationZ;
  label.sync();

  const group = new THREE.Group();
  group.name = `bracket:${opts.label}`;
  group.add(line, label);
  return group;
}

/**
 * Per-frame billboard updater: rotates `obj` to face the camera by copying
 * its quaternion. Use for free-floating labels (tensor names) that must
 * stay readable from any angle; NOT for brackets or their dimension labels,
 * which are aligned to the grid plane on purpose.
 */
export function makeBillboard(obj: THREE.Object3D): (camera: THREE.Camera) => void {
  return (camera) => {
    obj.quaternion.copy(camera.quaternion);
  };
}

/**
 * Release GL resources of anything produced by this module. troika Text
 * needs its own .dispose() (glyph geometry + derived material); plain Line
 * children get geometry/material disposed directly. Safe on nested groups.
 */
export function disposeLabel(obj: THREE.Object3D): void {
  obj.traverse((child) => {
    if (child instanceof Text) {
      child.dispose();
    } else if (child instanceof THREE.Line) {
      child.geometry.dispose();
      (child.material as THREE.Material).dispose();
    }
  });
  obj.removeFromParent();
}

/**
 * Resolves when every troika Text inside the given objects has finished its
 * async typeset (font fetched, SDF glyphs built) — i.e. the text is really
 * drawable. The screenshot harness waits on this through the page's
 * `document.body.dataset.settled` flag; see main.ts.
 *
 * Implementation note: troika's sync(callback) silently DROPS the callback
 * when no new sync is needed at call time (e.g. the creator's eager sync()
 * is still in flight) — so this waits on the 'synccomplete' EVENT, which
 * fires for every completed sync, and short-circuits via textRenderInfo
 * when the text already finished before we attached.
 */
export function labelsReady(...objs: THREE.Object3D[]): Promise<void> {
  const pending: Promise<void>[] = [];
  for (const obj of objs) {
    obj.traverse((child) => {
      if (child instanceof Text) {
        pending.push(
          new Promise((resolve) => {
            const finish = (): void => {
              child.removeEventListener("synccomplete", finish);
              resolve();
            };
            child.addEventListener("synccomplete", finish);
            child.sync(); // no-op if synced/in-flight; starts one if dirty
            if (child.textRenderInfo) finish(); // already drawable
          }),
        );
      }
    });
  }
  return Promise.all(pending).then(() => undefined);
}
