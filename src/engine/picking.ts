/**
 * Hover picking: a THREE.Raycaster over registered TensorViews'
 * InstancedMeshes, resolving the hit instanceId back to tensor
 * coordinates and the value from `lastValues`.
 *
 * Throttle pattern: `pointermove` fires far more often than the display
 * refreshes, and a raycast per event is wasted work. The handler only
 * STORES the latest pointer position; `update()` — called once per frame
 * from the render loop — consumes it and runs at most ONE pick per frame,
 * reporting through `onPick`. `pointerleave` queues a null report the
 * same way so the tooltip hides on the next frame.
 */

import * as THREE from "three";
import { unflattenIndex, type TensorView } from "./tensorView";

export interface PickTarget {
  view: TensorView;
  /**
   * Human-readable producing expression for this tensor, shown in the
   * tooltip (e.g. "h = exp(Δ·A)·h + Δ·B·x"). Supplied per-tensor by scenes.
   */
  formula?: string;
}

export interface PickResult {
  view: TensorView;
  name: string;
  /** Coordinate in the view's shape (rank-matched: [i] / [r,c] / [s,r,c]). */
  indices: number[];
  value: number;
  formula?: string;
}

/**
 * Pure-math half of a pick: instanceId -> {name, indices, value}.
 * No GL or raycaster involved, so this is unit-testable directly.
 * Throws when instanceId is out of range for the view's shape.
 */
export function resolvePick(
  view: TensorView,
  instanceId: number,
): { name: string; indices: number[]; value: number } {
  const indices = unflattenIndex(instanceId, view.shape);
  return { name: view.name, indices, value: view.lastValues[instanceId] };
}

export class Picker {
  /** Called from update() with the pick result (or null) for the latest pointer position. */
  onPick: ((result: PickResult | null, clientX: number, clientY: number) => void) | null = null;

  private readonly camera: THREE.Camera;
  private readonly domElement: HTMLElement;
  private readonly raycaster = new THREE.Raycaster();
  private readonly ndc = new THREE.Vector2();
  private readonly targets = new Map<THREE.InstancedMesh, PickTarget>();

  /** Latest pointer position; null after pointerleave. */
  private latest: { x: number; y: number } | null = null;
  /** True when a pointer event arrived since the last update(). */
  private dirty = false;
  private _enabled = true;

  private readonly onPointerMove = (e: PointerEvent): void => {
    if (!this._enabled) return;
    this.latest = { x: e.clientX, y: e.clientY };
    this.dirty = true;
  };

  private readonly onPointerLeave = (): void => {
    this.latest = null;
    this.dirty = true;
  };

  constructor(camera: THREE.Camera, domElement: HTMLElement) {
    this.camera = camera;
    this.domElement = domElement;
    domElement.addEventListener("pointermove", this.onPointerMove);
    domElement.addEventListener("pointerleave", this.onPointerLeave);
  }

  get enabled(): boolean {
    return this._enabled;
  }

  /**
   * Gate the whole pick path (e.g. while a camera tour animates). Disabling
   * drops the held pointer position and queues ONE null report through the
   * dirty path — same as pointerleave — so the tooltip hides on the next
   * update(). While disabled, pointer events are ignored and pick() returns
   * null. Re-enabling resumes on the next pointer move.
   */
  set enabled(v: boolean) {
    if (v === this._enabled) return;
    this._enabled = v;
    if (!v) {
      this.latest = null;
      this.dirty = true;
    }
  }

  /**
   * Re-resolve the held pointer position on the next update() even though
   * the cursor has not moved — call after tensor VALUES change (the cell
   * under a stationary cursor now reads differently). No-op when no
   * position is held (after pointerleave / while disabled).
   */
  requestRepick(): void {
    if (this.latest !== null) this.dirty = true;
  }

  add(target: PickTarget): void {
    this.targets.set(target.view.mesh, target);
  }

  remove(view: TensorView): void {
    this.targets.delete(view.mesh);
  }

  /**
   * Raycast at client coordinates against all registered views.
   * Returns the nearest instance hit, or null.
   */
  pick(clientX: number, clientY: number): PickResult | null {
    if (!this._enabled) return null;
    const rect = this.domElement.getBoundingClientRect();
    if (rect.width <= 0 || rect.height <= 0) return null;
    this.ndc.set(
      ((clientX - rect.left) / rect.width) * 2 - 1,
      -((clientY - rect.top) / rect.height) * 2 + 1,
    );
    this.raycaster.setFromCamera(this.ndc, this.camera);
    // NOTE: Raycaster vs InstancedMesh first tests a LAZILY CACHED bounding
    // sphere. TensorView writes instance matrices once in its constructor
    // (before the first pick computes the sphere), so this is safe today —
    // but if instance matrices are ever rewritten after a pick (animated
    // layouts), call mesh.computeBoundingSphere() afterwards or rays will
    // be tested against the stale sphere and silently miss.
    const hits = this.raycaster.intersectObjects([...this.targets.keys()], false);
    for (const hit of hits) {
      if (hit.instanceId === undefined) continue;
      const target = this.targets.get(hit.object as THREE.InstancedMesh);
      if (!target) continue;
      const { name, indices, value } = resolvePick(target.view, hit.instanceId);
      const result: PickResult = { view: target.view, name, indices, value };
      if (target.formula !== undefined) result.formula = target.formula;
      return result;
    }
    return null;
  }

  /**
   * Consume the latest pointer event (if any) and report through onPick.
   * Call once per animation frame from the render loop.
   */
  update(): void {
    if (!this.dirty) return;
    this.dirty = false;
    if (this.latest === null) {
      this.onPick?.(null, 0, 0);
      return;
    }
    const { x, y } = this.latest;
    this.onPick?.(this.pick(x, y), x, y);
  }

  dispose(): void {
    this.domElement.removeEventListener("pointermove", this.onPointerMove);
    this.domElement.removeEventListener("pointerleave", this.onPointerLeave);
    this.targets.clear();
    this.latest = null;
    this.dirty = false;
  }
}
