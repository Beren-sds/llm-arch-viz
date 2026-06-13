/**
 * Orbit controls — thin wrapper over three's battle-tested example
 * implementation so the rest of the engine depends on OUR surface
 * (target / update / dispose), not on three's example API. If we ever
 * need custom behavior (camera-tour blending, inertia tweaks) it lands
 * here without touching call sites.
 *
 * Interaction defaults: left-drag rotates, wheel/pinch zooms (distance
 * clamped), right-drag / two-finger pans. Damping on, so callers MUST
 * call update() every frame.
 */

import type { PerspectiveCamera, Vector3 } from "three";
import { OrbitControls as ThreeOrbitControls } from "three/examples/jsm/controls/OrbitControls.js";

export class OrbitControls {
  private readonly inner: ThreeOrbitControls;

  constructor(camera: PerspectiveCamera, domElement: HTMLElement) {
    this.inner = new ThreeOrbitControls(camera, domElement);
    this.inner.enableDamping = true;
    this.inner.dampingFactor = 0.08;
    this.inner.minDistance = 5;
    // The Mamba scene spine is ~750 world units tall; its home framing
    // sits ~950 out, so the zoom-out clamp must stay beyond that (camera
    // far plane is 2000 in createSceneShell).
    this.inner.maxDistance = 1500;
  }

  /** The point the camera orbits around; mutate or copy into it, then update(). */
  get target(): Vector3 {
    return this.inner.target;
  }

  /**
   * Subscribe to the user grabbing the controls (three's 'start' event:
   * pointer-down / wheel / touch). The camera-tour handoff wires this to
   * TourPlayer.cancel() so a user drag interrupts an in-flight fly-to.
   */
  onStart(cb: () => void): void {
    this.inner.addEventListener("start", cb);
  }

  /** Advance damping/inertia; call once per frame from the render loop. */
  update(): void {
    this.inner.update();
  }

  dispose(): void {
    this.inner.dispose();
  }
}
