/**
 * The surface a built scene exposes to the page layer (archPage) and to its
 * chapter builder. Every per-architecture scene (mamba, gpt, …) returns an
 * object satisfying this, so the page driver is architecture-agnostic.
 *
 * It is a structural superset target: a scene MAY expose more (e.g. Mamba's
 * showHState), but must at least provide these.
 */

import type * as THREE from "three";
import type { TensorView } from "../engine/tensorView";
import type { CameraKeyframe } from "../engine/cameraTour";
import type { SceneBinding } from "../walkthrough/timeline";

export interface SceneController {
  /** Tensor name (activation or weight) → its view. */
  views: Map<string, TensorView>;
  /** Timeline channel: runForward / applyActivations / highlight / dim. */
  binding: SceneBinding;
  /** Whole-scene framing (also reachable as anchors.get('home')). */
  cameraHome: CameraKeyframe;
  /** Named viewpoints the chapters fly to. */
  anchors: Map<string, CameraKeyframe>;
  /** Run the full forward on `tokens` and push every activation into the views. */
  setTokens(tokens: number[]): void;
  /** Per-frame: face billboarded labels toward the camera. */
  update(camera: THREE.Camera): void;
  /** Label/bracket objects to await (labelsReady) before a screenshot. */
  labelObjects: THREE.Object3D[];
  dispose(): void;
}
