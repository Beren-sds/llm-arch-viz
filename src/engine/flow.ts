/**
 * Flow segments: thin additive-glow tubes used to draw the residual-stream
 * spine threading the centre column of a scene, so it reads as a connected
 * circuit rather than a stack of separate grids. Unlit + additive blending
 * gives a soft glow against the dark background without any post-processing.
 */

import * as THREE from "three";
import { BLOOM_LAYER } from "./scene";

/** Default spine accent — a cool blue that reads as "the stream". */
export const FLOW_COLOR = 0x6fa8ff;

export interface FlowOptions {
  color?: number;
  /** Tube radius in world units. Default 0.45. */
  radius?: number;
  /** Additive opacity. Default 0.55. */
  opacity?: number;
}

type Vec3 = readonly [number, number, number];

/**
 * A glowing tube from `from` to `to`. Cheap hexagonal cylinder, oriented by
 * a single quaternion, additive + depthWrite:false so overlaps only ever add
 * light. Caller adds it to the scene and calls disposeFlow() on teardown.
 */
export function createFlowSegment(from: Vec3, to: Vec3, opts?: FlowOptions): THREE.Mesh {
  const a = new THREE.Vector3(from[0], from[1], from[2]);
  const b = new THREE.Vector3(to[0], to[1], to[2]);
  const dir = new THREE.Vector3().subVectors(b, a);
  const len = Math.max(dir.length(), 1e-4);
  const radius = opts?.radius ?? 0.45;

  const geometry = new THREE.CylinderGeometry(radius, radius, len, 6, 1, true);
  const material = new THREE.MeshBasicMaterial({
    color: opts?.color ?? FLOW_COLOR,
    transparent: true,
    opacity: opts?.opacity ?? 0.55,
    blending: THREE.AdditiveBlending,
    depthWrite: false,
  });
  const mesh = new THREE.Mesh(geometry, material);
  mesh.position.copy(a).addScaledVector(dir, 0.5); // segment midpoint
  // Cylinder is built along +Y; rotate that to the segment direction.
  mesh.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), dir.normalize());
  mesh.renderOrder = -1; // draw before the opaque cells
  mesh.layers.enable(BLOOM_LAYER); // selective bloom picks this up
  return mesh;
}

/** Dim wiring colour for computation-graph edges. */
export const EDGE_COLOR = 0x4a5870;

/**
 * A thin, faint computation-graph edge from `from` to `to` (e.g. a weight
 * feeding the activation it produces). Unlike the spine it does NOT bloom and
 * uses normal blending, so the wiring reads as quiet structure behind the
 * glowing stream. Caller adds it and disposes via disposeFlow().
 */
export function createEdge(from: Vec3, to: Vec3, opts?: FlowOptions): THREE.Mesh {
  const a = new THREE.Vector3(from[0], from[1], from[2]);
  const b = new THREE.Vector3(to[0], to[1], to[2]);
  const dir = new THREE.Vector3().subVectors(b, a);
  const len = Math.max(dir.length(), 1e-4);
  const radius = opts?.radius ?? 0.16;

  const geometry = new THREE.CylinderGeometry(radius, radius, len, 5, 1, true);
  const material = new THREE.MeshBasicMaterial({
    color: opts?.color ?? EDGE_COLOR,
    transparent: true,
    opacity: opts?.opacity ?? 0.5,
    depthWrite: false,
  });
  const mesh = new THREE.Mesh(geometry, material);
  mesh.position.copy(a).addScaledVector(dir, 0.5);
  mesh.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), dir.normalize());
  mesh.renderOrder = -2; // behind the spine and cells
  return mesh;
}

/** Release a flow segment's GPU resources. */
export function disposeFlow(mesh: THREE.Mesh): void {
  mesh.geometry.dispose();
  (mesh.material as THREE.Material).dispose();
}
