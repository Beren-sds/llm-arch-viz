/**
 * Activation recording for the TS forward passes, mirroring the `record`
 * dict of training/llmviz_train/{mamba,gpt}.py. Forward functions call
 * `rec?.record(name, tensor)` with exactly the names the Python reference
 * records; the golden tests then diff a MapRecorder against goldens.json.
 */

import type { T } from "./tensor";

export interface Recorder {
  record(name: string, t: T): void;
}

/** Collects activations into a Map, cloning each tensor on record. */
export class MapRecorder implements Recorder {
  readonly activations = new Map<string, T>();

  record(name: string, t: T): void {
    if (this.activations.has(name)) {
      throw new Error(`duplicate activation name "${name}"`);
    }
    // Clone: forward passes may mutate buffers after recording (the
    // selective-scan h state is snapshotted once per token).
    this.activations.set(name, t.clone());
  }
}
