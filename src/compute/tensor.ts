/**
 * Minimal row-major float32 tensor for re-executing the tiny PyTorch
 * models (training/llmviz_train/{mamba,gpt}.py) in the browser.
 *
 * Data is always a Float32Array; arithmetic in ops.ts accumulates in
 * f64 (plain JS numbers) and rounds to f32 on store, which stays within
 * the 1e-4 golden-test gate of PyTorch's f32-accumulated kernels.
 */

function checkShape(shape: readonly number[]): number {
  let size = 1;
  for (const d of shape) {
    if (!Number.isInteger(d) || d < 0) {
      throw new Error(`invalid shape [${shape.join(", ")}]: dims must be non-negative integers`);
    }
    size *= d;
  }
  if (shape.length === 0) {
    throw new Error("invalid shape []: rank must be >= 1");
  }
  return size;
}

export class T {
  readonly shape: readonly number[];
  readonly data: Float32Array;

  private constructor(shape: readonly number[], data: Float32Array) {
    this.shape = shape;
    this.data = data;
  }

  /** Build a tensor from values (copied) and a shape; throws on length mismatch. */
  static from(values: number[] | Float32Array, shape: readonly number[]): T {
    const size = checkShape(shape);
    if (values.length !== size) {
      throw new Error(
        `length mismatch: got ${values.length} values for shape [${shape.join(", ")}] (size ${size})`,
      );
    }
    // Float32Array.from copies and rounds each value to f32.
    return new T(shape.slice(), Float32Array.from(values));
  }

  /** All-zero tensor of the given shape. */
  static zeros(shape: readonly number[]): T {
    const size = checkShape(shape);
    return new T(shape.slice(), new Float32Array(size));
  }

  /** Number of elements (product of dims). */
  get size(): number {
    return this.data.length;
  }

  /** Bounds-checked row-major element access; arity must match rank. */
  at(...idx: number[]): number {
    if (idx.length !== this.shape.length) {
      throw new Error(`expected ${this.shape.length} indices, got ${idx.length}`);
    }
    let flat = 0;
    for (let d = 0; d < idx.length; d++) {
      const i = idx[d];
      if (!Number.isInteger(i) || i < 0 || i >= this.shape[d]) {
        throw new Error(
          `index ${i} out of range for dim ${d} of shape [${this.shape.join(", ")}]`,
        );
      }
      flat = flat * this.shape[d] + i;
    }
    return this.data[flat];
  }

  /** Independent deep copy (shape and data). */
  clone(): T {
    return new T(this.shape.slice(), this.data.slice());
  }
}
