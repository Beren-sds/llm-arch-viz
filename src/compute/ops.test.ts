import { describe, expect, it } from "vitest";
import { T } from "./tensor";
import {
  add,
  argmax,
  causalDepthwiseConv1d,
  embedding,
  gelu,
  layernorm,
  linear,
  exp,
  matmul,
  mul,
  rmsnorm,
  scale,
  silu,
  softmax,
  softplus,
} from "./ops";

describe("matmul", () => {
  it("computes a 2x3 by 3x2 product", () => {
    const a = T.from([1, 2, 3, 4, 5, 6], [2, 3]);
    const b = T.from([7, 8, 9, 10, 11, 12], [3, 2]);
    const c = matmul(a, b);
    expect(c.shape).toEqual([2, 2]);
    // row0: [1*7+2*9+3*11, 1*8+2*10+3*12] = [58, 64]
    // row1: [4*7+5*9+6*11, 4*8+5*10+6*12] = [139, 154]
    expect(Array.from(c.data)).toEqual([58, 64, 139, 154]);
  });

  it("does not mutate its inputs", () => {
    const a = T.from([1, 2], [1, 2]);
    const b = T.from([3, 4], [2, 1]);
    matmul(a, b);
    expect(Array.from(a.data)).toEqual([1, 2]);
    expect(Array.from(b.data)).toEqual([3, 4]);
  });

  it("throws on inner-dim mismatch and non-2D inputs", () => {
    expect(() => matmul(T.zeros([2, 3]), T.zeros([2, 3]))).toThrow();
    expect(() => matmul(T.zeros([2, 3, 4]), T.zeros([4, 2]))).toThrow();
    expect(() => matmul(T.zeros([3]), T.zeros([3, 2]))).toThrow();
  });
});

describe("linear", () => {
  it("applies y = x @ w^T + b with PyTorch (out,in) weight layout", () => {
    const x = T.from([1, 2], [1, 2]);
    const w = T.from([1, 0, 0, 1], [2, 2]); // identity
    const b = T.from([10, 20], [2]);
    const y = linear(x, w, b);
    expect(y.shape).toEqual([1, 2]);
    expect(Array.from(y.data)).toEqual([11, 22]);
  });

  it("works without bias", () => {
    const x = T.from([1, 2, 3, 4], [2, 2]);
    const w = T.from([1, 2, 3, 4], [2, 2]);
    const y = linear(x, w);
    // y[i,o] = sum_j x[i,j] * w[o,j]
    // y[0] = [1*1+2*2, 1*3+2*4] = [5, 11]; y[1] = [3*1+4*2, 3*3+4*4] = [11, 25]
    expect(Array.from(y.data)).toEqual([5, 11, 11, 25]);
  });

  it("throws on shape mismatches", () => {
    expect(() => linear(T.zeros([2, 3]), T.zeros([4, 2]))).toThrow(); // in-dim mismatch
    expect(() => linear(T.zeros([2, 3]), T.zeros([4, 3]), T.zeros([3]))).toThrow(); // bias len
    expect(() => linear(T.zeros([3]), T.zeros([4, 3]))).toThrow(); // x not 2D
  });
});

describe("embedding", () => {
  it("gathers rows from the table", () => {
    const table = T.from([0, 1, 2, 3, 4, 5], [3, 2]);
    const out = embedding([2, 0, 1], table);
    expect(out.shape).toEqual([3, 2]);
    expect(Array.from(out.data)).toEqual([4, 5, 0, 1, 2, 3]);
  });

  it("throws on out-of-range or non-integer ids", () => {
    const table = T.from([0, 1, 2, 3], [2, 2]);
    expect(() => embedding([2], table)).toThrow();
    expect(() => embedding([-1], table)).toThrow();
    expect(() => embedding([0.5], table)).toThrow();
  });
});

describe("rmsnorm", () => {
  it("matches the hand-computed plan case", () => {
    const out = rmsnorm(T.from([3, 4], [2]), T.from([1, 1], [2]));
    // 3 / sqrt((9+16)/2 + 1e-5)
    expect(out.data[0]).toBeCloseTo(3 / Math.sqrt(12.5 + 1e-5), 6);
    expect(out.data[1]).toBeCloseTo(4 / Math.sqrt(12.5 + 1e-5), 6);
  });

  it("normalizes each last-dim row independently and applies weight", () => {
    const x = T.from([3, 4, 1, 1], [2, 2]);
    const w = T.from([2, 1], [2]);
    const out = rmsnorm(x, w);
    const r0 = Math.sqrt((9 + 16) / 2 + 1e-5);
    const r1 = Math.sqrt(1 + 1e-5);
    expect(out.at(0, 0)).toBeCloseTo((3 / r0) * 2, 6);
    expect(out.at(0, 1)).toBeCloseTo(4 / r0, 6);
    expect(out.at(1, 0)).toBeCloseTo((1 / r1) * 2, 6);
    expect(out.at(1, 1)).toBeCloseTo(1 / r1, 6);
    // input untouched
    expect(Array.from(x.data)).toEqual([3, 4, 1, 1]);
  });

  it("throws when weight length differs from the last dim", () => {
    expect(() => rmsnorm(T.zeros([2, 3]), T.zeros([2]))).toThrow();
  });
});

describe("layernorm", () => {
  it("matches nn.LayerNorm with unit weight / zero bias", () => {
    const x = T.from([1, 2, 3, 4], [1, 4]);
    const out = layernorm(x, T.from([1, 1, 1, 1], [4]), T.zeros([4]));
    // mean 2.5, biased var = (2.25+0.25+0.25+2.25)/4 = 1.25
    const d = Math.sqrt(1.25 + 1e-5);
    expect(out.at(0, 0)).toBeCloseTo(-1.5 / d, 6);
    expect(out.at(0, 1)).toBeCloseTo(-0.5 / d, 6);
    expect(out.at(0, 2)).toBeCloseTo(0.5 / d, 6);
    expect(out.at(0, 3)).toBeCloseTo(1.5 / d, 6);
  });

  it("applies weight and bias after normalization", () => {
    const x = T.from([1, 2, 3, 4], [1, 4]);
    const out = layernorm(x, T.from([2, 2, 2, 2], [4]), T.from([1, 1, 1, 1], [4]));
    const d = Math.sqrt(1.25 + 1e-5);
    expect(out.at(0, 0)).toBeCloseTo((2 * -1.5) / d + 1, 6);
    expect(out.at(0, 3)).toBeCloseTo((2 * 1.5) / d + 1, 6);
  });

  it("throws on weight/bias length mismatch", () => {
    expect(() => layernorm(T.zeros([1, 4]), T.zeros([3]), T.zeros([4]))).toThrow();
    expect(() => layernorm(T.zeros([1, 4]), T.zeros([4]), T.zeros([3]))).toThrow();
  });
});

describe("softmax", () => {
  it("computes a stable softmax over the last dim", () => {
    const out = softmax(T.from([1, 2, 3], [1, 3]));
    // exp([-2,-1,0]) = [0.13533528, 0.36787944, 1]; sum = 1.50321472
    expect(out.at(0, 0)).toBeCloseTo(0.09003057, 6);
    expect(out.at(0, 1)).toBeCloseTo(0.24472847, 6);
    expect(out.at(0, 2)).toBeCloseTo(0.66524096, 6);
  });

  it("rows sum to 1", () => {
    const out = softmax(T.from([10, 20, 30, -5, 0, 5], [2, 3]));
    expect(out.at(0, 0) + out.at(0, 1) + out.at(0, 2)).toBeCloseTo(1, 6);
    expect(out.at(1, 0) + out.at(1, 1) + out.at(1, 2)).toBeCloseTo(1, 6);
  });

  it("maps -inf entries to exact 0 (causal mask rows)", () => {
    const out = softmax(T.from([0, -Infinity, 0], [1, 3]));
    expect(out.at(0, 0)).toBe(0.5);
    expect(out.at(0, 1)).toBe(0); // exact zero, not just small
    expect(out.at(0, 2)).toBe(0.5);
  });

  it("a single finite entry among -inf gets weight 1 (first causal row)", () => {
    const out = softmax(T.from([3.7, -Infinity, -Infinity], [1, 3]));
    expect(out.at(0, 0)).toBe(1);
    expect(out.at(0, 1)).toBe(0);
    expect(out.at(0, 2)).toBe(0);
  });

  it("an all--inf row yields NaN (documented: matches PyTorch; never occurs under a causal mask)", () => {
    const out = softmax(T.from([-Infinity, -Infinity], [1, 2]));
    expect(Number.isNaN(out.at(0, 0))).toBe(true);
    expect(Number.isNaN(out.at(0, 1))).toBe(true);
  });
});

describe("silu", () => {
  it("computes x * sigmoid(x)", () => {
    const out = silu(T.from([0, 1, -1], [3]));
    expect(out.data[0]).toBe(0);
    expect(out.data[1]).toBeCloseTo(0.7310586, 6); // 1/(1+e^-1)
    expect(out.data[2]).toBeCloseTo(-0.2689414, 6); // -1/(1+e)
  });
});

describe("gelu", () => {
  it("computes the exact erf variant (nn.GELU default)", () => {
    const out = gelu(T.from([0, 1, -2, 3], [4]));
    expect(out.data[0]).toBe(0);
    expect(out.data[1]).toBeCloseTo(0.8413447, 6); // 1 * Phi(1)
    expect(out.data[2]).toBeCloseTo(-0.0455003, 6); // -2 * Phi(-2)
    expect(out.data[3]).toBeCloseTo(2.9959502, 5); // 3 * Phi(3)
  });

  it("is odd-symmetric around the Phi weighting: gelu(x) + gelu(-x) = x for x>0", () => {
    // gelu(x) + gelu(-x) = x*Phi(x) - x*Phi(-x) = x*(2*Phi(x) - 1)... check identity directly:
    // gelu(1) - (1 - gelu(1)) relation: gelu(1) + gelu(-1) = 1*(Phi(1) - Phi(-1)) = erf(1/sqrt(2))
    const out = gelu(T.from([1, -1], [2]));
    expect(out.data[0] + out.data[1]).toBeCloseTo(0.6826895, 6); // erf(1/sqrt2)
  });
});

describe("softplus", () => {
  it("computes log1p(exp(x)) in the stable region", () => {
    const out = softplus(T.from([0, -1, 1], [3]));
    expect(out.data[0]).toBeCloseTo(Math.LN2, 6); // 0.6931472
    expect(out.data[1]).toBeCloseTo(0.3132617, 6); // log1p(e^-1)
    expect(out.data[2]).toBeCloseTo(1.3132617, 6); // 1 + log1p(e^-1)
  });

  it("returns x identically for x > 20 (PyTorch threshold)", () => {
    const out = softplus(T.from([25, 100], [2]));
    expect(out.data[0]).toBe(25);
    expect(out.data[1]).toBe(100); // exp(100) would overflow without the branch
  });
});

describe("causalDepthwiseConv1d", () => {
  it("matches a hand-computed single-channel case incl. left-edge zeros", () => {
    // T=3, C=1, dConv=3; kernels[0] = [1,2,3], bias = [10], x = [1,2,3]
    const x = T.from([1, 2, 3], [3, 1]);
    const kernels = T.from([1, 2, 3], [1, 3]);
    const bias = T.from([10], [1]);
    const out = causalDepthwiseConv1d(x, kernels, bias, 3);
    expect(out.shape).toEqual([3, 1]);
    // out[0] = 10 + 3*x[0]                    = 13   (x[-2], x[-1] are zero-padded)
    // out[1] = 10 + 2*x[0] + 3*x[1]           = 18
    // out[2] = 10 + 1*x[0] + 2*x[1] + 3*x[2]  = 24
    expect(Array.from(out.data)).toEqual([13, 18, 24]);
    expect(Array.from(x.data)).toEqual([1, 2, 3]); // input untouched
  });

  it("keeps channels independent (depthwise)", () => {
    // T=2, C=2, dConv=2; kernels = [[1,2],[3,4]], bias = [0,1]
    const x = T.from([1, 10, 2, 20], [2, 2]); // rows: t0=[1,10], t1=[2,20]
    const kernels = T.from([1, 2, 3, 4], [2, 2]);
    const bias = T.from([0, 1], [2]);
    const out = causalDepthwiseConv1d(x, kernels, bias, 2);
    // out[0,0] = 0 + 2*1           = 2
    // out[0,1] = 1 + 4*10          = 41
    // out[1,0] = 0 + 1*1 + 2*2     = 5
    // out[1,1] = 1 + 3*10 + 4*20   = 111
    expect(Array.from(out.data)).toEqual([2, 41, 5, 111]);
  });

  it("throws on inconsistent shapes", () => {
    expect(() =>
      causalDepthwiseConv1d(T.zeros([3, 2]), T.zeros([2, 4]), T.zeros([2]), 3),
    ).toThrow(); // kernels last dim != dConv
    expect(() =>
      causalDepthwiseConv1d(T.zeros([3, 2]), T.zeros([3, 4]), T.zeros([3]), 4),
    ).toThrow(); // channel mismatch x vs kernels
    expect(() =>
      causalDepthwiseConv1d(T.zeros([3, 2]), T.zeros([2, 4]), T.zeros([3]), 4),
    ).toThrow(); // bias length mismatch
  });
});

describe("add / mul", () => {
  it("adds elementwise", () => {
    const out = add(T.from([1, 2, 3], [3]), T.from([10, 20, 30], [3]));
    expect(Array.from(out.data)).toEqual([11, 22, 33]);
  });

  it("multiplies elementwise", () => {
    const out = mul(T.from([1, 2], [2]), T.from([3, 4], [2]));
    expect(Array.from(out.data)).toEqual([3, 8]);
  });

  it("does not mutate inputs", () => {
    const a = T.from([1, 2], [2]);
    const b = T.from([3, 4], [2]);
    add(a, b);
    mul(a, b);
    expect(Array.from(a.data)).toEqual([1, 2]);
    expect(Array.from(b.data)).toEqual([3, 4]);
  });

  it("throws on shape mismatch (same shape required, no broadcasting)", () => {
    expect(() => add(T.zeros([2, 3]), T.zeros([3, 2]))).toThrow();
    expect(() => add(T.zeros([2]), T.zeros([3]))).toThrow();
    expect(() => mul(T.zeros([2, 2]), T.zeros([4]))).toThrow();
  });
});

describe("exp", () => {
  it("computes Math.exp elementwise, preserving shape", () => {
    const out = exp(T.from([0, 1, -1], [3, 1]));
    expect(out.shape).toEqual([3, 1]);
    expect(out.data[0]).toBe(1);
    expect(out.data[1]).toBeCloseTo(Math.E, 6);
    expect(out.data[2]).toBeCloseTo(1 / Math.E, 6);
  });

  it("does not mutate its input", () => {
    const x = T.from([1, 2], [2]);
    exp(x);
    expect(Array.from(x.data)).toEqual([1, 2]);
  });
});

describe("scale", () => {
  it("multiplies every element by the scalar", () => {
    const out = scale(T.from([1, -2, 0.5], [3]), -1);
    expect(Array.from(out.data)).toEqual([-1, 2, -0.5]);
  });

  it("composes with exp to give A = -exp(A_log)", () => {
    const aLog = T.from([0, Math.log(2)], [1, 2]);
    const a = scale(exp(aLog), -1);
    expect(a.shape).toEqual([1, 2]);
    expect(a.data[0]).toBe(-1);
    expect(a.data[1]).toBeCloseTo(-2, 6);
  });

  it("does not mutate its input", () => {
    const x = T.from([3], [1]);
    scale(x, 5);
    expect(Array.from(x.data)).toEqual([3]);
  });
});

describe("argmax", () => {
  it("returns the index of the max along the last dim per row", () => {
    const x = T.from([1, 5, 3, 7, 2, 9], [2, 3]);
    expect(argmax(x)).toEqual([1, 2]);
  });

  it("handles 1D input as a single row", () => {
    expect(argmax(T.from([4, 1, 2], [3]))).toEqual([0]);
  });

  it("first occurrence wins on ties", () => {
    expect(argmax(T.from([2, 2, 1], [3]))).toEqual([0]);
  });
});
