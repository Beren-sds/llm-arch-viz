# Performance notes (Phase 1)

**Date:** 2026-06-13

## What is measured here vs. what needs your GPU

Frame **rate** is not measured in CI: the headless screenshot harness uses a
software rasterizer (SwiftShader/ANGLE), which runs the scenes at ~2 fps
regardless of how fast the real pipeline is. That number says nothing about
GPU performance and is not reported as one.

What *is* measured deterministically (and guarded by `src/scenes/perf.test.ts`)
is the instance/draw-call budget, which is what the 60 fps target actually
hinges on for this instanced-cube renderer.

## Measured budget (src/scenes/perf.test.ts)

One `InstancedMesh` per tensor (= one draw call); each cube is one instance.
Re-run `npx vitest run src/scenes/perf.test.ts --disable-console-intercept`
to refresh.

| Scene | Draw calls (tensors) | Cube instances | Budget | Headroom |
|---|---|---|---|---|
| Mamba | 84 | 100,062 | 2,000,000 | 20.0× |
| GPT   | 54 |  97,596 | 2,000,000 | 20.5× |

Both scenes sit ~20× under the design doc's ≤~2M-instance budget. Labels add
~50–60 troika text meshes per scene (a few dozen more draw calls), still well
within a discrete/Apple-Silicon GPU's comfort zone.

Per-frame CPU work is all O(views) or O(1) and touches no GL beyond the draw:
camera tour interpolation, `OrbitControls.update`, the timeline player,
billboard facing (one quaternion per label), and at most one raycast per
frame for hover picking. The Mamba scan chapter additionally re-runs the
TS forward on a token prefix per step (~450 ms apart), not per frame.

## Verify 60 fps on your machine (M-series, real Chrome)

The deterministic budget predicts comfortable 60 fps, but frame rate must be
confirmed on real hardware. Procedure:

1. `npm run dev`
2. In Chrome (Apple Silicon), open both:
   - `http://localhost:5173/llm-arch-viz/#/mamba`
   - `http://localhost:5173/llm-arch-viz/#/gpt`
3. The dev build shows a live FPS readout top-left (`.fps-stats`, dev-only).
   For an independent meter, also enable Chrome DevTools → ⋮ → More tools →
   Rendering → "Frame Rendering Stats".
4. Exercise the worst cases: orbit/zoom continuously; play the **Mamba scan**
   chapter (per-token forward + h-state updates); switch chapters (camera
   fly-to); hover cells (picking).
5. Target: readout stays **≥ 55 fps** throughout (design goal 60). Record the
   observed range below.

### Observed (fill in after running on the target machine)

| Scene | Idle (orbit) | Scan / animation | Notes |
|---|---|---|---|
| Mamba | _TBD_ | _TBD_ | |
| GPT   | _TBD_ | _TBD_ | |

## Optimization headroom (not needed at current scale)

If a future, larger scene ever approaches the budget:

- Merge the many static weight tensors into fewer `InstancedMesh`es (one mesh
  per region instead of per tensor) to cut draw calls.
- Swap the box geometry for camera-facing planes (6× fewer vertices/cube),
  losing only the faint fake-shaded cube edges.
- Cap the live h-state snapshots rendered at once (Mamba already shows one).

None are applied in Phase 1: at ~100k instances / <90 draw calls per scene
there is no measured need.
