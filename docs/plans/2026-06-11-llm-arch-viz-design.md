# Beyond Transformers — 3D Interactive Atlas of LLM Architectures

**Date:** 2026-06-11
**Status:** Approved
**Reference:** [bbycroft.net/llm](https://bbycroft.net/llm) (visual/pedagogical inspiration only — `bbycroft/llm-viz` has no license, so no code reuse)

## Goal

A bbycroft-style 3D tensor-level interactive tutorial site covering the major
non-transformer LLM architectures: Mamba (selective SSM), RWKV, MoE, KAN, plus
a compact Transformer baseline and a RetNet/Hyena/linear-attention family
chapter. Bilingual EN/ZH narration. Real toy-model weights computed live in
the browser — every visualized cell holds a true value.

## Core decisions

| Decision | Choice |
|---|---|
| Fidelity | Full 3D tensor-level (every weight matrix and activation rendered) |
| Engine | Three.js `InstancedMesh` + custom shaders (not raw WebGL, not React) |
| Stack | Vite + vanilla TypeScript, static site, GitHub Pages |
| Compute | Real toy weights trained in PyTorch, exported, TS forward pass in-browser |
| Language | EN/ZH toggle, narration in per-locale JSON |
| Flagship | Mamba first |

## The pedagogical through-line: one task, many machines

Every architecture is trained on the **same toy task — selective copying**
(remember the data tokens, ignore the noise tokens, emit the data at the end).
This is the task that motivates Mamba's selection mechanism, and GPT, RWKV,
and MoE-GPT can all learn it. Each architecture page is "same problem,
different machine": the user watches real weights solve the identical input,
which makes cross-architecture comparison honest.

Toy scale (per architecture, tuned as needed): ~2 layers, d_model ≈ 32–48,
vocab ≈ 16, sequence length ≈ 24. Small enough that every tensor is legible
in 3D and the forward pass runs instantly in TS.

## Architecture

```
llm-arch-viz/
├── training/            # Python/PyTorch (uv-managed, lockfile committed)
│   ├── tasks/           # selective-copying dataset generation (seeded)
│   ├── models/          # minimal reference impls (mamba.py, gpt.py, rwkv.py, ...)
│   ├── export.py        # weights → .bin + manifest.json; goldens → goldens.json
│   └── config.yaml      # single seed source, dims, training hparams
├── src/
│   ├── compute/         # TS tensor runtime (Float32Array ops),
│   │                    # per-arch forward pass emitting named activations
│   ├── engine/          # Three.js: instanced tensor renderer, camera tour,
│   │                    # GPU picking/hover tooltips, SDF labels, dim brackets
│   ├── walkthrough/     # chapter system: {camera keyframe, highlights,
│   │                    # timeline animation, narration key}; deep links
│   ├── i18n/            # en.json / zh.json narration, toggle in localStorage
│   ├── scenes/          # per-architecture 3D layout definitions
│   └── pages/           # landing (architecture cards) + per-arch pages
└── docs/plans/
```

### Engine notes

- One `InstancedMesh` per tensor; per-instance color attribute encodes value;
  custom `ShaderMaterial` drives highlight/pulse "data flowing" animation.
  Budget: ≤ ~2M instances per scene, 60 fps target on Apple Silicon.
- Camera tour: keyframed paths bound to chapters; click-to-step or scroll.
- Hover any cell → tensor name, indices, true value, producing formula.
- Shared visual grammar across scenes: token/residual stream is the vertical
  spine, weight matrices flank it.

### Mamba tour centerpiece

Time-stepped SSM scan: hidden state h updates token-by-token; the selection
gate Δ visibly closes on noise tokens and opens on copy tokens. The B/C/Δ
projections are shown per-token (the "selective" part), contrasted with a
static-A LTI recurrence.

## Verifiable success criteria

1. **Compute correctness gate (blocks all scene work):** TS forward pass
   activations match PyTorch golden activations within 1e-4 (vitest).
2. Phase 1 ships: Mamba page with full guided bilingual tour + compact GPT
   baseline scene, deployed on GitHub Pages.
3. 60 fps during camera tour on M-series at default scene scale.

## Phasing

- **Phase 1:** engine + walkthrough + i18n framework + Mamba (flagship tour)
  + compact GPT baseline scene.
- **Phase 2:** RWKV, MoE.
- **Phase 3:** KAN ("non-MLP alternative" chapter), RetNet/Hyena/linear-attn.

## Error handling

- Weight/manifest fetch failure → visible error state, never silent fallback
  to fake data.
- No WebGL2 → clear unsupported-browser message.
- TS/golden mismatch → test failure; scenes refuse to claim "real values"
  status (no synthetic stand-ins, per repo red lines).

## Testing

- Golden-activation tests per architecture (the load-bearing gate).
- Engine smoke tests (scene builds, tensor counts, chapter registry) in vitest.
- Manual visual acceptance per chapter before a page is listed on landing.
