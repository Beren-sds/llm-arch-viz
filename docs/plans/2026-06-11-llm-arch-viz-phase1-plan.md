# llm-arch-viz Phase 1 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Ship the Phase 1 site: a bbycroft-style 3D tensor-level interactive tutorial with a Mamba flagship page (real toy weights, bilingual guided tour) plus a compact GPT baseline scene, deployed to GitHub Pages.

**Architecture:** Two halves connected by a binary weight format and golden activations. (1) `training/` — PyTorch trains tiny Mamba and GPT models on the selective-copying task and exports weights + golden intermediate activations. (2) `src/` — a TS tensor runtime re-runs the exact forward pass in-browser (gated by golden tests at 1e-4), and a Three.js instanced-mesh engine renders every tensor as a 3D cell grid with camera tours, picking, and EN/ZH narration.

**Tech Stack:** Vite, TypeScript, Three.js, troika-three-text, vitest (web); Python 3.11+, uv, PyTorch (training). GitHub Pages via Actions.

**Design doc:** `docs/plans/2026-06-11-llm-arch-viz-design.md`

**Red lines (from user global config):** no mock/synthetic weights standing in for real ones; golden test failures block scene work; no test skipping or tolerance inflation; all randomness from the single seed in `training/config.yaml`.

---

## Model + task spec (single source of truth)

**Task — selective copying.** Vocab (16 ids): `0=PAD`, `1..8` data, `9=NOISE`, `10=GO`, `11..15` reserved. An example: 16 context positions holding 4 data tokens at random positions (rest `NOISE`), then `GO`, then the 4 data tokens in order. Sequence length 21. Loss only on the 4 answer positions. Train/val split generated deterministically from config seed; val accuracy target ≥ 0.99 for both models before export.

**Mamba toy:** `n_layer=2, d_model=48, d_state=8, d_conv=4, expand=2 (d_inner=96), vocab=16`. Untied embedding/lm_head (clearer to visualize).

**GPT toy:** `n_layer=2, d_model=48, n_head=3 (head_dim=16), mlp_ratio=4, vocab=16`, learned positional embedding, untied head.

**Weight export format:** `public/models/<arch>/manifest.json` (arch, dims, tensor table: name/shape/offset/length) + `weights.bin` (little-endian float32, concatenated in manifest order) + `goldens.json` (2 fixed eval inputs; for each, every named activation with shape + flattened values).

**Activation naming convention** (shared TS/Python, this exact grammar):
`embed.out`, `layer{i}.norm.out`, `layer{i}.<module>.<port>`, `final_norm.out`, `head.logits`. Mamba modules: `in_proj`, `conv`, `x_proj`, `dt`, `ssm` (ports `h.t{t}` for per-token hidden state, `out`), `gate`, `out_proj`. GPT modules: `attn` (ports `q,k,v,scores,weights,out`), `mlp` (ports `fc,act,proj`).

---

### Task 1: Scaffold web project

**Files:** Create `package.json`, `vite.config.ts`, `tsconfig.json`, `index.html`, `src/main.ts`, `.gitignore`.

**Step 1:** `cd ~/Desktop/GitHub/llm-arch-viz && npm create vite@latest . -- --template vanilla-ts` (accept files into existing dir), then `npm i three troika-three-text && npm i -D vitest @types/three`.
**Step 2:** Add to `package.json` scripts: `"test": "vitest run"`. Set `base: '/llm-arch-viz/'` in `vite.config.ts` (GitHub Pages path).
**Step 3:** Verify: `npm run build` succeeds; `npm test` reports "no test files" (exit 0 with `--passWithNoTests` flag added to script).
**Step 4:** Commit `chore: scaffold vite+ts+three project`.

### Task 2: Scaffold training env

**Files:** Create `training/pyproject.toml` (uv project, deps: `torch`, `numpy`, `pyyaml`, `pytest`), `training/config.yaml`, `training/llmviz_train/__init__.py`, `training/llmviz_train/seed.py`.

**Step 1:** `cd training && uv init --name llmviz-train && uv add torch numpy pyyaml && uv add --dev pytest`. Commit `uv.lock`.
**Step 2:** `config.yaml`:

```yaml
seed: 1128
task: {vocab_size: 16, n_data: 4, context_len: 16, data_ids: [1,2,3,4,5,6,7,8], noise_id: 9, go_id: 10}
mamba: {n_layer: 2, d_model: 48, d_state: 8, d_conv: 4, expand: 2}
gpt: {n_layer: 2, d_model: 48, n_head: 3, mlp_ratio: 4}
train: {steps: 20000, batch_size: 256, lr: 1.0e-3, val_size: 2048, val_acc_target: 0.99}
```

**Step 3:** `seed.py`: `def seed_all(seed: int)` seeding `random`, `numpy`, `torch` (and returning a `torch.Generator`). All later code takes seeds only from config.
**Step 4:** Commit `chore: scaffold training env with locked deps`.

### Task 3: Selective-copying dataset (TDD)

**Files:** Create `training/llmviz_train/task.py`, `training/tests/test_task.py`.

**Step 1 (failing test):**

```python
import torch
from llmviz_train.task import make_batch, TASK

def test_batch_shapes_and_structure():
    x, y, mask = make_batch(32, torch.Generator().manual_seed(0))
    assert x.shape == (32, 21) and y.shape == (32, 21) and mask.shape == (32, 21)
    assert (x[:, 16] == TASK.go_id).all()          # GO at position 16
    assert mask[:, 16:20].all() and mask[:, :16].sum() == 0  # loss on 4 answer preds only
    ctx = x[:, :16]
    assert ((ctx == TASK.noise_id) | ((ctx >= 1) & (ctx <= 8))).all()
    for b in range(32):                              # answers = data tokens in order
        data = ctx[b][ctx[b] != TASK.noise_id]
        assert torch.equal(y[b, 16:20], data)

def test_determinism():
    a = make_batch(8, torch.Generator().manual_seed(7))
    b = make_batch(8, torch.Generator().manual_seed(7))
    assert all(torch.equal(p, q) for p, q in zip(a, b))
```

`x` is the input sequence (positions 17..20 hold the previous answer tokens for autoregressive teacher forcing; position 20's target is the 4th data token — define `y` as next-token targets aligned so `y[t]` is the target for input position `t`).
**Step 2:** Run `uv run pytest tests/test_task.py -v` → FAIL (module missing).
**Step 3:** Implement `make_batch(batch, gen) -> (x, y, mask)` per spec; exactly 4 data tokens at sorted-random context positions; teacher-forced answer region.
**Step 4:** Tests pass. **Step 5:** Commit `feat(training): selective-copying task generator`.

### Task 4: Minimal Mamba reference implementation (TDD)

**Files:** Create `training/llmviz_train/mamba.py`, `training/tests/test_mamba.py`.

**Step 1 (failing tests):** shape test (logits `(B,T,vocab)`); causality test (perturbing token t+1 must not change logits at ≤ t); activation-recording test (forward with `record=True` returns dict containing all names from the naming convention, e.g. `layer0.ssm.h.t0` with shape `(d_inner, d_state)` for batch item 0).
**Step 2:** Run → FAIL.
**Step 3:** Implement minimal Mamba (no CUDA scan; sequential python-loop scan is fine at this scale):

```python
class MambaBlock(nn.Module):
    # in_proj: d_model -> 2*d_inner (x, z)
    # conv1d: depthwise causal, kernel d_conv, groups=d_inner
    # x_proj: d_inner -> dt_rank + 2*d_state   (dt_rank = ceil(d_model/16) = 3)
    # dt_proj: dt_rank -> d_inner ; delta = softplus(dt_proj(dt) + bias)
    # A = -exp(A_log)  (d_inner, d_state); D skip (d_inner,)
    # scan: h_t = exp(delta_t A) * h_{t-1} + delta_t * B_t * x_t ; y_t = C_t · h_t + D x_t
    # out = out_proj(y * silu(z))
```

Norm = RMSNorm. Record every named activation for batch item 0 when `record=True`.
**Step 4:** Tests pass. **Step 5:** Commit `feat(training): minimal Mamba reference impl with activation recording`.

### Task 5: Minimal GPT reference implementation (TDD)

**Files:** Create `training/llmviz_train/gpt.py`, `training/tests/test_gpt.py`.

Same step pattern as Task 4: shape, causality (causal mask), activation recording (`layer0.attn.weights` shape `(n_head, T, T)` etc.). Commit `feat(training): minimal GPT reference impl`.

### Task 6: Train both models to criterion

**Files:** Create `training/llmviz_train/train.py`, `training/run_train.py`.

**Step 1:** Training loop: AdamW, cosine decay, loss masked to answer positions, val accuracy = exact-match on the 4 answer tokens over a fixed seeded val set. Checkpoints to `training/outputs/<arch>/<git_short>/` atomically (`.tmp` → `os.replace`); save `best.pt` + `latest.pt` + resolved config + metrics JSONL (per user repro standards).
**Step 2:** `uv run python run_train.py --arch mamba` then `--arch gpt`.
**Verify:** printed AND JSONL-recorded final val_acc ≥ 0.99 for both. If a model plateaus below target, tune steps/lr in config (commit the change) — do NOT lower the target.
**Step 3:** Commit `feat(training): train loop; mamba+gpt reach >=0.99 val acc` (include metrics JSONL, exclude checkpoints via `.gitignore`, but record git_short + final metrics in commit message).

### Task 7: Export weights + goldens

**Files:** Create `training/export.py`, `training/tests/test_export.py`, output to `public/models/{mamba,gpt}/`.

**Step 1 (failing test):** round-trip test — export to tmpdir, reload `weights.bin` via manifest offsets with numpy, compare every tensor to the checkpoint state_dict exactly; goldens test — `goldens.json` has 2 inputs, each with `head.logits` whose argmax over answer positions equals the true answers.
**Step 2:** Implement: manifest tensor order = deterministic (sorted by name); float32 little-endian; 2 fixed eval inputs generated from config seed; record all named activations (batch item 0) into goldens.
**Step 3:** Run export for both archs; tests pass. Files land in `public/models/`.
**Step 4:** Commit `feat(training): weight+golden export to public/models` (the .bin/.json artifacts ARE committed — they're site assets, few hundred KB).

### Task 8: TS tensor runtime ops (TDD)

**Files:** Create `src/compute/tensor.ts`, `src/compute/ops.ts`, `src/compute/tensor.test.ts`.

**Step 1 (failing tests):** hand-computed cases for: `matmul` (2D×2D), `linear` (x·Wᵀ+b), `embedding`, `rmsnorm` (eps 1e-5), `softmax`, `silu`, `softplus`, `causalDepthwiseConv1d`, `argmax`. Example:

```ts
test('rmsnorm', () => {
  const x = T.from([3, 4], [2]);            // shape [2]
  const w = T.from([1, 1], [2]);
  expect(rmsnorm(x, w).data[0]).toBeCloseTo(3 / Math.sqrt((9 + 16) / 2 + 1e-5), 6);
});
```

**Step 2:** FAIL → **Step 3:** implement `Tensor` (shape + Float32Array, row-major) and ops. No external math deps. **Step 4:** PASS. **Step 5:** Commit `feat(compute): TS tensor runtime ops`.

### Task 9: Weight loader + manifest types (TDD)

**Files:** Create `src/compute/loader.ts`, `src/compute/loader.test.ts`.

Test against a tiny fixture manifest+bin built in-test (Uint8Array). Loader: fetch manifest + bin (in tests, injectable `fetch`), slice per tensor table into `Tensor`s, validate byte length matches manifest total — mismatch throws (no silent truncation). Commit `feat(compute): weight loader`.

### Task 10: TS Mamba forward pass — GOLDEN GATE

**Files:** Create `src/compute/mamba.ts`, `src/compute/mamba.golden.test.ts`.

**Step 1 (failing test):** load `public/models/mamba/{manifest.json,weights.bin,goldens.json}` from disk (vitest node env, `fs`), run `mambaForward(weights, input, recorder)`, assert **every** golden activation matches: `maxAbsDiff < 1e-4`, and report the worst offender name in the assertion message.
**Step 2:** FAIL → **Step 3:** implement forward emitting activations through a `Recorder` (`record(name, tensor)`), mirroring Task 4's reference exactly. **Step 4:** PASS — this unblocks all Mamba scene work. **Step 5:** Commit `feat(compute): TS Mamba forward matches PyTorch goldens <1e-4`.

### Task 11: TS GPT forward pass — golden gate

**Files:** `src/compute/gpt.ts`, `src/compute/gpt.golden.test.ts`. Same pattern as Task 10. Commit.

### Task 12: Colormap + tensor cell-grid geometry (TDD)

**Files:** Create `src/engine/colormap.ts`, `src/engine/tensorView.ts`, tests for both.

**Step 1 (failing tests):** colormap: diverging blue-white-red over [-max,+max], unit-test exact RGB at -max/0/+max and NaN→magenta. tensorView: `new TensorView(tensor, layout)` exposes `mesh: InstancedMesh` with `count === tensor.size`, instance (i,j) world position matches `layout` (cell size, gap, origin), and `setValues()` writes the colormap into the instance color buffer (assert buffer values; no GL context needed — three's scene graph works in node).
**Step 2-5:** FAIL → implement → PASS → commit `feat(engine): instanced tensor view with value colormap`.

Implementation notes: one `InstancedMesh(boxGeom, shaderMat, n)` per tensor; 2D tensors → (rows×cols) grid; 1D → row; 3D (e.g. per-head) → stacked slabs with z-offset. Custom `ShaderMaterial` uniforms: `uHighlight` (0..1 pulse), `uDim` (fade when not in focus); per-instance color attribute.

### Task 13: Scene shell + render loop (manual acceptance)

**Files:** Create `src/engine/scene.ts`, `src/engine/controls.ts`; modify `src/main.ts`.

Build: WebGL2 check (throw → visible error overlay if absent), renderer, scene, orbit controls (custom minimal or three's OrbitControls), resize handling, animation loop, stats overlay in dev. Render one dummy 48×96 TensorView.
**Verify (manual):** `npm run dev` — grid renders, orbits at 60fps. **Commit** `feat(engine): scene shell + render loop`.

### Task 14: Camera tour system (TDD)

**Files:** Create `src/engine/cameraTour.ts` + test.

Keyframes `{pos, target, t}`; cubic ease; `tourAt(time)` returns interpolated pos/target — unit-test endpoints, midpoint monotonicity, and clamping. `TourPlayer` advances per-frame, supports `goToChapter(i)` with animated transition. Commit `feat(engine): keyframed camera tour`.

### Task 15: Picking + tooltip (manual acceptance + unit test for index math)

**Files:** Create `src/engine/picking.ts`, `src/engine/tooltip.ts` + test for `instanceId → (tensorName, indices)` mapping.

Three's `Raycaster` supports `instanceId` on `InstancedMesh`. On hover: tooltip shows tensor name, `[i,j]`, true value (from the live activation tensor), and the producing formula string from a per-tensor `formula` field in scene definitions. Unit-test the index mapping for 1D/2D/3D layouts. Manual: hover works in dev scene. Commit.

### Task 16: Labels + dimension brackets (manual acceptance)

**Files:** Create `src/engine/labels.ts`.

troika-three-text labels for tensor names; bracket lines + `d_model=48`-style annotations along grid edges. Billboard toward camera. Manual acceptance in dev scene. Commit.

### Task 17: Walkthrough chapter system + i18n (TDD)

**Files:** Create `src/walkthrough/chapters.ts`, `src/i18n/i18n.ts`, `src/i18n/en.json`, `src/i18n/zh.json`, tests.

Chapter = `{id, cameraKeyframe, highlights: string[] (tensor names), timeline?: TimelineSpec, narrationKey}`. Unit tests: registry rejects duplicate ids and unknown narration keys (validated against BOTH locales — a key missing in either locale is a hard error, keeps EN/ZH in sync); `t(key)` falls back EN→key-name with console.error, never silently empty; locale persists to localStorage; deep link `#mamba/ssm-scan` resolves to chapter index. Commit `feat(walkthrough): chapter registry + bilingual i18n`.

### Task 18: Timeline animation primitives (TDD)

**Files:** Create `src/walkthrough/timeline.ts` + test.

Primitives: `highlightTensor(name)`, `pulseFlow(fromTensor, toTensor)`, `stepToken(t)` (re-runs forward up to token t, updates all TensorViews from recorder — this is what animates the SSM scan). Unit-test the scheduler (sequencing, per-step durations, pause/resume); `stepToken` correctness already guaranteed by golden tests. Commit.

### Task 19: Mamba scene layout

**Files:** Create `src/scenes/mamba.ts`.

Layout per design: token/residual stream as vertical spine (top→bottom: embed → block0 → block1 → final norm → logits); per block, weights flank the spine (in_proj W left; conv kernels, x_proj, dt_proj right; A_log/D as compact grids beside the SSM stage; hidden-state h grid (d_inner×d_state) center-stage at the SSM step). All tensors fed from the live recorder. Smoke test: scene builds from real manifest, contains a TensorView for every golden activation + every weight. Manual: visually inspect, dims labeled. Commit `feat(scenes): mamba 3D layout`.

### Task 20: Mamba guided tour content (EN/ZH)

**Files:** Modify `src/scenes/mamba.ts`, `src/i18n/{en,zh}.json`.

Chapters (≈10): 1 task intro (what selective copying is, watch the model solve it) · 2 embedding · 3 block anatomy · 4 conv as local mixer · 5 **the selection idea: Δ, B, C are functions of input** (contrast LTI) · 6 **the scan, token-by-token** — `stepToken` animation, watch `layer0.ssm.h.t*` evolve, Δ gate visibly small on NOISE tokens / large on data tokens · 7 gating + out_proj · 8 layer 2 refines · 9 logits + answer readout · 10 recap + "compare with GPT" link. Both locales authored together (registry test enforces parity).
**Verify:** full tour plays start-to-finish in browser without console errors; narration matches what's on screen at each chapter (manual checklist per chapter).
Commit `feat(content): mamba bilingual guided tour`.

### Task 21: GPT baseline scene + mini-tour

**Files:** Create `src/scenes/gpt.ts` + narration keys.

Compact treatment (≈5 chapters): embedding+positions, attention (QKV, score matrix on the same selective-copy input — attention weights visibly attend from answer positions back to data tokens; the direct visual contrast with Mamba's fixed-size h), MLP, logits. Smoke test like Task 19. Commit.

### Task 22: Landing page + routing + error states

**Files:** Create `src/pages/landing.ts`, `src/pages/archPage.ts`; modify `src/main.ts`, `index.html`.

Hash routing (`#/`, `#/mamba`, `#/gpt`). Landing: cards (Mamba, GPT live; RWKV/MoE/KAN/RetNet "coming soon" — not clickable, no dead pages). Arch page chrome: chapter sidebar, EN/ZH toggle, token-step controls, play/pause. Error states: model fetch failure → visible retry panel (never synthetic data); no WebGL2 → message. Unit-test the router. Manual acceptance of full flows. Commit.

### Task 23: Perf check

**Verify:** Chrome devtools FPS meter during full Mamba tour ≥ 55fps on this machine; instance count logged per scene. If below: merge static weight tensors into fewer InstancedMeshes / reduce cell geometry to planes. Record numbers in `docs/plans/perf-notes.md`. Commit.

### Task 24: Deploy to GitHub Pages

**Files:** Create `.github/workflows/deploy.yml` (checkout → npm ci → vitest run → vite build → upload-pages-artifact → deploy-pages; tests failing blocks deploy).

Create GitHub repo `llm-arch-viz`, push, enable Pages. **Verify:** site loads at `https://<user>.github.io/llm-arch-viz/`, Mamba tour works in production build, model fetches succeed under the `base` path. Commit.

---

## Task → success-criterion map

| Criterion (from design doc) | Gate |
|---|---|
| TS activations match PyTorch ≤1e-4 | Tasks 10, 11 (vitest, blocks scenes) |
| Mamba page with full bilingual tour live | Tasks 20, 22, 24 |
| GPT baseline scene live | Tasks 21, 22, 24 |
| 60fps tour on M-series | Task 23 |
