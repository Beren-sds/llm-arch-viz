/**
 * Mamba 3D scene: the flagship visual. The token/residual stream runs as
 * a vertical spine (top → bottom, y decreasing), one TensorView per
 * recorded activation; weights flank the spine left/right at the stage
 * that consumes them; the recurrent h-state (d_inner × d_state) is the
 * hero grid offset right of the spine at the scan stage.
 *
 *           [title]
 *         embed.out                       ← spine, centered on x = 0
 *   ┌─ block0 ─────────────────────────┐
 *   │  norm.out                        │
 *   │  in_proj.out      (in_proj.W ←)  │
 *   │  conv.out         (→ conv.W/b)   │
 *   │  x_proj.out       (→ x_proj.W)   │
 *   │  delta.out        (→ dt_proj.W/b)│
 *   │  ssm.out   (A_log/D ←) (→ h HERO)│
 *   │  gate.out                        │
 *   │  out_proj.out                    │
 *   └──────────────────────────────────┘
 *   [block1 — same anatomy]
 *         final_norm.out
 *         head.logits
 *
 * All 21 per-token h snapshots (`layer{i}.ssm.h.t{t}`) are real views
 * stacked at the SAME hero position; only one is visible (and pickable)
 * at a time — see showHState. Task 20's stepToken animates the scan by
 * re-running prefixes through binding.runForward.
 */

import * as THREE from "three";
import { T } from "../compute/tensor";
import { getTensor, type Manifest } from "../compute/loader";
import { mambaDimsFrom, mambaForward } from "../compute/mamba";
import { MapRecorder } from "../compute/recorder";
import { TensorView } from "../engine/tensorView";
import { createFlowSegment, disposeFlow } from "../engine/flow";
import {
  createDimBracket,
  createTensorLabel,
  disposeLabel,
  makeBillboard,
  type DimBracketOptions,
  type LabelOptions,
} from "../engine/labels";
import type { Picker } from "../engine/picking";
import type { CameraKeyframe } from "../engine/cameraTour";
import type { SceneBinding, TimelineSpec, TimelineStep } from "../walkthrough/timeline";
import { ChapterRegistry, type Chapter } from "../walkthrough/chapters";
import type { I18n } from "../i18n/i18n";
import type { SceneController } from "./sceneController";
import { CELL, GAP, PITCH, frameRect, gridSize, pad, union, type Rect } from "./layout";

/** Sequence length the layout is built for (selective-copying training T). */
export const MAMBA_SEQ_LEN = 21;

// ----- layout constants (world units) --------------------------------------

/** Vertical gap between consecutive spine tensors (room for a name label). */
const SPINE_GAP = 8;
/** Extra vertical separation between regions (embed / block0 / block1 / head). */
const REGION_GAP = 12;
/** Min horizontal margin between the spine's widest overlap and a flank. */
const FLANK_MARGIN = 8;
/** Horizontal gap between flank lanes. */
const LANE_GAP = 10;
/** Vertical gap between stacked flank items. */
const FLANK_GAP = 8;
/** Label sits this far above its grid's top edge (anchorY: top). */
const LABEL_RISE = 4.2;

/** Anchor names Task 20 chapters fly to (all present in `anchors`). */
export const MAMBA_ANCHOR_NAMES = [
  "home",
  "embed",
  "block0",
  "conv0",
  "selection0",
  "scan0",
  "gate0",
  "block1",
  "head",
] as const;

// ----- public surface -------------------------------------------------------

/**
 * Label/bracket factories. Defaults to the real troika-backed ones; the
 * node smoke test injects stubs (troika's sync() needs a browser-like
 * global and a font fetch, neither of which exists under vitest's node
 * environment).
 */
export interface LabelFactory {
  label(text: string, opts?: LabelOptions): THREE.Object3D;
  bracket(opts: DimBracketOptions): THREE.Object3D;
}

export interface MambaSceneDeps {
  scene: THREE.Scene;
  weights: Map<string, T>;
  manifest: Manifest;
  /**
   * When given, every view is registered as a pick target (with its
   * formula) and applyActivations calls picker.requestRepick().
   */
  picker?: Picker;
  i18n?: I18n;
  labelFactory?: LabelFactory;
}

export interface MambaScene {
  /** Tensor name (activation or weight) → its view. */
  views: Map<string, TensorView>;
  /** Timeline channel. Owns highlight/dim exclusively while a timeline plays. */
  binding: SceneBinding;
  /** Frames the whole spine, straight on. Same object as anchors.get('home'). */
  cameraHome: CameraKeyframe;
  /** Named viewpoints per region (keys = MAMBA_ANCHOR_NAMES) for Task 20. */
  anchors: Map<string, CameraKeyframe>;
  /** Run the full forward on `tokens` (1..MAMBA_SEQ_LEN) and apply all activations. */
  setTokens(tokens: number[]): void;
  /** Show ONLY the h snapshot at token t for the given layer (others hidden). */
  showHState(layer: number, t: number): void;
  /** Per-frame: face all billboarded labels toward the camera. */
  update(camera: THREE.Camera): void;
  /** Every label/bracket object (await labelsReady(...) on these before screenshots). */
  labelObjects: THREE.Object3D[];
  dispose(): void;
}

// ----- formulas (tooltip strings) -------------------------------------------

/** Producing expression for an activation; weights get none (plain names). */
function formulaFor(name: string): string | undefined {
  if (name === "embed.out") return "x = E[token]";
  if (name === "final_norm.out") return "x̂ = RMSNorm(x) ⊙ g";
  if (name === "head.logits") return "logits = W_lm·x";
  const suffix = name.replace(/^layer\d+\./, "");
  if (/^ssm\.h\.t\d+$/.test(suffix)) return "h = exp(Δ⊙A)·h + Δ⊙B⊗u";
  switch (suffix) {
    case "norm.out":
      return "x̂ = RMSNorm(x) ⊙ g";
    case "in_proj.out":
      return "[x | z] = W_in·x̂";
    case "conv.out":
      return "u = SiLU(causal_conv1d(x))";
    case "x_proj.out":
      return "[dt | B | C] = W_x·u";
    case "delta.out":
      return "Δ = softplus(W_Δ·dt + b)";
    case "ssm.out":
      return "y = C·h + D⊙u";
    case "gate.out":
      return "out = y ⊙ SiLU(z)";
    case "out_proj.out":
      return "x ← x + W_out·out";
    default:
      return undefined;
  }
}

// ----- the scene -------------------------------------------------------------

export function buildMambaScene(deps: MambaSceneDeps): MambaScene {
  const { weights, manifest, picker } = deps;
  const dims = mambaDimsFrom(manifest);
  const dInner = dims.expand * dims.d_model;
  const SEQ = MAMBA_SEQ_LEN;

  const factory: LabelFactory = deps.labelFactory ?? {
    label: createTensorLabel,
    bracket: createDimBracket,
  };

  const root = new THREE.Group();
  root.name = "mamba-scene";
  deps.scene.add(root);

  const views = new Map<string, TensorView>();
  const rects = new Map<string, Rect>();
  const labelObjects: THREE.Object3D[] = [];
  const billboards: Array<(camera: THREE.Camera) => void> = [];

  /** Current input; empty until setTokens. */
  let tokens: number[] = [];
  let disposed = false;

  // -- placement helpers ------------------------------------------------------

  function addLabel(text: string, x: number, y: number, opts?: LabelOptions): void {
    const obj = factory.label(text, opts);
    obj.position.set(x, y, 0);
    root.add(obj);
    labelObjects.push(obj);
    billboards.push(makeBillboard(obj));
  }

  function addBracket(opts: DimBracketOptions): void {
    // Brackets stay fixed in the grid plane — never billboarded.
    const obj = factory.bracket(opts);
    root.add(obj);
    labelObjects.push(obj);
  }

  interface PlaceOpts {
    /** Exactly one horizontal reference: grid center / left edge / right edge. */
    centerX?: number;
    leftX?: number;
    rightX?: number;
    topY: number;
    /** Label text; null = no label (h-state stack shares one). */
    label?: string | null;
    /** Initial values (weights set once here; activations via setTokens). */
    values?: T;
    /** Register with the picker (h-states manage their own registration). */
    pickable?: boolean;
  }

  function place(name: string, shape: readonly number[], opts: PlaceOpts): Rect {
    const { w, h } = gridSize(shape);
    const left = opts.leftX ?? (opts.rightX !== undefined ? opts.rightX - w : opts.centerX! - w / 2);
    const top = opts.topY;
    const view = new TensorView(name, shape, {
      cellSize: CELL,
      gap: GAP,
      // origin = center of cell (0,0); grid edges extend CELL/2 past it.
      origin: [left + CELL / 2, top - CELL / 2, 0],
    });
    if (opts.values) view.setValues(opts.values);
    root.add(view.mesh);
    views.set(name, view);
    const rect: Rect = { left, right: left + w, top, bottom: top - h };
    rects.set(name, rect);
    if (opts.label !== null) {
      addLabel(opts.label ?? name, (rect.left + rect.right) / 2, rect.top + LABEL_RISE);
    }
    if (picker && opts.pickable !== false) {
      const formula = formulaFor(name);
      picker.add(formula === undefined ? { view } : { view, formula });
    }
    return rect;
  }

  /** Weight tensor by name, as the 2D display shape (conv kernels squeezed). */
  function weightFor(name: string): T {
    const t = getTensor(weights, name);
    if (name.endsWith("conv1d.weight")) {
      // PyTorch (d_inner, 1, d_conv) → display (d_inner, d_conv).
      return T.from(t.data, [t.shape[0], t.shape[2]]);
    }
    return t;
  }

  /** place() for a weight: values set once, display label = name sans block prefix. */
  function placeWeight(name: string, opts: Omit<PlaceOpts, "values" | "label">): Rect {
    const t = weightFor(name);
    return place(name, t.shape, {
      ...opts,
      values: t,
      label: name.replace(/^blocks\.\d+\./, ""),
    });
  }

  // -- the spine + flanks -------------------------------------------------------

  const dM = dims.d_model; // 48
  const spineW = (cols: number): number => cols * PITCH - GAP;

  let y = 0;

  // Title
  const titleText = deps.i18n?.t("scene.mamba.title") ?? "Mamba — selective copying";
  addLabel(titleText, 0, y + 30, { size: 12, color: "#d7e3ff" });

  // embed.out + its weight + dim brackets
  const embedRect = place("embed.out", [SEQ, dM], { centerX: 0, topY: y });
  placeWeight("embedding.weight", {
    rightX: -spineW(dM) / 2 - FLANK_MARGIN,
    topY: y,
  });
  addBracket({
    from: [embedRect.left, embedRect.bottom, 0],
    to: [embedRect.right, embedRect.bottom, 0],
    offset: -2,
    label: `d_model = ${dM}`,
  });
  addBracket({
    from: [embedRect.left, embedRect.top, 0],
    to: [embedRect.left, embedRect.bottom, 0],
    offset: -2, // left of the grid, ticks pointing left
    label: `T = ${SEQ}`,
  });
  y = embedRect.bottom - REGION_GAP - SPINE_GAP;

  const regionRects: Rect[] = [];

  for (let i = 0; i < dims.n_layer; i++) {
    const L = (s: string): string => `layer${i}.${s}`;
    const B = (s: string): string => `blocks.${i}.${s}`;
    const blockParts: Rect[] = [];
    const spine = (name: string, cols: number): Rect => {
      const r = place(name, [SEQ, cols], { centerX: 0, topY: y, label: name });
      blockParts.push(r);
      y = r.bottom - SPINE_GAP;
      return r;
    };

    // Spine stages
    const normRect = spine(L("norm.out"), dM);
    blockParts.push(
      placeWeight(`norms.${i}.weight`, {
        rightX: normRect.left - FLANK_MARGIN,
        topY: normRect.top,
      }),
    );
    const inProjRect = spine(L("in_proj.out"), 2 * dInner);
    const convRect = spine(L("conv.out"), dInner);
    spine(L("x_proj.out"), dims.dt_rank + 2 * dims.d_state);
    const deltaRect = spine(L("delta.out"), dInner);
    const ssmRect = spine(L("ssm.out"), dInner);
    spine(L("gate.out"), dInner);
    spine(L("out_proj.out"), dM);

    // Right flank, lane 1: the h-state HERO — all SEQ snapshots stacked at
    // the same spot, centered on the scan band (delta → ssm.out). Only the
    // latest is visible/pickable; see showHState.
    const heroLeft = ssmRect.right + FLANK_MARGIN;
    const heroH = gridSize([dInner, dims.d_state]).h;
    const heroTop = (deltaRect.top + ssmRect.bottom) / 2 + heroH / 2;
    let heroRect: Rect = { left: 0, right: 0, top: 0, bottom: 0 };
    for (let t = 0; t < SEQ; t++) {
      heroRect = place(L(`ssm.h.t${t}`), [dInner, dims.d_state], {
        leftX: heroLeft,
        topY: heroTop,
        label: null,
        pickable: false,
      });
    }
    addLabel(`ssm.h  (layer ${i})`, (heroRect.left + heroRect.right) / 2, heroRect.top + LABEL_RISE);
    addBracket({
      from: [heroRect.left, heroRect.bottom, 0],
      to: [heroRect.right, heroRect.bottom, 0],
      offset: -2,
      label: `d_state = ${dims.d_state}`,
    });
    addBracket({
      from: [heroRect.right, heroRect.top, 0],
      to: [heroRect.right, heroRect.bottom, 0],
      offset: 2,
      label: `d_inner = ${dInner}`,
    });
    blockParts.push(heroRect);

    // Inner-left lane: A_log beside the spine at scan level (it parameterizes
    // the scan's decay), with the D skip row tucked underneath. Both clear
    // the wide in_proj.out band vertically.
    const aRight = ssmRect.left - FLANK_MARGIN;
    const aRect = placeWeight(B("A_log"), { rightX: aRight, topY: heroRect.top });
    const dRect = placeWeight(B("D"), { rightX: aRight, topY: aRect.bottom - FLANK_GAP });
    blockParts.push(aRect, dRect);

    // Outer-left lane: the big projection weights, clear of BOTH the wide
    // in_proj.out row and the A_log/D group.
    const lane1Right = Math.min(-(inProjRect.right + FLANK_MARGIN), dRect.left - LANE_GAP);
    const inWRect = placeWeight(B("in_proj.weight"), { rightX: lane1Right, topY: normRect.top });
    const outWRect = placeWeight(B("out_proj.weight"), {
      rightX: lane1Right,
      topY: inWRect.bottom - FLANK_GAP,
    });
    blockParts.push(inWRect, outWRect);

    // Right flank, lane 2: conv → x_proj → dt_proj weight stack.
    const lane2Left = heroRect.right + LANE_GAP + 6; // +6: room for the d_inner bracket label
    let fy = convRect.top;
    const convWRect = placeWeight(B("conv1d.weight"), { leftX: lane2Left, topY: fy });
    fy = convWRect.bottom - FLANK_GAP;
    const convBRect = placeWeight(B("conv1d.bias"), { leftX: lane2Left, topY: fy });
    fy = convBRect.bottom - FLANK_GAP;
    const xWRect = placeWeight(B("x_proj.weight"), { leftX: lane2Left, topY: fy });
    fy = xWRect.bottom - FLANK_GAP;
    const dtWRect = placeWeight(B("dt_proj.weight"), { leftX: lane2Left, topY: fy });
    fy = dtWRect.bottom - FLANK_GAP;
    const dtBRect = placeWeight(B("dt_proj.bias"), { leftX: lane2Left, topY: fy });
    blockParts.push(convWRect, convBRect, xWRect, dtWRect, dtBRect);

    const regionRect = union(...blockParts);
    regionRects.push(regionRect);
    // Continue below the spine AND the outer-left weight stack (it runs
    // deeper than the spine; the next block's weights start at its top).
    // The right stack runs deeper still but sits far enough off-axis
    // that only ±120-wide in_proj.out could reach it — verified clear.
    y = Math.min(y, outWRect.bottom - SPINE_GAP);
    y -= REGION_GAP;
  }

  // final_norm.out + head.logits
  const finRect = place("final_norm.out", [SEQ, dM], { centerX: 0, topY: y });
  placeWeight("final_norm.weight", {
    rightX: finRect.left - FLANK_MARGIN,
    topY: finRect.top,
  });
  y = finRect.bottom - SPINE_GAP;
  const logitsRect = place("head.logits", [SEQ, dims.vocab_size], { centerX: 0, topY: y });
  placeWeight("lm_head.weight", {
    rightX: logitsRect.left - FLANK_MARGIN,
    topY: logitsRect.top,
  });

  // -- residual-stream spine -----------------------------------------------------
  // Glowing connectors threading the centred .out tensors top→bottom, sitting
  // purely in the vertical gutters so they never occlude a cell. Reads as the
  // residual stream flowing down the scene.
  const flowMeshes: THREE.Mesh[] = [];
  const spineCol = [...rects.values()]
    .filter((rc) => Math.abs((rc.left + rc.right) / 2) < CELL)
    .sort((p, q) => q.top - p.top);
  for (let i = 1; i < spineCol.length; i++) {
    const seg = createFlowSegment([0, spineCol[i - 1].bottom, 0], [0, spineCol[i].top, 0]);
    root.add(seg);
    flowMeshes.push(seg);
  }

  // -- anchors -------------------------------------------------------------------

  const r = (name: string): Rect => {
    const rect = rects.get(name);
    if (!rect) throw new Error(`mamba scene: no rect for "${name}"`);
    return rect;
  };

  const allRect = union(...rects.values());
  const cameraHome = frameRect(pad(allRect, 8), false);
  const anchors = new Map<string, CameraKeyframe>();
  anchors.set("home", cameraHome);
  anchors.set("embed", frameRect(pad(union(r("embed.out"), r("embedding.weight")), 7), true));
  anchors.set("block0", frameRect(pad(regionRects[0], 8), true));
  anchors.set("block1", frameRect(pad(regionRects[1], 8), true));
  anchors.set(
    "conv0",
    frameRect(
      pad(union(r("layer0.conv.out"), r("blocks.0.conv1d.weight"), r("blocks.0.conv1d.bias")), 6),
      true,
    ),
  );
  anchors.set(
    "selection0",
    frameRect(
      pad(
        union(
          r("layer0.x_proj.out"),
          r("layer0.delta.out"),
          r("blocks.0.x_proj.weight"),
          r("blocks.0.dt_proj.weight"),
        ),
        6,
      ),
      true,
    ),
  );
  anchors.set(
    "scan0",
    frameRect(
      pad(union(r("layer0.ssm.h.t0"), r("layer0.ssm.out"), r("blocks.0.A_log"), r("blocks.0.D")), 6),
      true,
    ),
  );
  anchors.set(
    "gate0",
    frameRect(pad(union(r("layer0.ssm.out"), r("layer0.gate.out"), r("layer0.out_proj.out")), 6), true),
  );
  anchors.set(
    "head",
    frameRect(
      pad(union(r("final_norm.out"), r("head.logits"), r("lm_head.weight"), r("final_norm.weight")), 7),
      true,
    ),
  );

  // -- h-state visibility ----------------------------------------------------------

  /** Visible h snapshot per layer (mirrors mesh.visible; -1 before first show). */
  const visibleH: number[] = new Array(dims.n_layer).fill(-1);

  function showHState(layer: number, t: number): void {
    if (layer < 0 || layer >= dims.n_layer || !Number.isInteger(t) || t < 0 || t >= SEQ) {
      throw new Error(`showHState: layer ${layer} / t ${t} out of range`);
    }
    if (visibleH[layer] === t) return;
    for (let tt = 0; tt < SEQ; tt++) {
      const view = views.get(`layer${layer}.ssm.h.t${tt}`)!;
      const show = tt === t;
      view.mesh.visible = show;
      // Keep exactly one snapshot pickable: the raycaster ignores
      // mesh.visible, and all SEQ snapshots are coplanar.
      if (picker) {
        if (show) picker.add({ view, formula: formulaFor(view.name)! });
        else picker.remove(view);
      }
    }
    visibleH[layer] = t;
  }

  for (let i = 0; i < dims.n_layer; i++) showHState(i, SEQ - 1);

  // -- compute plumbing ---------------------------------------------------------------

  /**
   * Fit an activation into a view: exact shape passes through; a shorter
   * prefix (fewer rows, same cols) is zero-padded — un-computed future
   * token rows render as the colormap's neutral zero. Anything else throws.
   */
  function fitToView(view: TensorView, t: T): T {
    const vs = view.shape;
    if (t.shape.length === vs.length && t.shape.every((d, k) => d === vs[k])) return t;
    if (t.shape.length === 2 && vs.length === 2 && t.shape[1] === vs[1] && t.shape[0] <= vs[0]) {
      const padded = T.zeros(vs);
      padded.data.set(t.data);
      return padded;
    }
    throw new Error(
      `mamba scene: activation "${view.name}" shape [${t.shape.join(", ")}] ` +
        `does not fit view [${vs.join(", ")}]`,
    );
  }

  function applyActivations(acts: Map<string, T>): void {
    const latest: number[] = new Array(dims.n_layer).fill(-1);
    for (const [name, t] of acts) {
      const view = views.get(name);
      if (!view) throw new Error(`mamba scene: no view for activation "${name}"`);
      view.setValues(fitToView(view, t));
      const m = /^layer(\d+)\.ssm\.h\.t(\d+)$/.exec(name);
      if (m) {
        const layer = Number(m[1]);
        latest[layer] = Math.max(latest[layer], Number(m[2]));
      }
    }
    for (let i = 0; i < dims.n_layer; i++) {
      if (latest[i] >= 0) showHState(i, latest[i]);
    }
    picker?.requestRepick();
  }

  function runTokens(toks: number[]): Map<string, T> {
    const rec = new MapRecorder();
    mambaForward(weights, dims, toks, rec);
    return rec.activations;
  }

  const binding: SceneBinding = {
    // INCLUSIVE: tokens[0..uptoToken] — uptoToken = 0 runs a 1-token prefix.
    runForward(uptoToken: number): Map<string, T> {
      if (tokens.length === 0) {
        throw new Error("mamba scene: runForward before setTokens");
      }
      if (!Number.isInteger(uptoToken) || uptoToken < 0 || uptoToken >= tokens.length) {
        throw new Error(
          `mamba scene: uptoToken ${uptoToken} out of range [0, ${tokens.length})`,
        );
      }
      return runTokens(tokens.slice(0, uptoToken + 1));
    },
    applyActivations,
    // Idempotent by construction: TensorView highlight/dim are uniform
    // writes (set, not toggled), so repeated on/off calls are safe.
    setHighlight(names: string[], on: boolean): void {
      for (const name of names) {
        const view = views.get(name);
        if (!view) throw new Error(`mamba scene: setHighlight on unknown "${name}"`);
        view.setHighlight(on);
      }
    },
    setDim(namesNotIn: string[] | null): void {
      if (namesNotIn === null) {
        for (const view of views.values()) view.setDim(false);
        return;
      }
      const keep = new Set(namesNotIn);
      for (const [name, view] of views) view.setDim(!keep.has(name));
    },
    pulse(_from: string, _to: string): void {
      // TODO(Task 20): decide whether a visual flow pulse is needed for the
      // tour; intentionally a no-op until then (SceneBinding allows it).
    },
  };

  function setTokens(toks: number[]): void {
    if (toks.length < 1 || toks.length > SEQ) {
      throw new Error(`mamba scene: setTokens needs 1..${SEQ} tokens, got ${toks.length}`);
    }
    for (const t of toks) {
      if (!Number.isInteger(t) || t < 0 || t >= dims.vocab_size) {
        throw new Error(`mamba scene: token ${t} outside vocab [0, ${dims.vocab_size})`);
      }
    }
    tokens = toks.slice();
    applyActivations(runTokens(tokens));
  }

  function update(camera: THREE.Camera): void {
    for (const face of billboards) face(camera);
  }

  function dispose(): void {
    if (disposed) return;
    disposed = true;
    for (const view of views.values()) {
      picker?.remove(view);
      view.dispose();
    }
    for (const obj of labelObjects) disposeLabel(obj);
    for (const m of flowMeshes) disposeFlow(m);
    deps.scene.remove(root);
  }

  return {
    views,
    binding,
    cameraHome,
    anchors,
    setTokens,
    showHState,
    update,
    labelObjects,
    dispose,
  };
}

// ----- guided tour -----------------------------------------------------------

/**
 * The ~10-chapter Mamba tour over a built scene. Each chapter flies to a
 * named anchor and focuses a set of tensors; the scan chapter additionally
 * carries a token-by-token timeline. That timeline uses ONLY stepToken
 * steps (no highlight/focus), so it never touches highlight/dim — the page
 * driver can keep the chapter's static focus applied while it plays without
 * violating timeline.ts's exclusive-ownership contract.
 *
 * Narration bodies are `mamba.ch.<id>.body`; the page derives the sidebar
 * title as the same key with `.body` → `.title`. The registry validates
 * every narrationKey against BOTH locales at construction (a key missing in
 * either locale is a hard error), so an authored-in-one-locale slip fails
 * the tests, not the viewer.
 */
export function buildMambaChapters(scene: SceneController, i18n: I18n): ChapterRegistry {
  const at = (anchor: string): CameraKeyframe => {
    const kf = scene.anchors.get(anchor);
    if (!kf) throw new Error(`mamba chapters: scene has no anchor "${anchor}"`);
    return kf;
  };
  const spine = (layer: number): string[] =>
    ["norm", "in_proj", "conv", "x_proj", "delta", "ssm", "gate", "out_proj"].map(
      (m) => `layer${layer}.${m}.out`,
    );
  // Every layer-0 h snapshot, so the scan chapter's focus never dims
  // whichever one showHState has made visible (they are coplanar; one shows).
  const hSnaps0 = Array.from({ length: MAMBA_SEQ_LEN }, (_, t) => `layer0.ssm.h.t${t}`);
  const scanSteps: TimelineStep[] = Array.from({ length: MAMBA_SEQ_LEN }, (_, t) => ({
    kind: "stepToken",
    token: t,
    durationMs: 450,
  }));
  const scanTimeline: TimelineSpec = { steps: scanSteps, loop: true };

  const chapters: Chapter[] = [
    { id: "intro", camera: at("home"), highlights: [], narrationKey: "mamba.ch.intro.body" },
    {
      id: "embed",
      camera: at("embed"),
      highlights: ["embed.out", "embedding.weight"],
      narrationKey: "mamba.ch.embed.body",
    },
    { id: "block", camera: at("block0"), highlights: spine(0), narrationKey: "mamba.ch.block.body" },
    {
      id: "conv",
      camera: at("conv0"),
      highlights: ["layer0.conv.out", "blocks.0.conv1d.weight", "blocks.0.conv1d.bias"],
      narrationKey: "mamba.ch.conv.body",
    },
    {
      id: "selection",
      camera: at("selection0"),
      highlights: [
        "layer0.x_proj.out",
        "layer0.delta.out",
        "blocks.0.x_proj.weight",
        "blocks.0.dt_proj.weight",
      ],
      narrationKey: "mamba.ch.selection.body",
    },
    {
      id: "scan",
      camera: at("scan0"),
      highlights: ["layer0.delta.out", "layer0.ssm.out", "blocks.0.A_log", "blocks.0.D", ...hSnaps0],
      narrationKey: "mamba.ch.scan.body",
      timeline: scanTimeline,
    },
    {
      id: "gate",
      camera: at("gate0"),
      highlights: ["layer0.ssm.out", "layer0.gate.out", "layer0.out_proj.out"],
      narrationKey: "mamba.ch.gate.body",
    },
    {
      id: "layer2",
      camera: at("block1"),
      highlights: spine(1),
      narrationKey: "mamba.ch.layer2.body",
    },
    {
      id: "readout",
      camera: at("head"),
      highlights: ["final_norm.out", "head.logits", "lm_head.weight", "final_norm.weight"],
      narrationKey: "mamba.ch.readout.body",
    },
    { id: "recap", camera: at("home"), highlights: [], narrationKey: "mamba.ch.recap.body" },
  ];
  return new ChapterRegistry("mamba", chapters, i18n);
}
