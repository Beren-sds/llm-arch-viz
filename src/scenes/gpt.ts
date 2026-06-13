/**
 * GPT baseline 3D scene — the same vertical-spine visual grammar as the
 * Mamba scene (engine/sceneController), built on the same input so the two
 * pages are honestly comparable. The residual stream runs top → bottom
 * (embed → per-layer ln1/attn/ln2/mlp → final norm → logits); the per-head
 * attention matrices (scores, then softmax weights — both n_head × T × T)
 * are the hero, the GPT counterpart of Mamba's fixed-size h-state: where
 * Mamba carries one state forward, GPT recomputes a full T×T attention map.
 *
 * The big projection weights (qkv 144×48, fc 192×48) and the wide MLP
 * activations (T×192) make this scene tall; the attention and MLP flank
 * clusters sit to the right of the spine, with y advancing past each so
 * they never overlap. Weights flank the stage that consumes them.
 *
 * All values are live from compute/gpt.ts (golden-gated <1e-4). The GPT
 * chapters carry no per-token timeline — attention is computed in one shot,
 * not scanned — so the page only ever applies the full-sequence activations.
 */

import * as THREE from "three";
import { T } from "../compute/tensor";
import { getTensor, type Manifest } from "../compute/loader";
import { gptDimsFrom, gptForward } from "../compute/gpt";
import { MapRecorder } from "../compute/recorder";
import { TensorView } from "../engine/tensorView";
import { createEdge, createFlowSegment, disposeFlow } from "../engine/flow";
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
import type { SceneBinding } from "../walkthrough/timeline";
import { ChapterRegistry, type Chapter } from "../walkthrough/chapters";
import type { I18n } from "../i18n/i18n";
import type { SceneController } from "./sceneController";
import { CELL, GAP, frameRect, gridSize, pad, union, type Rect } from "./layout";

/** Sequence length the layout is built for (selective-copying training T). */
export const GPT_SEQ_LEN = 21;

// ----- layout spacing (world units) ----------------------------------------

const SPINE_GAP = 8;
const REGION_GAP = 20;
const FLANK_MARGIN = 8;
const LANE_GAP = 10;
const FLANK_GAP = 7;
const LABEL_RISE = 4.2;

/** Anchor names the GPT chapters fly to (all present in `anchors`). */
export const GPT_ANCHOR_NAMES = [
  "home",
  "embed",
  "block0",
  "qkv0",
  "scores0",
  "weights0",
  "attnout0",
  "mlp0",
  "block1",
  "head",
] as const;

// ----- public surface -------------------------------------------------------

export interface LabelFactory {
  label(text: string, opts?: LabelOptions): THREE.Object3D;
  bracket(opts: DimBracketOptions): THREE.Object3D;
}

export interface GptSceneDeps {
  scene: THREE.Scene;
  weights: Map<string, T>;
  manifest: Manifest;
  picker?: Picker;
  i18n?: I18n;
  labelFactory?: LabelFactory;
}

// ----- formulas (tooltip strings) -------------------------------------------

/** Producing expression for an activation; weights get none. */
function formulaFor(name: string): string | undefined {
  if (name === "embed.out") return "x = TokEmb[t] + PosEmb[t]";
  if (name === "final_norm.out") return "x̂ = LayerNorm(x)";
  if (name === "head.logits") return "logits = W_lm·x̂";
  const s = name.replace(/^layer\d+\./, "");
  switch (s) {
    case "ln1.out":
    case "ln2.out":
      return "x̂ = LayerNorm(x)";
    case "attn.q":
      return "Q = x̂·W_qkv (q part), per head";
    case "attn.k":
      return "K = x̂·W_qkv (k part), per head";
    case "attn.v":
      return "V = x̂·W_qkv (v part), per head";
    case "attn.scores":
      return "S = Q·Kᵀ / √d_head  (causal mask → −∞)";
    case "attn.weights":
      return "A = softmax(S)  (row-stochastic)";
    case "attn.out":
      return "x ← x + (A·V)·W_o";
    case "mlp.fc":
      return "h = x̂·W_fc + b";
    case "mlp.act":
      return "GELU(h)";
    case "mlp.proj":
      return "x ← x + GELU(h)·W_proj + b";
    default:
      return undefined;
  }
}

/** Compact tooltip/label name for a weight tensor. */
function weightLabel(name: string): string {
  if (name === "tok_embedding.weight") return "tok_embed";
  if (name === "pos_embedding.weight") return "pos_embed";
  if (name === "lm_head.weight") return "lm_head";
  if (name === "ln_f.weight") return "ln_f.g";
  if (name === "ln_f.bias") return "ln_f.b";
  const m = /^(attns|mlps|ln1s|ln2s)\.\d+\.(.+)$/.exec(name);
  if (!m) return name;
  const tail = m[2]
    .replace(/\.weight$/, m[1] === "ln1s" || m[1] === "ln2s" ? ".g" : ".W")
    .replace(/\.bias$/, ".b");
  if (m[1] === "ln1s") return `ln1.${tail}`;
  if (m[1] === "ln2s") return `ln2.${tail}`;
  return tail; // attns / mlps: qkv_proj.W, out_proj.b, fc.W, proj.b, …
}

// ----- the scene -------------------------------------------------------------

export function buildGptScene(deps: GptSceneDeps): SceneController {
  const { weights, manifest, picker } = deps;
  const dims = gptDimsFrom(manifest);
  const SEQ = GPT_SEQ_LEN;
  const dM = dims.d_model;
  const nHead = dims.n_head;
  const headDim = dims.head_dim;
  const dMlp = dims.mlp_ratio * dM;

  const factory: LabelFactory = deps.labelFactory ?? {
    label: createTensorLabel,
    bracket: createDimBracket,
  };

  const root = new THREE.Group();
  root.name = "gpt-scene";
  deps.scene.add(root);

  const views = new Map<string, TensorView>();
  const rects = new Map<string, Rect>();
  const labelObjects: THREE.Object3D[] = [];
  const billboards: Array<(camera: THREE.Camera) => void> = [];

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
    const obj = factory.bracket(opts);
    root.add(obj);
    labelObjects.push(obj);
  }

  interface PlaceOpts {
    centerX?: number;
    leftX?: number;
    rightX?: number;
    topY: number;
    label?: string | null;
    values?: T;
    pickable?: boolean;
  }

  function place(name: string, shape: readonly number[], opts: PlaceOpts): Rect {
    const { w, h } = gridSize(shape);
    const left = opts.leftX ?? (opts.rightX !== undefined ? opts.rightX - w : opts.centerX! - w / 2);
    const top = opts.topY;
    const view = new TensorView(name, shape, {
      cellSize: CELL,
      gap: GAP,
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

  /** Place a weight by name (display label = compact weightLabel). */
  function placeW(name: string, opts: Omit<PlaceOpts, "values" | "label">): Rect {
    const t = getTensor(weights, name);
    return place(name, t.shape, { ...opts, values: t, label: weightLabel(name) });
  }

  /** Place a weight and, if present, its bias one row below (same left edge). */
  function placeWB(weightName: string, biasName: string, opts: Omit<PlaceOpts, "values" | "label">): Rect {
    const wRect = placeW(weightName, opts);
    const bRect = placeW(biasName, { leftX: wRect.left, topY: wRect.bottom - FLANK_GAP });
    return union(wRect, bRect);
  }

  const spineHalf = gridSize([SEQ, dM]).w / 2;

  let y = 0;

  // Title
  const titleText = deps.i18n?.t("scene.gpt.title") ?? "GPT — selective copying";
  addLabel(titleText, 0, y + 30, { size: 12, color: "#d7e3ff" });

  // embed.out + token/position embeddings (left) + dim brackets
  const embedRect = place("embed.out", [SEQ, dM], { centerX: 0, topY: y });
  const tokRect = placeW("tok_embedding.weight", {
    rightX: -spineHalf - FLANK_MARGIN,
    topY: y,
  });
  placeW("pos_embedding.weight", {
    rightX: -spineHalf - FLANK_MARGIN,
    topY: tokRect.bottom - FLANK_GAP,
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
    offset: -2,
    label: `T = ${SEQ}`,
  });
  y = embedRect.bottom - REGION_GAP - SPINE_GAP;

  for (let i = 0; i < dims.n_layer; i++) {
    const L = (s: string): string => `layer${i}.${s}`;
    const spine = (name: string, cols: number): Rect => {
      const rect = place(name, [SEQ, cols], { centerX: 0, topY: y, label: name });
      y = rect.bottom - SPINE_GAP;
      return rect;
    };

    // -- attention band -------------------------------------------------------
    const ln1Rect = spine(L("ln1.out"), dM);
    placeWB(`ln1s.${i}.weight`, `ln1s.${i}.bias`, {
      rightX: ln1Rect.left - FLANK_MARGIN,
      topY: ln1Rect.top,
    });
    const attnOutRect = spine(L("attn.out"), dM);

    // Right flank cluster, top-aligned at the attention input. Left→right:
    // qkv_proj.W (tall) · Q · K · V · scores · weights (hero) · out_proj.W.
    const clusterTop = ln1Rect.top;
    let cx = spineHalf + FLANK_MARGIN;
    const qkvRect = placeWB(`attns.${i}.qkv_proj.weight`, `attns.${i}.qkv_proj.bias`, {
      leftX: cx,
      topY: clusterTop,
    });
    cx = qkvRect.right + LANE_GAP;
    const qRect = place(L("attn.q"), [nHead, SEQ, headDim], { leftX: cx, topY: clusterTop });
    cx = qRect.right + LANE_GAP;
    const kRect = place(L("attn.k"), [nHead, SEQ, headDim], { leftX: cx, topY: clusterTop });
    cx = kRect.right + LANE_GAP;
    const vRect = place(L("attn.v"), [nHead, SEQ, headDim], { leftX: cx, topY: clusterTop });
    cx = vRect.right + LANE_GAP;
    const scoresRect = place(L("attn.scores"), [nHead, SEQ, SEQ], { leftX: cx, topY: clusterTop });
    cx = scoresRect.right + LANE_GAP;
    const weightsRect = place(L("attn.weights"), [nHead, SEQ, SEQ], { leftX: cx, topY: clusterTop });
    cx = weightsRect.right + LANE_GAP;
    const outWRect = placeWB(`attns.${i}.out_proj.weight`, `attns.${i}.out_proj.bias`, {
      leftX: cx,
      topY: clusterTop,
    });

    // Drop the spine below the deepest of {attn.out, the right cluster}.
    const attnBottom = union(qkvRect, qRect, kRect, vRect, scoresRect, weightsRect, outWRect).bottom;
    y = Math.min(attnOutRect.bottom, attnBottom) - REGION_GAP - SPINE_GAP;

    // -- MLP band -------------------------------------------------------------
    const ln2Rect = spine(L("ln2.out"), dM);
    placeWB(`ln2s.${i}.weight`, `ln2s.${i}.bias`, {
      rightX: ln2Rect.left - FLANK_MARGIN,
      topY: ln2Rect.top,
    });
    const projOutRect = spine(L("mlp.proj"), dM);

    // Right flank cluster: fc.W (tall) · [fc; act] (wide, stacked) · proj.W.
    const mlpTop = ln2Rect.top;
    let mx = spineHalf + FLANK_MARGIN;
    const fcWRect = placeWB(`mlps.${i}.fc.weight`, `mlps.${i}.fc.bias`, { leftX: mx, topY: mlpTop });
    mx = fcWRect.right + LANE_GAP;
    const fcRect = place(L("mlp.fc"), [SEQ, dMlp], { leftX: mx, topY: mlpTop });
    const actRect = place(L("mlp.act"), [SEQ, dMlp], { leftX: mx, topY: fcRect.bottom - FLANK_GAP });
    mx = union(fcRect, actRect).right + LANE_GAP;
    const projWRect = placeWB(`mlps.${i}.proj.weight`, `mlps.${i}.proj.bias`, {
      leftX: mx,
      topY: mlpTop,
    });

    const mlpBottom = union(fcWRect, actRect, projWRect).bottom;
    y = Math.min(projOutRect.bottom, mlpBottom) - REGION_GAP - SPINE_GAP;
  }

  // final_norm.out + head.logits
  const finRect = place("final_norm.out", [SEQ, dM], { centerX: 0, topY: y });
  placeWB("ln_f.weight", "ln_f.bias", { rightX: finRect.left - FLANK_MARGIN, topY: finRect.top });
  y = finRect.bottom - SPINE_GAP;
  const logitsRect = place("head.logits", [SEQ, dims.vocab_size], { centerX: 0, topY: y });
  placeW("lm_head.weight", { rightX: logitsRect.left - FLANK_MARGIN, topY: logitsRect.top });

  // -- residual-stream spine --------------------------------------------------
  // Glowing connectors threading the centred .out tensors top→bottom, in the
  // vertical gutters so they never occlude a cell — the residual stream.
  const flowMeshes: THREE.Mesh[] = [];
  const spineCol = [...rects.values()]
    .filter((rc) => Math.abs((rc.left + rc.right) / 2) < CELL)
    .sort((p, q) => q.top - p.top);
  for (let i = 1; i < spineCol.length; i++) {
    const seg = createFlowSegment([0, spineCol[i - 1].bottom, 0], [0, spineCol[i].top, 0]);
    root.add(seg);
    flowMeshes.push(seg);
  }

  // -- computation-graph wiring + operation labels ----------------------------
  // Faint edges weight→output and activation→activation, with a small op
  // annotation at each midpoint, so the attention/MLP computation reads as a
  // wired graph rather than isolated grids.
  const wire = (fromName: string, toName: string, op?: string): void => {
    const a = rects.get(fromName);
    const b = rects.get(toName);
    if (!a || !b) return;
    const ay = (a.top + a.bottom) / 2;
    const by = (b.top + b.bottom) / 2;
    const [ax, bx] = a.right <= b.left ? [a.right, b.left] : [a.left, b.right];
    const e = createEdge([ax, ay, 0], [bx, by, 0]);
    root.add(e);
    flowMeshes.push(e);
    if (op) addLabel(op, (ax + bx) / 2, (ay + by) / 2 + 1.7, { size: 1.7, color: "#8a96b0" });
  };

  wire("tok_embedding.weight", "embed.out", "lookup");
  wire("pos_embedding.weight", "embed.out", "+ pos");
  for (let i = 0; i < dims.n_layer; i++) {
    const L = (s: string): string => `layer${i}.${s}`;
    wire(`ln1s.${i}.weight`, L("ln1.out"), "LayerNorm");
    wire(`attns.${i}.qkv_proj.weight`, L("attn.q"), "Q,K,V = ·Wᵀ");
    wire(L("attn.q"), L("attn.scores"), "QKᵀ/√d");
    wire(L("attn.scores"), L("attn.weights"), "softmax");
    wire(L("attn.weights"), L("attn.out"), "·V");
    wire(`attns.${i}.out_proj.weight`, L("attn.out"), "× Wᵀ");
    wire(`ln2s.${i}.weight`, L("ln2.out"), "LayerNorm");
    wire(`mlps.${i}.fc.weight`, L("mlp.fc"), "× Wᵀ");
    wire(L("mlp.fc"), L("mlp.act"), "GELU");
    wire(`mlps.${i}.proj.weight`, L("mlp.proj"), "× Wᵀ");
  }
  wire("ln_f.weight", "final_norm.out", "LayerNorm");
  wire("lm_head.weight", "head.logits", "× Wᵀ");

  // -- anchors ----------------------------------------------------------------

  const r = (name: string): Rect => {
    const rect = rects.get(name);
    if (!rect) throw new Error(`gpt scene: no rect for "${name}"`);
    return rect;
  };

  const allRect = union(...rects.values());
  const cameraHome = frameRect(pad(allRect, 10), false);
  const anchors = new Map<string, CameraKeyframe>();
  const lyr = (i: number, ...names: string[]): Rect => union(...names.map((n) => r(`layer${i}.${n}`)));
  anchors.set("home", cameraHome);
  anchors.set("embed", frameRect(pad(union(r("embed.out"), r("tok_embedding.weight"), r("pos_embedding.weight")), 7), true));
  anchors.set("block0", frameRect(pad(lyr(0, "ln1.out", "attn.out", "ln2.out", "mlp.proj", "attn.weights", "mlp.fc"), 7), true));
  anchors.set("qkv0", frameRect(pad(union(r("layer0.attn.q"), r("layer0.attn.k"), r("layer0.attn.v"), r("attns.0.qkv_proj.weight")), 6), true));
  anchors.set("scores0", frameRect(pad(union(r("layer0.attn.q"), r("layer0.attn.k"), r("layer0.attn.scores")), 6), true));
  anchors.set("weights0", frameRect(pad(union(r("layer0.attn.scores"), r("layer0.attn.weights")), 6), true));
  anchors.set("attnout0", frameRect(pad(union(r("layer0.attn.weights"), r("layer0.attn.v"), r("layer0.attn.out"), r("attns.0.out_proj.weight")), 6), true));
  anchors.set("mlp0", frameRect(pad(union(r("layer0.mlp.fc"), r("layer0.mlp.act"), r("layer0.mlp.proj"), r("mlps.0.fc.weight")), 6), true));
  anchors.set("block1", frameRect(pad(lyr(1, "ln1.out", "attn.out", "ln2.out", "mlp.proj", "attn.weights", "mlp.fc"), 7), true));
  anchors.set("head", frameRect(pad(union(r("final_norm.out"), r("head.logits"), r("lm_head.weight")), 7), true));

  // -- compute plumbing -------------------------------------------------------

  function applyActivations(acts: Map<string, T>): void {
    for (const [name, t] of acts) {
      const view = views.get(name);
      if (!view) throw new Error(`gpt scene: no view for activation "${name}"`);
      view.setValues(t); // exact-shape; GPT never applies a token prefix
    }
    picker?.requestRepick();
  }

  function runTokens(toks: number[]): Map<string, T> {
    const rec = new MapRecorder();
    gptForward(weights, dims, toks, rec);
    return rec.activations;
  }

  const binding: SceneBinding = {
    runForward(uptoToken: number): Map<string, T> {
      if (tokens.length === 0) {
        throw new Error("gpt scene: runForward before setTokens");
      }
      if (!Number.isInteger(uptoToken) || uptoToken < 0 || uptoToken >= tokens.length) {
        throw new Error(`gpt scene: uptoToken ${uptoToken} out of range [0, ${tokens.length})`);
      }
      return runTokens(tokens.slice(0, uptoToken + 1));
    },
    applyActivations,
    setHighlight(names: string[], on: boolean): void {
      for (const name of names) {
        const view = views.get(name);
        if (!view) throw new Error(`gpt scene: setHighlight on unknown "${name}"`);
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
    pulse(): void {
      // No flow pulse in the GPT tour.
    },
  };

  function setTokens(toks: number[]): void {
    if (toks.length < 1 || toks.length > SEQ) {
      throw new Error(`gpt scene: setTokens needs 1..${SEQ} tokens, got ${toks.length}`);
    }
    for (const t of toks) {
      if (!Number.isInteger(t) || t < 0 || t >= dims.vocab_size) {
        throw new Error(`gpt scene: token ${t} outside vocab [0, ${dims.vocab_size})`);
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

  return { views, binding, cameraHome, anchors, setTokens, update, labelObjects, dispose };
}

// ----- guided tour -----------------------------------------------------------

/**
 * The 5-chapter GPT mini-tour. No per-chapter timeline: attention is a
 * one-shot T×T map, not a scan, so each chapter is a camera stop with a
 * static focus. Narration bodies are `gpt.ch.<id>.body`; the page derives
 * the sidebar title as the same key with `.body` → `.title`.
 */
export function buildGptChapters(scene: SceneController, i18n: I18n): ChapterRegistry {
  const at = (anchor: string): CameraKeyframe => {
    const kf = scene.anchors.get(anchor);
    if (!kf) throw new Error(`gpt chapters: scene has no anchor "${anchor}"`);
    return kf;
  };
  const chapters: Chapter[] = [
    { id: "intro", camera: at("home"), highlights: [], narrationKey: "gpt.ch.intro.body" },
    {
      id: "embed",
      camera: at("embed"),
      highlights: ["embed.out", "tok_embedding.weight", "pos_embedding.weight"],
      narrationKey: "gpt.ch.embed.body",
    },
    {
      id: "block",
      camera: at("block0"),
      highlights: ["layer0.ln1.out", "layer0.attn.out", "layer0.ln2.out", "layer0.mlp.proj"],
      narrationKey: "gpt.ch.block.body",
    },
    {
      id: "qkv",
      camera: at("qkv0"),
      highlights: ["layer0.attn.q", "layer0.attn.k", "layer0.attn.v", "attns.0.qkv_proj.weight"],
      narrationKey: "gpt.ch.qkv.body",
    },
    {
      id: "scores",
      camera: at("scores0"),
      highlights: ["layer0.attn.q", "layer0.attn.k", "layer0.attn.scores"],
      narrationKey: "gpt.ch.scores.body",
    },
    {
      id: "weights",
      camera: at("weights0"),
      highlights: ["layer0.attn.scores", "layer0.attn.weights"],
      narrationKey: "gpt.ch.weights.body",
    },
    {
      id: "attnout",
      camera: at("attnout0"),
      highlights: ["layer0.attn.weights", "layer0.attn.v", "layer0.attn.out", "attns.0.out_proj.weight"],
      narrationKey: "gpt.ch.attnout.body",
    },
    {
      id: "mlp",
      camera: at("mlp0"),
      highlights: [
        "layer0.mlp.fc",
        "layer0.mlp.act",
        "layer0.mlp.proj",
        "mlps.0.fc.weight",
        "mlps.0.proj.weight",
      ],
      narrationKey: "gpt.ch.mlp.body",
    },
    {
      id: "layer2",
      camera: at("block1"),
      highlights: ["layer1.ln1.out", "layer1.attn.out", "layer1.ln2.out", "layer1.mlp.proj"],
      narrationKey: "gpt.ch.layer2.body",
    },
    {
      id: "readout",
      camera: at("head"),
      highlights: ["final_norm.out", "head.logits", "lm_head.weight"],
      narrationKey: "gpt.ch.readout.body",
    },
  ];
  return new ChapterRegistry("gpt", chapters, i18n);
}
