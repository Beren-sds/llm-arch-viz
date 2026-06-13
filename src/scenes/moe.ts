/**
 * MoE 3D scene — a GPT whose MLP is a top-k mixture of experts. Same
 * vertical-spine grammar; attention clusters identical to the GPT scene.
 * The MoE band shows the router probabilities and the top-k gate matrix
 * (the routing hero) feeding four expert MLPs whose dense outputs combine
 * into moe.out. All values live from compute/moe.ts (golden-gated).
 */

import * as THREE from "three";
import { T } from "../compute/tensor";
import { getTensor, type Manifest } from "../compute/loader";
import { moeDimsFrom, moeForward } from "../compute/moe";
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

export const MOE_SEQ_LEN = 21;

const SPINE_GAP = 8;
const REGION_GAP = 20;
const FLANK_MARGIN = 8;
const LANE_GAP = 10;
const FLANK_GAP = 7;
const LABEL_RISE = 4.2;

export const MOE_ANCHOR_NAMES = ["home", "embed", "attn0", "moe0", "head"] as const;

export interface LabelFactory {
  label(text: string, opts?: LabelOptions): THREE.Object3D;
  bracket(opts: DimBracketOptions): THREE.Object3D;
}

export interface MoeSceneDeps {
  scene: THREE.Scene;
  weights: Map<string, T>;
  manifest: Manifest;
  picker?: Picker;
  i18n?: I18n;
  labelFactory?: LabelFactory;
}

function formulaFor(name: string): string | undefined {
  if (name === "embed.out") return "x = TokEmb[t] + PosEmb[t]";
  if (name === "final_norm.out") return "x̂ = LayerNorm(x)";
  if (name === "head.logits") return "logits = W_lm·x̂";
  const s = name.replace(/^layer\d+\./, "");
  if (s === "ln1.out" || s === "ln2.out") return "x̂ = LayerNorm(x)";
  if (s === "attn.q") return "Q = x̂·W_qkv (q)";
  if (s === "attn.k") return "K = x̂·W_qkv (k)";
  if (s === "attn.v") return "V = x̂·W_qkv (v)";
  if (s === "attn.scores") return "S = Q·Kᵀ/√d (causal)";
  if (s === "attn.weights") return "A = softmax(S)";
  if (s === "attn.out") return "x ← x + (A·V)·W_o";
  if (s === "moe.router") return "p = softmax(x̂·W_router)";
  if (s === "moe.gates") return "top-k(p), renormalized";
  if (/^moe\.expert\d+\.out$/.test(s)) return "Eₑ(x̂) = proj(GELU(fc·x̂))";
  if (s === "moe.out") return "x ← x + Σ gateₑ · Eₑ(x̂)";
  return undefined;
}

function weightLabel(name: string): string {
  if (name === "tok_embedding.weight") return "tok_embed";
  if (name === "pos_embedding.weight") return "pos_embed";
  if (name === "lm_head.weight") return "lm_head";
  if (name === "ln_f.weight") return "ln_f.g";
  if (name === "ln_f.bias") return "ln_f.b";
  const ln = /^(ln1s|ln2s)\.\d+\.(weight|bias)$/.exec(name);
  if (ln) return `${ln[1] === "ln1s" ? "ln1" : "ln2"}.${ln[2] === "weight" ? "g" : "b"}`;
  const ex = /^moes\.\d+\.experts\.(\d+)\.(.+)$/.exec(name);
  if (ex) return `e${ex[1]}.${ex[2].replace(".weight", ".W").replace(".bias", ".b")}`;
  if (/^moes\.\d+\.router\.weight$/.test(name)) return "router.W";
  const at = /^attns\.\d+\.(.+)$/.exec(name);
  if (at) return at[1].replace(".weight", ".W").replace(".bias", ".b");
  return name;
}

export function buildMoeScene(deps: MoeSceneDeps): SceneController {
  const { weights, manifest, picker } = deps;
  const dims = moeDimsFrom(manifest);
  const SEQ = MOE_SEQ_LEN;
  const dM = dims.d_model;
  const nHead = dims.n_head;
  const headDim = dims.head_dim;
  const nExp = dims.n_experts;

  const factory: LabelFactory = deps.labelFactory ?? {
    label: createTensorLabel,
    bracket: createDimBracket,
  };

  const root = new THREE.Group();
  root.name = "moe-scene";
  deps.scene.add(root);

  const views = new Map<string, TensorView>();
  const rects = new Map<string, Rect>();
  const labelObjects: THREE.Object3D[] = [];
  const billboards: Array<(camera: THREE.Camera) => void> = [];
  let tokens: number[] = [];
  let disposed = false;

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
    if (opts.label !== null) addLabel(opts.label ?? name, (rect.left + rect.right) / 2, rect.top + LABEL_RISE);
    if (picker && opts.pickable !== false) {
      const formula = formulaFor(name);
      picker.add(formula === undefined ? { view } : { view, formula });
    }
    return rect;
  }
  function placeW(name: string, opts: Omit<PlaceOpts, "values" | "label">): Rect {
    const t = getTensor(weights, name);
    return place(name, t.shape, { ...opts, values: t, label: weightLabel(name) });
  }
  function placeWB(wn: string, bn: string, opts: Omit<PlaceOpts, "values" | "label">): Rect {
    const wr = placeW(wn, opts);
    const br = placeW(bn, { leftX: wr.left, topY: wr.bottom - FLANK_GAP });
    return union(wr, br);
  }

  const spineHalf = gridSize([SEQ, dM]).w / 2;
  let y = 0;

  addLabel(deps.i18n?.t("scene.moe.title") ?? "MoE — selective copying", 0, y + 30, {
    size: 12,
    color: "#d7e3ff",
  });

  const embedRect = place("embed.out", [SEQ, dM], { centerX: 0, topY: y });
  const tokRect = placeW("tok_embedding.weight", { rightX: -spineHalf - FLANK_MARGIN, topY: y });
  placeW("pos_embedding.weight", { rightX: -spineHalf - FLANK_MARGIN, topY: tokRect.bottom - FLANK_GAP });
  addBracket({ from: [embedRect.left, embedRect.bottom, 0], to: [embedRect.right, embedRect.bottom, 0], offset: -2, label: `d_model = ${dM}` });
  addBracket({ from: [embedRect.left, embedRect.top, 0], to: [embedRect.left, embedRect.bottom, 0], offset: -2, label: `T = ${SEQ}` });
  y = embedRect.bottom - REGION_GAP - SPINE_GAP;

  for (let i = 0; i < dims.n_layer; i++) {
    const L = (s: string): string => `layer${i}.${s}`;
    const spine = (name: string, cols: number): Rect => {
      const rect = place(name, [SEQ, cols], { centerX: 0, topY: y, label: name });
      y = rect.bottom - SPINE_GAP;
      return rect;
    };

    // -- attention band (identical to the GPT scene) --------------------------
    const ln1Rect = spine(L("ln1.out"), dM);
    placeWB(`ln1s.${i}.weight`, `ln1s.${i}.bias`, { rightX: ln1Rect.left - FLANK_MARGIN, topY: ln1Rect.top });
    const attnOutRect = spine(L("attn.out"), dM);
    const aTop = ln1Rect.top;
    let cx = spineHalf + FLANK_MARGIN;
    const qkv = placeWB(`attns.${i}.qkv_proj.weight`, `attns.${i}.qkv_proj.bias`, { leftX: cx, topY: aTop });
    cx = qkv.right + LANE_GAP;
    const q = place(L("attn.q"), [nHead, SEQ, headDim], { leftX: cx, topY: aTop });
    cx = q.right + LANE_GAP;
    const k = place(L("attn.k"), [nHead, SEQ, headDim], { leftX: cx, topY: aTop });
    cx = k.right + LANE_GAP;
    const v = place(L("attn.v"), [nHead, SEQ, headDim], { leftX: cx, topY: aTop });
    cx = v.right + LANE_GAP;
    const sc = place(L("attn.scores"), [nHead, SEQ, SEQ], { leftX: cx, topY: aTop });
    cx = sc.right + LANE_GAP;
    const wt = place(L("attn.weights"), [nHead, SEQ, SEQ], { leftX: cx, topY: aTop });
    cx = wt.right + LANE_GAP;
    const ow = placeWB(`attns.${i}.out_proj.weight`, `attns.${i}.out_proj.bias`, { leftX: cx, topY: aTop });
    y = Math.min(attnOutRect.bottom, union(qkv, q, k, v, sc, wt, ow).bottom) - REGION_GAP - SPINE_GAP;

    // -- MoE band -------------------------------------------------------------
    const ln2Rect = spine(L("ln2.out"), dM);
    placeWB(`ln2s.${i}.weight`, `ln2s.${i}.bias`, { rightX: ln2Rect.left - FLANK_MARGIN, topY: ln2Rect.top });
    const moeOutRect = spine(L("moe.out"), dM);

    const mTop = ln2Rect.top;
    let mx = spineHalf + FLANK_MARGIN;
    const rW = placeW(`moes.${i}.router.weight`, { leftX: mx, topY: mTop });
    mx = rW.right + LANE_GAP;
    const router = place(L("moe.router"), [SEQ, nExp], { leftX: mx, topY: mTop });
    mx = router.right + LANE_GAP;
    const gates = place(L("moe.gates"), [SEQ, nExp], { leftX: mx, topY: mTop });
    mx = gates.right + LANE_GAP + 2;
    const expRects: Rect[] = [];
    for (let e = 0; e < nExp; e++) {
      const eo = place(L(`moe.expert${e}.out`), [SEQ, dM], { leftX: mx, topY: mTop });
      const fcw = placeWB(`moes.${i}.experts.${e}.fc.weight`, `moes.${i}.experts.${e}.fc.bias`, {
        leftX: mx,
        topY: eo.bottom - FLANK_GAP,
      });
      placeWB(`moes.${i}.experts.${e}.proj.weight`, `moes.${i}.experts.${e}.proj.bias`, {
        leftX: mx,
        topY: fcw.bottom - FLANK_GAP,
      });
      expRects.push(eo);
      mx = union(eo, fcw).right + LANE_GAP;
    }
    y =
      Math.min(moeOutRect.bottom, union(rW, router, gates, ...expRects).bottom) -
      REGION_GAP -
      SPINE_GAP;
  }

  const finRect = place("final_norm.out", [SEQ, dM], { centerX: 0, topY: y });
  placeWB("ln_f.weight", "ln_f.bias", { rightX: finRect.left - FLANK_MARGIN, topY: finRect.top });
  y = finRect.bottom - SPINE_GAP;
  const logitsRect = place("head.logits", [SEQ, dims.vocab_size], { centerX: 0, topY: y });
  placeW("lm_head.weight", { rightX: logitsRect.left - FLANK_MARGIN, topY: logitsRect.top });

  // -- spine + wiring -----------------------------------------------------------
  const flowMeshes: THREE.Mesh[] = [];
  const spineCol = [...rects.values()]
    .filter((rc) => Math.abs((rc.left + rc.right) / 2) < CELL)
    .sort((p, q2) => q2.top - p.top);
  for (let i = 1; i < spineCol.length; i++) {
    const seg = createFlowSegment([0, spineCol[i - 1].bottom, 0], [0, spineCol[i].top, 0]);
    root.add(seg);
    flowMeshes.push(seg);
  }
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
  for (let i = 0; i < dims.n_layer; i++) {
    const L = (s: string): string => `layer${i}.${s}`;
    wire(`ln1s.${i}.weight`, L("ln1.out"), "LayerNorm");
    wire(L("attn.weights"), L("attn.out"), "·V");
    wire(`ln2s.${i}.weight`, L("ln2.out"), "LayerNorm");
    wire(`moes.${i}.router.weight`, L("moe.router"), "softmax");
    wire(L("moe.router"), L("moe.gates"), "top-k");
    for (let e = 0; e < nExp; e++) wire(L(`moe.expert${e}.out`), L("moe.out"), e === 0 ? "Σ gate·E" : "");
  }
  wire("ln_f.weight", "final_norm.out", "LayerNorm");
  wire("lm_head.weight", "head.logits", "× Wᵀ");

  // -- anchors ------------------------------------------------------------------
  const r = (name: string): Rect => {
    const rect = rects.get(name);
    if (!rect) throw new Error(`moe scene: no rect for "${name}"`);
    return rect;
  };
  const allRect = union(...rects.values());
  const cameraHome = frameRect(pad(allRect, 10), false);
  const anchors = new Map<string, CameraKeyframe>();
  anchors.set("home", cameraHome);
  anchors.set("embed", frameRect(pad(union(r("embed.out"), r("tok_embedding.weight"), r("pos_embedding.weight")), 7), true));
  anchors.set("attn0", frameRect(pad(union(r("layer0.attn.q"), r("layer0.attn.weights"), r("layer0.attn.out")), 6), true));
  anchors.set(
    "moe0",
    frameRect(pad(union(r("layer0.moe.gates"), r("layer0.moe.expert0.out"), r("layer0.moe.expert3.out"), r("layer0.moe.out")), 6), true),
  );
  anchors.set("head", frameRect(pad(union(r("final_norm.out"), r("head.logits"), r("lm_head.weight")), 7), true));

  // -- compute ------------------------------------------------------------------
  function applyActivations(acts: Map<string, T>): void {
    for (const [name, t] of acts) {
      const view = views.get(name);
      if (!view) throw new Error(`moe scene: no view for activation "${name}"`);
      view.setValues(t);
    }
    picker?.requestRepick();
  }
  function runTokens(toks: number[]): Map<string, T> {
    const rec = new MapRecorder();
    moeForward(weights, dims, toks, rec);
    return rec.activations;
  }
  const binding: SceneBinding = {
    runForward(uptoToken: number): Map<string, T> {
      if (tokens.length === 0) throw new Error("moe scene: runForward before setTokens");
      if (!Number.isInteger(uptoToken) || uptoToken < 0 || uptoToken >= tokens.length) {
        throw new Error(`moe scene: uptoToken ${uptoToken} out of range [0, ${tokens.length})`);
      }
      return runTokens(tokens.slice(0, uptoToken + 1));
    },
    applyActivations,
    setHighlight(names: string[], on: boolean): void {
      for (const name of names) {
        const view = views.get(name);
        if (!view) throw new Error(`moe scene: setHighlight on unknown "${name}"`);
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
    pulse(): void {},
  };
  function setTokens(toks: number[]): void {
    if (toks.length < 1 || toks.length > SEQ) {
      throw new Error(`moe scene: setTokens needs 1..${SEQ} tokens, got ${toks.length}`);
    }
    for (const t of toks) {
      if (!Number.isInteger(t) || t < 0 || t >= dims.vocab_size) {
        throw new Error(`moe scene: token ${t} outside vocab [0, ${dims.vocab_size})`);
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

export function buildMoeChapters(scene: SceneController, i18n: I18n): ChapterRegistry {
  const at = (anchor: string): CameraKeyframe => {
    const kf = scene.anchors.get(anchor);
    if (!kf) throw new Error(`moe chapters: scene has no anchor "${anchor}"`);
    return kf;
  };
  const expertHighlights = [0, 1, 2, 3].map((e) => `layer0.moe.expert${e}.out`);
  const chapters: Chapter[] = [
    { id: "intro", camera: at("home"), highlights: [], narrationKey: "moe.ch.intro.body" },
    {
      id: "embed",
      camera: at("embed"),
      highlights: ["embed.out", "tok_embedding.weight", "pos_embedding.weight"],
      narrationKey: "moe.ch.embed.body",
    },
    {
      id: "attention",
      camera: at("attn0"),
      highlights: ["layer0.attn.scores", "layer0.attn.weights", "layer0.attn.out"],
      narrationKey: "moe.ch.attention.body",
    },
    {
      id: "moe",
      camera: at("moe0"),
      highlights: ["layer0.moe.router", "layer0.moe.gates", "layer0.moe.out", ...expertHighlights],
      narrationKey: "moe.ch.moe.body",
    },
    {
      id: "readout",
      camera: at("head"),
      highlights: ["final_norm.out", "head.logits", "lm_head.weight"],
      narrationKey: "moe.ch.readout.body",
    },
  ];
  return new ChapterRegistry("moe", chapters, i18n);
}
