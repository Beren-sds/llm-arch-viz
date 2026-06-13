/**
 * RWKV-4 3D scene — same vertical-spine grammar as the Mamba/GPT scenes on
 * the same selective-copy input. The residual stream runs top→bottom
 * (embed → per-layer ln1/att/ln2/ffn → final norm → logits). The WKV
 * linear-attention output (per-token, T × d_model) is the hero, the RWKV
 * counterpart of Mamba's h-state and GPT's attention map. All values live
 * from compute/rwkv.ts (golden-gated). No per-chapter timeline.
 */

import * as THREE from "three";
import { T } from "../compute/tensor";
import { getTensor, type Manifest } from "../compute/loader";
import { rwkvDimsFrom, rwkvForward } from "../compute/rwkv";
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

export const RWKV_SEQ_LEN = 21;

const SPINE_GAP = 8;
const REGION_GAP = 20;
const FLANK_MARGIN = 8;
const LANE_GAP = 10;
const FLANK_GAP = 7;
const LABEL_RISE = 4.2;

export const RWKV_ANCHOR_NAMES = [
  "home",
  "embed",
  "block0",
  "rkv0",
  "wkv0",
  "timeout0",
  "channelmix0",
  "block1",
  "head",
] as const;

export interface LabelFactory {
  label(text: string, opts?: LabelOptions): THREE.Object3D;
  bracket(opts: DimBracketOptions): THREE.Object3D;
}

export interface RwkvSceneDeps {
  scene: THREE.Scene;
  weights: Map<string, T>;
  manifest: Manifest;
  picker?: Picker;
  i18n?: I18n;
  labelFactory?: LabelFactory;
}

function formulaFor(name: string): string | undefined {
  if (name === "embed.out") return "x = E[token]";
  if (name === "final_norm.out") return "x̂ = LayerNorm(x)";
  if (name === "head.logits") return "logits = W_head·x̂";
  const s = name.replace(/^layer\d+\./, "");
  switch (s) {
    case "ln1.out":
    case "ln2.out":
      return "x̂ = LayerNorm(x)";
    case "att.r":
      return "r = σ(shift(x̂)·Wr)";
    case "att.k":
      return "k = shift(x̂)·Wk";
    case "att.v":
      return "v = shift(x̂)·Wv";
    case "att.wkv":
      return "wkv = Σ exp(−(t−i)w+k)·v / Σ exp(…)";
    case "att.out":
      return "x ← x + (r⊙wkv)·Wo";
    case "ffn.k":
      return "k = relu(shift(x̂)·Wk)²";
    case "ffn.out":
      return "x ← x + σ(r)·(k·Wv)";
    default:
      return undefined;
  }
}

function weightLabel(name: string): string {
  if (name === "embedding.weight") return "embed";
  if (name === "head.weight") return "head";
  if (name === "ln_out.weight") return "ln_out.g";
  if (name === "ln_out.bias") return "ln_out.b";
  const ln = /^ln([12])\.\d+\.(weight|bias)$/.exec(name);
  if (ln) return `ln${ln[1]}.${ln[2] === "weight" ? "g" : "b"}`;
  const m = /^(att|ffn)\.\d+\.(.+)$/.exec(name);
  if (!m) return name;
  return m[2]
    .replace("receptance.weight", "recept.W")
    .replace("key.weight", "key.W")
    .replace("value.weight", "value.W")
    .replace("output.weight", "out.W")
    .replace("time_decay", "decay")
    .replace("time_first", "bonus")
    .replace("time_mix_k", "mix_k")
    .replace("time_mix_v", "mix_v")
    .replace("time_mix_r", "mix_r");
}

export function buildRwkvScene(deps: RwkvSceneDeps): SceneController {
  const { weights, manifest, picker } = deps;
  const dims = rwkvDimsFrom(manifest);
  const SEQ = RWKV_SEQ_LEN;
  const dM = dims.d_model;
  const dF = dims.d_ffn;

  const factory: LabelFactory = deps.labelFactory ?? {
    label: createTensorLabel,
    bracket: createDimBracket,
  };

  const root = new THREE.Group();
  root.name = "rwkv-scene";
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
    if (opts.label !== null) {
      addLabel(opts.label ?? name, (rect.left + rect.right) / 2, rect.top + LABEL_RISE);
    }
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

  const spineHalf = gridSize([SEQ, dM]).w / 2;
  let y = 0;

  const titleText = deps.i18n?.t("scene.rwkv.title") ?? "RWKV — selective copying";
  addLabel(titleText, 0, y + 30, { size: 12, color: "#d7e3ff" });

  // embed
  const embedRect = place("embed.out", [SEQ, dM], { centerX: 0, topY: y });
  placeW("embedding.weight", { rightX: -spineHalf - FLANK_MARGIN, topY: y });
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

    // -- time-mix (WKV) band --------------------------------------------------
    const ln1Rect = spine(L("ln1.out"), dM);
    const ln1w = placeW(`ln1.${i}.weight`, { rightX: ln1Rect.left - FLANK_MARGIN, topY: ln1Rect.top });
    placeW(`ln1.${i}.bias`, { rightX: ln1Rect.left - FLANK_MARGIN, topY: ln1w.bottom - FLANK_GAP });
    const attOutRect = spine(L("att.out"), dM);

    // Right cluster: recept.W · r · key.W · k · value.W · v · wkv(hero) · out.W
    const top = ln1Rect.top;
    let cx = spineHalf + FLANK_MARGIN;
    const lay = (rect: Rect): number => rect.right + LANE_GAP;
    const rW = placeW(`att.${i}.receptance.weight`, { leftX: cx, topY: top });
    cx = lay(rW);
    const rR = place(L("att.r"), [SEQ, dM], { leftX: cx, topY: top });
    cx = lay(rR);
    const kW = placeW(`att.${i}.key.weight`, { leftX: cx, topY: top });
    cx = lay(kW);
    const kR = place(L("att.k"), [SEQ, dM], { leftX: cx, topY: top });
    cx = lay(kR);
    const vW = placeW(`att.${i}.value.weight`, { leftX: cx, topY: top });
    cx = lay(vW);
    const vR = place(L("att.v"), [SEQ, dM], { leftX: cx, topY: top });
    cx = lay(vR);
    const wkvR = place(L("att.wkv"), [SEQ, dM], { leftX: cx, topY: top });
    cx = lay(wkvR);
    const oW = placeW(`att.${i}.output.weight`, { leftX: cx, topY: top });

    // time params (1-D, per channel) tucked left of the spine at this band
    let py = ln1Rect.bottom - FLANK_GAP;
    for (const s of ["time_decay", "time_first", "time_mix_k", "time_mix_v", "time_mix_r"]) {
      const r = placeW(`att.${i}.${s}`, { rightX: -spineHalf - FLANK_MARGIN, topY: py });
      py = r.bottom - FLANK_GAP;
    }

    const attBottom = union(rW, rR, kW, kR, vW, vR, wkvR, oW).bottom;
    y = Math.min(attOutRect.bottom, attBottom) - REGION_GAP - SPINE_GAP;

    // -- channel-mix band -----------------------------------------------------
    const ln2Rect = spine(L("ln2.out"), dM);
    const ln2w = placeW(`ln2.${i}.weight`, { rightX: ln2Rect.left - FLANK_MARGIN, topY: ln2Rect.top });
    placeW(`ln2.${i}.bias`, { rightX: ln2Rect.left - FLANK_MARGIN, topY: ln2w.bottom - FLANK_GAP });
    const ffnOutRect = spine(L("ffn.out"), dM);

    // Right cluster: key.W · k(wide hidden) · value.W · recept.W
    const mtop = ln2Rect.top;
    let mx = spineHalf + FLANK_MARGIN;
    const fkW = placeW(`ffn.${i}.key.weight`, { leftX: mx, topY: mtop });
    mx = fkW.right + LANE_GAP;
    const fk = place(L("ffn.k"), [SEQ, dF], { leftX: mx, topY: mtop });
    mx = fk.right + LANE_GAP;
    const fvW = placeW(`ffn.${i}.value.weight`, { leftX: mx, topY: mtop });
    mx = fvW.right + LANE_GAP;
    const frW = placeW(`ffn.${i}.receptance.weight`, { leftX: mx, topY: mtop });
    let fpy = ln2Rect.bottom - FLANK_GAP;
    for (const s of ["time_mix_k", "time_mix_r"]) {
      const r = placeW(`ffn.${i}.${s}`, { rightX: -spineHalf - FLANK_MARGIN, topY: fpy });
      fpy = r.bottom - FLANK_GAP;
    }

    const ffnBottom = union(fkW, fk, fvW, frW).bottom;
    y = Math.min(ffnOutRect.bottom, ffnBottom) - REGION_GAP - SPINE_GAP;
  }

  // final norm + logits
  const finRect = place("final_norm.out", [SEQ, dM], { centerX: 0, topY: y });
  const lnOutW = placeW("ln_out.weight", { rightX: finRect.left - FLANK_MARGIN, topY: finRect.top });
  placeW("ln_out.bias", { rightX: finRect.left - FLANK_MARGIN, topY: lnOutW.bottom - FLANK_GAP });
  y = finRect.bottom - SPINE_GAP;
  const logitsRect = place("head.logits", [SEQ, dims.vocab_size], { centerX: 0, topY: y });
  placeW("head.weight", { rightX: logitsRect.left - FLANK_MARGIN, topY: logitsRect.top });

  // -- residual-stream spine + wiring -------------------------------------------
  const flowMeshes: THREE.Mesh[] = [];
  const spineCol = [...rects.values()]
    .filter((rc) => Math.abs((rc.left + rc.right) / 2) < CELL)
    .sort((p, q) => q.top - p.top);
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
  wire("embedding.weight", "embed.out", "lookup");
  for (let i = 0; i < dims.n_layer; i++) {
    const L = (s: string): string => `layer${i}.${s}`;
    wire(`ln1.${i}.weight`, L("ln1.out"), "LayerNorm");
    wire(L("att.k"), L("att.wkv"), "WKV");
    wire(L("att.v"), L("att.wkv"), "");
    wire(L("att.wkv"), L("att.out"), "r⊙wkv");
    wire(`att.${i}.output.weight`, L("att.out"), "× Wᵀ");
    wire(`ln2.${i}.weight`, L("ln2.out"), "LayerNorm");
    wire(`ffn.${i}.key.weight`, L("ffn.k"), "relu²");
    wire(L("ffn.k"), L("ffn.out"), "× Wv");
  }
  wire("ln_out.weight", "final_norm.out", "LayerNorm");
  wire("head.weight", "head.logits", "× Wᵀ");

  // -- anchors ------------------------------------------------------------------
  const r = (name: string): Rect => {
    const rect = rects.get(name);
    if (!rect) throw new Error(`rwkv scene: no rect for "${name}"`);
    return rect;
  };
  const allRect = union(...rects.values());
  const cameraHome = frameRect(pad(allRect, 10), false);
  const anchors = new Map<string, CameraKeyframe>();
  const lyr = (i: number, ...names: string[]): Rect => union(...names.map((n) => r(`layer${i}.${n}`)));
  anchors.set("home", cameraHome);
  anchors.set("embed", frameRect(pad(union(r("embed.out"), r("embedding.weight")), 7), true));
  anchors.set("block0", frameRect(pad(lyr(0, "ln1.out", "att.out", "ln2.out", "ffn.out"), 7), true));
  anchors.set(
    "rkv0",
    frameRect(
      pad(
        union(
          r("layer0.att.r"),
          r("layer0.att.k"),
          r("layer0.att.v"),
          r("att.0.receptance.weight"),
          r("att.0.key.weight"),
          r("att.0.value.weight"),
        ),
        6,
      ),
      true,
    ),
  );
  anchors.set(
    "wkv0",
    frameRect(
      pad(union(r("layer0.att.k"), r("layer0.att.v"), r("layer0.att.wkv"), r("att.0.time_decay"), r("att.0.time_first")), 6),
      true,
    ),
  );
  anchors.set(
    "timeout0",
    frameRect(pad(union(r("layer0.att.r"), r("layer0.att.wkv"), r("layer0.att.out"), r("att.0.output.weight")), 6), true),
  );
  anchors.set(
    "channelmix0",
    frameRect(pad(union(r("layer0.ffn.k"), r("layer0.ffn.out"), r("ffn.0.key.weight")), 6), true),
  );
  anchors.set("block1", frameRect(pad(lyr(1, "ln1.out", "att.out", "ln2.out", "ffn.out"), 7), true));
  anchors.set("head", frameRect(pad(union(r("final_norm.out"), r("head.logits"), r("head.weight")), 7), true));

  // -- compute plumbing ---------------------------------------------------------
  function applyActivations(acts: Map<string, T>): void {
    for (const [name, t] of acts) {
      const view = views.get(name);
      if (!view) throw new Error(`rwkv scene: no view for activation "${name}"`);
      view.setValues(t);
    }
    picker?.requestRepick();
  }

  function runTokens(toks: number[]): Map<string, T> {
    const rec = new MapRecorder();
    rwkvForward(weights, dims, toks, rec);
    return rec.activations;
  }

  const binding: SceneBinding = {
    runForward(uptoToken: number): Map<string, T> {
      if (tokens.length === 0) throw new Error("rwkv scene: runForward before setTokens");
      if (!Number.isInteger(uptoToken) || uptoToken < 0 || uptoToken >= tokens.length) {
        throw new Error(`rwkv scene: uptoToken ${uptoToken} out of range [0, ${tokens.length})`);
      }
      return runTokens(tokens.slice(0, uptoToken + 1));
    },
    applyActivations,
    setHighlight(names: string[], on: boolean): void {
      for (const name of names) {
        const view = views.get(name);
        if (!view) throw new Error(`rwkv scene: setHighlight on unknown "${name}"`);
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
      throw new Error(`rwkv scene: setTokens needs 1..${SEQ} tokens, got ${toks.length}`);
    }
    for (const t of toks) {
      if (!Number.isInteger(t) || t < 0 || t >= dims.vocab_size) {
        throw new Error(`rwkv scene: token ${t} outside vocab [0, ${dims.vocab_size})`);
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

export function buildRwkvChapters(scene: SceneController, i18n: I18n): ChapterRegistry {
  const at = (anchor: string): CameraKeyframe => {
    const kf = scene.anchors.get(anchor);
    if (!kf) throw new Error(`rwkv chapters: scene has no anchor "${anchor}"`);
    return kf;
  };
  const chapters: Chapter[] = [
    { id: "intro", camera: at("home"), highlights: [], narrationKey: "rwkv.ch.intro.body" },
    {
      id: "embed",
      camera: at("embed"),
      highlights: ["embed.out", "embedding.weight"],
      narrationKey: "rwkv.ch.embed.body",
    },
    {
      id: "block",
      camera: at("block0"),
      highlights: ["layer0.ln1.out", "layer0.att.out", "layer0.ln2.out", "layer0.ffn.out"],
      narrationKey: "rwkv.ch.block.body",
    },
    {
      id: "rkv",
      camera: at("rkv0"),
      highlights: [
        "layer0.att.r",
        "layer0.att.k",
        "layer0.att.v",
        "att.0.receptance.weight",
        "att.0.key.weight",
        "att.0.value.weight",
      ],
      narrationKey: "rwkv.ch.rkv.body",
    },
    {
      id: "wkv",
      camera: at("wkv0"),
      highlights: ["layer0.att.k", "layer0.att.v", "layer0.att.wkv", "att.0.time_decay", "att.0.time_first"],
      narrationKey: "rwkv.ch.wkv.body",
    },
    {
      id: "timeout",
      camera: at("timeout0"),
      highlights: ["layer0.att.r", "layer0.att.wkv", "layer0.att.out", "att.0.output.weight"],
      narrationKey: "rwkv.ch.timeout.body",
    },
    {
      id: "channelmix",
      camera: at("channelmix0"),
      highlights: ["layer0.ffn.k", "layer0.ffn.out", "ffn.0.key.weight", "ffn.0.value.weight"],
      narrationKey: "rwkv.ch.channelmix.body",
    },
    {
      id: "layer2",
      camera: at("block1"),
      highlights: ["layer1.ln1.out", "layer1.att.out", "layer1.ln2.out", "layer1.ffn.out"],
      narrationKey: "rwkv.ch.layer2.body",
    },
    {
      id: "readout",
      camera: at("head"),
      highlights: ["final_norm.out", "head.logits", "head.weight"],
      narrationKey: "rwkv.ch.readout.body",
    },
  ];
  return new ChapterRegistry("rwkv", chapters, i18n);
}
