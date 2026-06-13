import "./style.css";
import { I18n } from "./i18n/i18n";
import { DICTS } from "./i18n/dicts";
import { buildMambaScene, buildMambaChapters, MAMBA_SEQ_LEN } from "./scenes/mamba";
import { buildGptScene, buildGptChapters, GPT_SEQ_LEN } from "./scenes/gpt";
import { buildRwkvScene, buildRwkvChapters, RWKV_SEQ_LEN } from "./scenes/rwkv";
import { buildMoeScene, buildMoeChapters, MOE_SEQ_LEN } from "./scenes/moe";
import { buildKanScene, buildKanChapters, KAN_SEQ_LEN } from "./scenes/kan";
import { buildDiffusionScene, buildDiffusionChapters, DIFFUSION_SEQ_LEN } from "./scenes/diffusion";
import { createRouter, type ArchDef } from "./pages/router";
import type { TaskShape } from "./pages/inputEditor";

// The shared selective-copying input every architecture runs (Task 3's
// sanity example): 9 = NOISE, 10 = GO, the tail repeats the 4 data tokens.
const SELECTIVE_COPY_INPUT = [9, 4, 9, 9, 9, 1, 9, 4, 9, 6, 9, 9, 9, 9, 9, 9, 10, 4, 1, 4, 6];

// Selective-copying task shape (mirrors training/config.yaml `task`); drives
// the live input editor so users can edit the data tokens and re-forward.
const TASK: TaskShape = {
  vocabSize: 16,
  nData: 4,
  contextLen: 16,
  dataIds: [1, 2, 3, 4, 5, 6, 7, 8],
  noiseId: 9,
  goId: 10,
};

function requireEl(selector: string): HTMLDivElement {
  const node = document.querySelector<HTMLDivElement>(selector);
  if (!node) {
    throw new Error(`${selector} element not found`);
  }
  return node;
}

if (
  SELECTIVE_COPY_INPUT.length !== MAMBA_SEQ_LEN ||
  SELECTIVE_COPY_INPUT.length !== GPT_SEQ_LEN ||
  SELECTIVE_COPY_INPUT.length !== RWKV_SEQ_LEN ||
  SELECTIVE_COPY_INPUT.length !== MOE_SEQ_LEN ||
  SELECTIVE_COPY_INPUT.length !== KAN_SEQ_LEN ||
  SELECTIVE_COPY_INPUT.length !== DIFFUSION_SEQ_LEN
) {
  throw new Error("SELECTIVE_COPY_INPUT length must match all scene sequence lengths");
}

const app = requireEl("#app");
const i18n = new I18n(DICTS);

const archs: ArchDef[] = [
  {
    id: "mamba",
    titleKey: "scene.mamba.title",
    tokens: SELECTIVE_COPY_INPUT,
    buildScene: ({ scene, picker, i18n: i }, model) =>
      buildMambaScene({ scene, weights: model.weights, manifest: model.manifest, picker, i18n: i }),
    buildChapters: buildMambaChapters,
  },
  {
    id: "gpt",
    titleKey: "scene.gpt.title",
    tokens: SELECTIVE_COPY_INPUT,
    buildScene: ({ scene, picker, i18n: i }, model) =>
      buildGptScene({ scene, weights: model.weights, manifest: model.manifest, picker, i18n: i }),
    buildChapters: buildGptChapters,
  },
  {
    id: "rwkv",
    titleKey: "scene.rwkv.title",
    tokens: SELECTIVE_COPY_INPUT,
    buildScene: ({ scene, picker, i18n: i }, model) =>
      buildRwkvScene({ scene, weights: model.weights, manifest: model.manifest, picker, i18n: i }),
    buildChapters: buildRwkvChapters,
  },
  {
    id: "moe",
    titleKey: "scene.moe.title",
    tokens: SELECTIVE_COPY_INPUT,
    buildScene: ({ scene, picker, i18n: i }, model) =>
      buildMoeScene({ scene, weights: model.weights, manifest: model.manifest, picker, i18n: i }),
    buildChapters: buildMoeChapters,
  },
  {
    id: "kan",
    titleKey: "scene.kan.title",
    tokens: SELECTIVE_COPY_INPUT,
    buildScene: ({ scene, picker, i18n: i }, model) =>
      buildKanScene({ scene, weights: model.weights, manifest: model.manifest, picker, i18n: i }),
    buildChapters: buildKanChapters,
  },
  {
    id: "diffusion",
    titleKey: "scene.diffusion.title",
    tokens: SELECTIVE_COPY_INPUT,
    buildScene: ({ scene, picker, i18n: i }, model) =>
      buildDiffusionScene({ scene, weights: model.weights, manifest: model.manifest, picker, i18n: i }),
    buildChapters: buildDiffusionChapters,
  },
];

createRouter({
  container: app,
  i18n,
  baseUrl: import.meta.env.BASE_URL,
  archs,
  comingSoon: [],
  task: TASK,
});
