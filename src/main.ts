import "./style.css";
import { I18n } from "./i18n/i18n";
import { DICTS } from "./i18n/dicts";
import { loadModel, type LoadedModel } from "./compute/loader";
import { buildMambaScene, buildMambaChapters, MAMBA_SEQ_LEN } from "./scenes/mamba";
import { createArchPage, type ArchPage } from "./pages/archPage";

// A valid 21-token selective-copying input (Task 3's sanity-check example):
// 9 = NOISE, 10 = GO, the tail repeats the 4 data tokens in order.
// Task 22 replaces this with a landing page + an input UI + routing.
const EXAMPLE_TOKENS = [9, 4, 9, 9, 9, 1, 9, 4, 9, 6, 9, 9, 9, 9, 9, 9, 10, 4, 1, 4, 6];

function requireEl(selector: string): HTMLDivElement {
  const node = document.querySelector<HTMLDivElement>(selector);
  if (!node) {
    throw new Error(`${selector} element not found`);
  }
  return node;
}

// Declared (not flow-narrowed) non-null type so the mount() closure keeps it.
const app = requireEl("#app");
if (EXAMPLE_TOKENS.length !== MAMBA_SEQ_LEN) {
  throw new Error(`EXAMPLE_TOKENS must have ${MAMBA_SEQ_LEN} tokens`);
}

const i18n = new I18n(DICTS);

let page: ArchPage | null = null;
let model: LoadedModel | null = null;

/**
 * Mount the Mamba page at `chapterId` (or its first chapter). The screenshot
 * harness watches document.body.dataset.settled: "0" until the model loads,
 * the scene builds, and every troika label has typeset. A load failure
 * leaves it at "0" and the error propagates — no fake scene is shown (Task
 * 22 adds the visible retry panel).
 */
async function mount(chapterId?: string): Promise<void> {
  document.body.dataset.settled = "0";
  page?.dispose();
  // Fetch once; locale rebuilds reuse the cached weights.
  if (!model) model = await loadModel("mamba", import.meta.env.BASE_URL);
  const { manifest, weights } = model;
  page = createArchPage({
    container: app,
    i18n,
    titleKey: "scene.mamba.title",
    tokens: EXAMPLE_TOKENS,
    buildScene: ({ scene, picker }) => buildMambaScene({ scene, weights, manifest, picker, i18n }),
    buildChapters: buildMambaChapters,
    initialChapterId: chapterId,
  });
  void page.ready.then(() => {
    document.body.dataset.settled = "1";
  });
}

// Locale toggle → rebuild in place at the same chapter (the 3D title label is
// baked per build, so a swap is the clean way to retranslate the whole page).
i18n.onChange(() => {
  void mount(page?.chapterId);
});

await mount();
