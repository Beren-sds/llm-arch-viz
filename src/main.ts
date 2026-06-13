import "./style.css";
import * as THREE from "three";
import { createSceneShell } from "./engine/scene";
import { OrbitControls } from "./engine/controls";
import { Picker } from "./engine/picking";
import { Tooltip } from "./engine/tooltip";
import { labelsReady } from "./engine/labels";
import { loadModel } from "./compute/loader";
import { buildMambaScene, MAMBA_SEQ_LEN } from "./scenes/mamba";

// A valid 21-token selective-copying input (Task 3's sanity-check example):
// 9 = NOISE, 10 = the recall marker, the tail repeats the data tokens.
// Task 22 replaces this with a proper input UI.
const EXAMPLE_TOKENS = [9, 4, 9, 9, 9, 1, 9, 4, 9, 6, 9, 9, 9, 9, 9, 9, 10, 4, 1, 4, 6];

/** Debug/test hooks for scripts/hover-check.mjs (see bottom of file). */
declare global {
  interface Window {
    __viz?: {
      /** Screen (client px) center of cell (row, col) of the named view. */
      cellScreen(name: string, row: number, col: number): { x: number; y: number } | null;
      /** Current value at (row, col) of the named 2D view. */
      cellValue(name: string, row: number, col: number): number | null;
    };
  }
}

const app = document.querySelector<HTMLDivElement>("#app");
if (!app) {
  throw new Error("#app element not found");
}

const shell = createSceneShell(app);

// Async-settle handshake for the screenshot harness: "0" until the model
// is loaded, the scene is built, and every troika label finished its
// async typeset (see labelsReady). A load failure leaves it at "0" and
// the error propagates — no fake scene is ever shown (Task 22 adds the
// visible retry panel).
document.body.dataset.settled = "0";

const { manifest, weights } = await loadModel("mamba", import.meta.env.BASE_URL);

const picker = new Picker(shell.camera, shell.renderer.domElement);
const scene = buildMambaScene({ scene: shell.scene, weights, manifest, picker });
if (EXAMPLE_TOKENS.length !== MAMBA_SEQ_LEN) {
  throw new Error(`EXAMPLE_TOKENS must have ${MAMBA_SEQ_LEN} tokens`);
}
scene.setTokens(EXAMPLE_TOKENS);

// Camera: start at the scene's home framing (whole spine). Debug aid:
// ?anchor=<name> (e.g. ?anchor=scan0) starts at a named anchor instead.
const anchorName = new URLSearchParams(location.search).get("anchor");
const startView = (anchorName ? scene.anchors.get(anchorName) : undefined) ?? scene.cameraHome;
shell.camera.position.set(...startView.pos);
const controls = new OrbitControls(shell.camera, shell.renderer.domElement);
controls.target.set(...startView.target);

const tooltip = new Tooltip(app);
picker.onPick = (result, x, y) => {
  if (result) tooltip.show(result, x, y);
  else tooltip.hide();
};

void labelsReady(...scene.labelObjects).then(() => {
  document.body.dataset.settled = "1";
});

shell.start(() => {
  controls.update();
  picker.update();
  scene.update(shell.camera);
});

// ---- hover-check.mjs hooks (cheap; no effect on the app itself) -------------

window.__viz = {
  cellScreen(name, row, col) {
    const view = scene.views.get(name);
    if (!view || view.shape.length !== 2) return null;
    const pitch = view.layout.cellSize + view.layout.gap;
    const [ox, oy, oz] = view.layout.origin;
    const v = new THREE.Vector3(ox + col * pitch, oy - row * pitch, oz);
    v.project(shell.camera);
    const rect = shell.renderer.domElement.getBoundingClientRect();
    return {
      x: rect.left + ((v.x + 1) / 2) * rect.width,
      y: rect.top + ((1 - v.y) / 2) * rect.height,
    };
  },
  cellValue(name, row, col) {
    const view = scene.views.get(name);
    if (!view || view.shape.length !== 2) return null;
    return view.lastValues[row * view.shape[1] + col];
  },
};
