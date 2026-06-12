import "./style.css";
import { createSceneShell } from "./engine/scene";
import { OrbitControls } from "./engine/controls";
import { TensorView } from "./engine/tensorView";
import { T } from "./compute/tensor";

const app = document.querySelector<HTMLDivElement>("#app");
if (!app) {
  throw new Error("#app element not found");
}

const shell = createSceneShell(app);

// Demo tensor: deterministic 48x96 test pattern (no network in this task).
// sin/cos interference gives a smooth diverging blue/white/red field; the
// top-left corner carries a few NaN (magenta) and -Infinity (dark slate)
// cells to eyeball the special colors end-to-end.
const ROWS = 48;
const COLS = 96;
const values = new Float32Array(ROWS * COLS);
for (let i = 0; i < ROWS; i++) {
  for (let j = 0; j < COLS; j++) {
    values[i * COLS + j] = Math.sin(i * 0.3) + Math.cos(j * 0.2);
  }
}
for (let i = 0; i < 3; i++) {
  for (let j = 0; j < 3; j++) {
    values[i * COLS + j] = NaN; // magenta cluster
    values[i * COLS + (j + 4)] = -Infinity; // dark-slate cluster
  }
}
const demo = T.from(values, [ROWS, COLS]);

const CELL = 1;
const GAP = 0.25;
const PITCH = CELL + GAP;
const view = new TensorView("demo", [ROWS, COLS], {
  cellSize: CELL,
  gap: GAP,
  // Center the grid on the world origin (row 0 on top, per TensorView docs).
  origin: [(-(COLS - 1) * PITCH) / 2, ((ROWS - 1) * PITCH) / 2, 0],
});
view.setValues(demo);
shell.scene.add(view.mesh);

// Frame the grid: width ~120 world units; at fov 45 a distance of 160
// keeps the full grid in view down to square-ish aspect ratios.
shell.camera.position.set(0, 0, 160);
const controls = new OrbitControls(shell.camera, shell.renderer.domElement);
controls.target.set(0, 0, 0);

shell.start(() => {
  controls.update();
});
