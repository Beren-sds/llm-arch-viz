#!/usr/bin/env node
/**
 * Hover/tooltip visual check: serve the app, move the mouse over known
 * cells of the demo grid, and screenshot the tooltip for inspection.
 *
 * Usage: node scripts/hover-check.mjs
 *
 * Writes data/screenshots/hover-center.png (a normal finite cell) and
 * data/screenshots/hover-nan.png (a NaN cell from the demo's top-left
 * cluster), and prints the tooltip's text content for each hover so the
 * run is also checkable from the terminal.
 *
 * Cell -> screen math (must match src/main.ts):
 *   48x96 grid, cellSize 1, gap 0.25 (pitch 1.25), centered on origin;
 *   camera at z=160, fov 45 -> visible height 2*160*tan(22.5deg) ~= 132.55
 *   world units over the viewport height -> px/world = vh / 132.55.
 *   Cell (i, j) center: x = -(COLS-1)*P/2 + j*P, y = (ROWS-1)*P/2 - i*P.
 */

import { mkdirSync } from "node:fs";
import { resolve } from "node:path";
import { chromium } from "playwright";
import { createServer } from "vite";

const root = resolve(import.meta.dirname, "..");
const VW = 1280;
const VH = 720;

const ROWS = 48;
const COLS = 96;
const PITCH = 1.25;
const PX_PER_WORLD = VH / (2 * 160 * Math.tan((45 / 2) * (Math.PI / 180)));

/** Screen position (px) of cell (i, j)'s center. */
function cellScreen(i, j) {
  const wx = (-(COLS - 1) * PITCH) / 2 + j * PITCH;
  const wy = ((ROWS - 1) * PITCH) / 2 - i * PITCH;
  return { x: VW / 2 + wx * PX_PER_WORLD, y: VH / 2 - wy * PX_PER_WORLD };
}

const server = await createServer({ root, server: { port: 5199, strictPort: false } });
await server.listen();
const baseUrl = server.resolvedUrls.local[0];
const browser = await chromium.launch();
const problems = [];
let failed = false;

try {
  const page = await browser.newPage({ viewport: { width: VW, height: VH } });
  page.on("console", (m) => {
    if (m.type() === "error") problems.push(`console.error: ${m.text()}`);
  });
  page.on("pageerror", (e) => problems.push(`pageerror: ${e}`));

  await page.goto(baseUrl, { waitUntil: "load" });
  await page.waitForSelector("canvas", { timeout: 10_000 });
  await page.waitForTimeout(1000); // let the render loop settle

  const outDir = resolve(root, "data/screenshots");
  mkdirSync(outDir, { recursive: true });

  const hovers = [
    { name: "hover-center", cell: [24, 48] }, // finite value mid-grid
    { name: "hover-nan", cell: [1, 1] }, // NaN cluster, top-left
  ];

  for (const { name, cell } of hovers) {
    const [i, j] = cell;
    const { x, y } = cellScreen(i, j);
    await page.mouse.move(x, y);
    await page.waitForTimeout(200); // >1 frame: picker.update() consumes the move
    const text = await page.evaluate(() => {
      const el = document.querySelector(".tensor-tooltip");
      return el && el.classList.contains("visible") ? el.textContent : null;
    });
    console.log(`${name}: cell [${i}, ${j}] at (${x.toFixed(1)}, ${y.toFixed(1)}) -> ${text}`);
    if (text === null) {
      failed = true;
      console.error(`${name}: tooltip NOT visible`);
    }
    await page.screenshot({ path: resolve(outDir, `${name}.png`) });
  }
} finally {
  await browser.close();
  await server.close();
}

if (problems.length > 0) {
  console.error(`Page reported ${problems.length} problem(s):`);
  for (const p of problems) console.error(`  ${p}`);
  failed = true;
}
process.exit(failed ? 1 : 0);
