#!/usr/bin/env node
/**
 * Headless visual check: serve the app, wait for the WebGL canvas,
 * screenshot it to data/screenshots/<name>.png.
 *
 * Usage:
 *   node scripts/screenshot.mjs <name> [url-path] [--preview]
 *
 *   <name>      output file stem (data/screenshots/<name>.png)
 *   [url-path]  path relative to the served base (default: app root)
 *   --preview   serve dist/ via `vite preview` (run `npm run build` first)
 *               instead of the default vite dev server
 *
 * Server: uses vite's Node API (createServer / preview) on an ephemeral
 * port — no race against an externally started dev server. Headless
 * chromium on macOS ships WebGL2 via ANGLE/SwiftShader, which is enough
 * for this check (verified: the instanced ShaderMaterial compiles).
 *
 * Exit code 1 when the page logs console errors or throws — a shader
 * compile failure (e.g. a USE_INSTANCING regression) surfaces here even
 * though a canvas still mounts.
 */

import { mkdirSync } from "node:fs";
import { resolve } from "node:path";
import { chromium } from "playwright";
import { createServer, preview } from "vite";

const argv = process.argv.slice(2);
const usePreview = argv.includes("--preview");
const args = argv.filter((a) => a !== "--preview");
const name = args[0];
if (!name) {
  console.error("usage: node scripts/screenshot.mjs <name> [url-path] [--preview]");
  process.exit(2);
}
const urlPath = args[1] ?? "";

const root = resolve(import.meta.dirname, "..");

let server;
let baseUrl;
if (usePreview) {
  server = await preview({ root, preview: { port: 4199, strictPort: false } });
  baseUrl = server.resolvedUrls.local[0];
} else {
  server = await createServer({ root, server: { port: 5199, strictPort: false } });
  await server.listen();
  baseUrl = server.resolvedUrls.local[0];
}

const browser = await chromium.launch(); // default headless; WebGL2 via SwiftShader
const page = await browser.newPage({ viewport: { width: 1280, height: 720 } });

const problems = [];
page.on("console", (m) => {
  if (m.type() === "error") problems.push(`console.error: ${m.text()}`);
});
page.on("pageerror", (e) => problems.push(`pageerror: ${e}`));

const url = new URL(urlPath, baseUrl).href;
console.log(`Loading ${url}`);
await page.goto(url, { waitUntil: "load" });
await page.waitForSelector("canvas", { timeout: 10_000 });
await page.waitForTimeout(1500); // let the render loop draw a few frames

const outDir = resolve(root, "data/screenshots");
mkdirSync(outDir, { recursive: true });
const outPath = resolve(outDir, `${name}.png`);
await page.screenshot({ path: outPath });

await browser.close();
if (usePreview) {
  await new Promise((r) => server.httpServer.close(r));
} else {
  await server.close();
}

if (problems.length > 0) {
  console.error(`Page reported ${problems.length} problem(s):`);
  for (const p of problems) console.error(`  ${p}`);
  console.error(`Screenshot still saved to ${outPath} for inspection.`);
  process.exit(1);
}
console.log(`Saved ${outPath}`);
