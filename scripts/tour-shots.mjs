#!/usr/bin/env node
/**
 * Guided-tour visual acceptance: serve the app, click through every chapter
 * in the sidebar (camera fly-to + focus + any per-chapter timeline), and
 * screenshot each to data/screenshots/tour-NN-<id>.png. Also flips the
 * locale once and screenshots the result.
 *
 * Gates on console errors / page exceptions across the WHOLE run — a missing
 * i18n key, a setHighlight on an unknown tensor, or a shader regression all
 * surface here, not just on first paint.
 *
 * Usage:
 *   node scripts/tour-shots.mjs [url-path]
 *   (url-path defaults to the app root; pass e.g. "#/gpt" once routing lands)
 */

import { mkdirSync } from "node:fs";
import { resolve } from "node:path";
import { chromium } from "playwright";
import { createServer } from "vite";

const root = resolve(import.meta.dirname, "..");
const urlPath = process.argv[2] ?? "";
const outDir = resolve(root, "data/screenshots");
mkdirSync(outDir, { recursive: true });

const server = await createServer({ root, server: { port: 5198, strictPort: false } });
await server.listen();
const baseUrl = server.resolvedUrls.local[0];

const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1280, height: 720 } });
const problems = [];
page.on("console", (m) => {
  if (m.type() === "error") problems.push(`console.error: ${m.text()}`);
});
page.on("pageerror", (e) => problems.push(`pageerror: ${e}`));

const pad2 = (n) => String(n).padStart(2, "0");

try {
  const url = new URL(urlPath, baseUrl).href;
  console.log(`Loading ${url}`);
  await page.goto(url, { waitUntil: "load" });
  await page.waitForSelector("canvas", { timeout: 30_000 });
  await page.waitForFunction(() => document.body.dataset.settled !== "0", undefined, {
    timeout: 30_000,
  });

  const count = await page.locator(".viz-chapter-item").count();
  console.log(`${count} chapters`);
  if (count === 0) throw new Error("no .viz-chapter-item buttons found");

  for (let i = 0; i < count; i++) {
    const item = page.locator(".viz-chapter-item").nth(i);
    const title = (await item.textContent())?.trim() ?? `ch${i}`;
    await item.click();
    // Camera fly is 1000ms (wall-clock); give the scan chapter extra so its
    // token-by-token timeline visibly advances before the frame is grabbed.
    const hasTimeline = !(await page.locator(".viz-slider-row.is-hidden").count());
    await page.waitForTimeout(hasTimeline ? 3500 : 1700);
    const body = (await page.locator(".viz-narration-body").textContent())?.trim() ?? "";
    const counter = (await page.locator(".viz-counter").textContent())?.trim() ?? "";
    const id = title.replace(/^\d+\.\s*/, "").toLowerCase().replace(/[^a-z0-9]+/g, "-");
    const file = resolve(outDir, `tour-${pad2(i)}-${id}.png`);
    await page.screenshot({ path: file });
    console.log(`  [${counter}] ${title} — "${body.slice(0, 56)}…"  -> ${file}`);
  }

  // Locale flip: rebuilds the page; verify it lands and re-screenshot ch.1.
  await page.locator(".viz-lang").click();
  await page.waitForFunction(() => document.body.dataset.settled !== "0", undefined, {
    timeout: 30_000,
  });
  await page.waitForTimeout(1700);
  await page.screenshot({ path: resolve(outDir, "tour-zh.png") });
  console.log(`  locale flip -> ${resolve(outDir, "tour-zh.png")}`);
} finally {
  await browser.close();
  await server.close();
}

if (problems.length > 0) {
  console.error(`\n${problems.length} problem(s):`);
  for (const p of problems) console.error(`  ${p}`);
  process.exit(1);
}
console.log("\nNo console errors across the full tour.");
