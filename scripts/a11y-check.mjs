#!/usr/bin/env node
/**
 * Accessibility check:
 *  1. prefers-reduced-motion — a timeline chapter (mamba scan) must NOT
 *     auto-play (the transport stays on "Play", not "Pause").
 *  2. mobile layout — at a phone viewport the arch chrome (sidebar,
 *     narration, input) all render; screenshot to data/screenshots/mobile.png.
 * Console errors fail the run in both passes.
 */

import { resolve } from "node:path";
import { chromium } from "playwright";
import { createServer } from "vite";

const root = resolve(import.meta.dirname, "..");
const problems = [];
const server = await createServer({ root, server: { port: 5205, strictPort: false } });
await server.listen();
const baseUrl = server.resolvedUrls.local[0];
const browser = await chromium.launch();

async function load(page, path) {
  page.on("console", (m) => {
    if (m.type() === "error") problems.push(`console.error: ${m.text()}`);
  });
  page.on("pageerror", (e) => problems.push(`pageerror: ${e}`));
  await page.goto(new URL(path, baseUrl).href, { waitUntil: "load" });
  await page.waitForSelector("canvas", { timeout: 30_000 });
  await page.waitForFunction(() => document.body.dataset.settled !== "0", undefined, {
    timeout: 30_000,
  });
  await page.waitForTimeout(800);
}

try {
  // 1. Reduced motion: the scan chapter's looping timeline must not auto-run.
  const rm = await browser.newPage({ viewport: { width: 1280, height: 720 }, reducedMotion: "reduce" });
  await load(rm, "#/mamba/scan");
  const playLabel = (await rm.locator(".viz-play").textContent())?.trim();
  console.log(`reduced-motion: scan transport shows "${playLabel}"`);
  if (playLabel !== "Play") problems.push(`reduced-motion: timeline auto-played (transport="${playLabel}")`);
  await rm.close();

  // 2. Mobile layout: chrome panels all present, no overflow crash.
  const mob = await browser.newPage({ viewport: { width: 390, height: 844 } });
  await load(mob, "#/gpt");
  for (const sel of [".viz-sidebar", ".viz-narration", ".viz-input", "canvas"]) {
    if ((await mob.locator(sel).count()) === 0) problems.push(`mobile: missing ${sel}`);
  }
  await mob.screenshot({ path: resolve(root, "data/screenshots/mobile.png") });
  console.log("mobile: screenshot -> data/screenshots/mobile.png");
  await mob.close();
} finally {
  await browser.close();
  await server.close();
}

if (problems.length > 0) {
  console.error(`\n${problems.length} problem(s):`);
  for (const p of problems) console.error(`  ${p}`);
  process.exit(1);
}
console.log("\nA11y OK (reduced-motion respected, mobile chrome renders, no console errors).");
