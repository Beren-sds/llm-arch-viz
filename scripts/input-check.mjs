#!/usr/bin/env node
/**
 * Live-input-editor check: prove that editing a data token actually
 * re-forwards the model (an activation cell changes), then that "Randomize"
 * and "Reset" work — and gate console errors throughout.
 *
 * Usage: node scripts/input-check.mjs [#/arch]   (default #/gpt)
 *
 * Reads activation cell values through window.__viz.cellValue (the debug
 * hook archPage installs). embed.out at an edited data position must change
 * when its token changes — a direct, deterministic signal.
 */

import { resolve } from "node:path";
import { chromium } from "playwright";
import { createServer } from "vite";

const urlPath = process.argv[2] ?? "#/gpt";
const root = resolve(import.meta.dirname, "..");
const problems = [];

const server = await createServer({ root, server: { port: 5202, strictPort: false } });
await server.listen();
const baseUrl = server.resolvedUrls.local[0];
const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1280, height: 720 } });
page.on("console", (m) => {
  if (m.type() === "error") problems.push(`console.error: ${m.text()}`);
});
page.on("pageerror", (e) => problems.push(`pageerror: ${e}`));

const fail = (m) => {
  problems.push(m);
};

try {
  await page.goto(new URL(urlPath, baseUrl).href, { waitUntil: "load" });
  await page.waitForSelector(".viz-input", { timeout: 30_000 });
  await page.waitForFunction(() => document.body.dataset.settled !== "0", undefined, {
    timeout: 30_000,
  });
  await page.waitForTimeout(800);

  const dataChips = page.locator(".viz-tok.is-data");
  const n = await dataChips.count();
  console.log(`input editor: ${n} editable data chips`);
  if (n !== 4) fail(`expected 4 data chips, got ${n}`);

  // Find the position of the first data chip (its index among all chips).
  const firstPos = await page.evaluate(() => {
    const chips = [...document.querySelectorAll(".viz-input-row .viz-tok")];
    return chips.findIndex((c) => c.classList.contains("is-data"));
  });

  const before = await page.evaluate((p) => window.__viz.cellValue("embed.out", p, 0), firstPos);
  const beforeText = (await dataChips.first().textContent())?.trim();
  await dataChips.first().click(); // cycle the value
  await page.waitForTimeout(400);
  const afterText = (await dataChips.first().textContent())?.trim();
  const after = await page.evaluate((p) => window.__viz.cellValue("embed.out", p, 0), firstPos);

  console.log(`  chip value ${beforeText} -> ${afterText}; embed.out[${firstPos},0] ${before} -> ${after}`);
  if (beforeText === afterText) fail("chip value did not change on click");
  if (before === after) fail("embed.out did not change after edit (no live re-forward)");

  // Randomize: should change the data positions / values and stay well-formed.
  const randBtn = page.locator(".viz-input-btn").first();
  await randBtn.click();
  await page.waitForTimeout(400);
  const afterRand = await page.locator(".viz-tok.is-data").count();
  if (afterRand !== 4) fail(`after randomize expected 4 data chips, got ${afterRand}`);

  // Reset: restores the canonical example (value 4 at the first data chip).
  const resetBtn = page.locator(".viz-input-btn").nth(1);
  await resetBtn.click();
  await page.waitForTimeout(400);
  const resetText = (await page.locator(".viz-tok.is-data").first().textContent())?.trim();
  if (resetText !== "4") fail(`after reset expected first data chip "4", got "${resetText}"`);

  await page.screenshot({ path: resolve(root, "data/screenshots/input-editor.png") });
  console.log("  screenshot -> data/screenshots/input-editor.png");
} finally {
  await browser.close();
  await server.close();
}

if (problems.length > 0) {
  console.error(`\n${problems.length} problem(s):`);
  for (const p of problems) console.error(`  ${p}`);
  process.exit(1);
}
console.log("\nLive input editor OK (edit re-forwards, randomize + reset work, no console errors).");
