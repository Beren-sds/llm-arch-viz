#!/usr/bin/env node
/**
 * Comparison-page check: load #/compare, confirm a row per architecture with
 * live parameter counts fetched from the real manifests (not "…"/"—"), and
 * gate console errors. Screenshots to data/screenshots/compare.png.
 */

import { resolve } from "node:path";
import { chromium } from "playwright";
import { createServer } from "vite";

const root = resolve(import.meta.dirname, "..");
const problems = [];

const server = await createServer({ root, server: { port: 5203, strictPort: false } });
await server.listen();
const baseUrl = server.resolvedUrls.local[0];
const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1280, height: 860 } });
page.on("console", (m) => {
  if (m.type() === "error") problems.push(`console.error: ${m.text()}`);
});
page.on("pageerror", (e) => problems.push(`pageerror: ${e}`));

try {
  await page.goto(new URL("#/compare", baseUrl).href, { waitUntil: "load" });
  await page.waitForSelector(".compare-table", { timeout: 30_000 });
  await page.waitForFunction(() => document.body.dataset.settled !== "0", undefined, {
    timeout: 30_000,
  });
  await page.waitForTimeout(400);

  const rows = await page.locator(".compare-row").count();
  console.log(`compare: ${rows} rows`);
  if (rows !== 6) problems.push(`expected 6 rows, got ${rows}`);

  const params = await page.locator(".compare-cell-params").allTextContents();
  console.log(`  param counts: ${params.join(", ")}`);
  for (const p of params) {
    if (p === "…") problems.push("a param cell never resolved (still …)");
    if (p === "—") problems.push("a param cell failed to fetch its manifest (—)");
  }

  await page.screenshot({ path: resolve(root, "data/screenshots/compare.png") });
  console.log("  screenshot -> data/screenshots/compare.png");
} finally {
  await browser.close();
  await server.close();
}

if (problems.length > 0) {
  console.error(`\n${problems.length} problem(s):`);
  for (const p of problems) console.error(`  ${p}`);
  process.exit(1);
}
console.log("\nComparison page OK (6 rows, live param counts, no console errors).");
