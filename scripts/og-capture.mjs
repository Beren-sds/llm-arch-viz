#!/usr/bin/env node
/**
 * Capture the social-preview image: load the landing page and screenshot it
 * at 1200×630 to public/og.png (committed; referenced by the OG/Twitter meta
 * in index.html). Re-run after landing visual changes.
 */

import { resolve } from "node:path";
import { chromium } from "playwright";
import { createServer } from "vite";

const root = resolve(import.meta.dirname, "..");
const server = await createServer({ root, server: { port: 5204, strictPort: false } });
await server.listen();
const baseUrl = server.resolvedUrls.local[0];
const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1200, height: 630 } });
try {
  await page.goto(new URL("#/", baseUrl).href, { waitUntil: "load" });
  await page.waitForSelector(".landing-grid", { timeout: 30_000 });
  await page.waitForTimeout(500);
  const out = resolve(root, "public/og.png");
  await page.screenshot({ path: out });
  console.log(`Saved ${out}`);
} finally {
  await browser.close();
  await server.close();
}
