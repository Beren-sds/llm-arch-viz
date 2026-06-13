/**
 * Comparison page: the atlas thesis made visible — "one task, many machines".
 * A contrast table over every live architecture: how it moves information
 * between positions, its cost, its feed-forward, live parameter count, and
 * the one-line trick. Conceptual cells are i18n (compare.<id>.*); parameter
 * counts are computed live from each manifest (no weights fetch). A failed
 * manifest fetch shows "—", never a fabricated number.
 */

import type { I18n } from "../i18n/i18n";
import type { Manifest } from "../compute/loader";
import { button, el } from "./dom";

/** Total parameters in a manifest = Σ over tensors of the shape product. */
export function totalParams(manifest: { tensors: { shape: number[] }[] }): number {
  return manifest.tensors.reduce((sum, t) => sum + t.shape.reduce((a, d) => a * d, 1), 0);
}

/** Compact human count: 61234 -> "61K", 1_250_000 -> "1.3M". */
export function formatParams(n: number): string {
  if (n >= 1e6) return `${(n / 1e6).toFixed(n >= 1e7 ? 0 : 1)}M`;
  if (n >= 1e3) return `${(n / 1e3).toFixed(n >= 1e4 ? 0 : 1)}K`;
  return String(n);
}

export interface CompareDeps {
  container: HTMLElement;
  i18n: I18n;
  /** import.meta.env.BASE_URL — Pages path prefix, for manifest fetches. */
  baseUrl: string;
  /** Architecture ids, in display order. */
  archs: string[];
  onOpen: (archId: string) => void;
  onHome: () => void;
  /** Override for tests. */
  fetchFn?: typeof fetch;
}

export interface ComparePage {
  /** Resolves once every parameter cell has settled (filled or "—"). */
  readonly ready: Promise<void>;
  dispose(): void;
}

const COLS = ["arch", "mix", "cost", "ff", "params", "idea"] as const;

export function createComparePage(deps: CompareDeps): ComparePage {
  const { container, i18n, archs } = deps;
  const fetchFn = deps.fetchFn ?? fetch;
  const base = deps.baseUrl.replace(/\/+$/, "");

  const root = el("div", "compare");

  const top = el("div", "compare-top");
  const homeBtn = button("viz-home");
  homeBtn.addEventListener("click", deps.onHome);
  const langBtn = button("viz-lang");
  langBtn.addEventListener("click", () => i18n.setLocale(i18n.locale === "en" ? "zh" : "en"));
  top.append(homeBtn, langBtn);

  const hero = el("header", "compare-hero");
  const h1 = el("h1", "compare-title");
  const sub = el("p", "compare-subtitle");
  hero.append(h1, sub);

  const table = el("table", "compare-table");
  const thead = el("thead", "");
  const headRow = el("tr", "");
  const headCells = COLS.map(() => el("th", ""));
  for (const th of headCells) headRow.appendChild(th);
  thead.appendChild(headRow);

  const tbody = el("tbody", "");
  /** Per-arch refs so render() can re-apply i18n text without rebuilding. */
  const rows: Array<{
    id: string;
    name: HTMLButtonElement;
    mix: HTMLElement;
    cost: HTMLElement;
    ff: HTMLElement;
    params: HTMLElement;
    idea: HTMLElement;
  }> = [];
  for (const id of archs) {
    const tr = el("tr", "compare-row");
    const nameCell = el("td", "compare-cell-arch");
    const name = button("compare-open");
    name.addEventListener("click", () => deps.onOpen(id));
    nameCell.appendChild(name);
    const mix = el("td", "");
    const cost = el("td", "compare-cell-cost");
    const ff = el("td", "");
    const params = el("td", "compare-cell-params");
    const idea = el("td", "compare-cell-idea");
    tr.append(nameCell, mix, cost, ff, params, idea);
    tbody.appendChild(tr);
    rows.push({ id, name, mix, cost, ff, params, idea });
  }

  table.append(thead, tbody);
  root.append(top, hero, table);
  container.appendChild(root);

  function render(): void {
    homeBtn.textContent = `‹ ${i18n.t("ui.home")}`;
    langBtn.textContent = i18n.t("ui.langToggle");
    h1.textContent = i18n.t("compare.title");
    sub.textContent = i18n.t("compare.subtitle");
    for (let i = 0; i < COLS.length; i++) headCells[i].textContent = i18n.t(`compare.col.${COLS[i]}`);
    for (const row of rows) {
      row.name.textContent = i18n.t(`card.${row.id}.title`);
      row.mix.textContent = i18n.t(`compare.${row.id}.mix`);
      row.cost.textContent = i18n.t(`compare.${row.id}.cost`);
      row.ff.textContent = i18n.t(`compare.${row.id}.ff`);
      row.idea.textContent = i18n.t(`compare.${row.id}.idea`);
    }
  }
  render();

  // Live parameter counts: fetch each manifest (no weights). "…" while
  // pending, formatted count on success, "—" on failure (never faked).
  for (const row of rows) row.params.textContent = "…";
  const ready = Promise.all(
    rows.map(async (row) => {
      try {
        const res = await fetchFn(`${base}/models/${row.id}/manifest.json`);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const manifest = (await res.json()) as Manifest;
        row.params.textContent = formatParams(totalParams(manifest));
      } catch (err) {
        console.error(`compare: param count for "${row.id}" failed:`, err);
        row.params.textContent = "—";
      }
    }),
  ).then(() => {});

  // Locale changes are handled by the router remounting this page (same as
  // the landing and arch pages), so no self-subscription here.
  return {
    ready,
    dispose(): void {
      root.remove();
    },
  };
}
