/**
 * Landing page: the atlas cover. A hero plus a grid of architecture cards.
 * Live archs (mamba, gpt) are clickable and call onOpen; the rest are
 * "coming soon" placeholders — shown so the roadmap is visible, but not
 * clickable, so there are no dead routes. Bilingual via i18n (its own EN/ZH
 * toggle); the router rebuilds it on locale change.
 */

import type { I18n } from "../i18n/i18n";
import { button, el } from "./dom";

export interface LandingCard {
  /** Architecture id; card.<id>.title / card.<id>.desc must exist in i18n. */
  id: string;
  /** Live (clickable) vs coming-soon (disabled). */
  live: boolean;
}

export interface LandingDeps {
  container: HTMLElement;
  i18n: I18n;
  cards: LandingCard[];
  /** Open a live architecture (the router turns this into a hash navigation). */
  onOpen: (archId: string) => void;
  /** Open the cross-architecture comparison page. */
  onCompare: () => void;
}

export interface Landing {
  dispose(): void;
}

export function createLanding(deps: LandingDeps): Landing {
  const { container, i18n } = deps;

  const root = el("div", "landing");

  const top = el("div", "landing-top");
  const langBtn = button("viz-lang");
  langBtn.addEventListener("click", () => {
    i18n.setLocale(i18n.locale === "en" ? "zh" : "en");
  });
  top.appendChild(langBtn);

  const hero = el("header", "landing-hero");
  const h1 = el("h1", "landing-title");
  const sub = el("p", "landing-subtitle");
  const compareBtn = button("landing-compare");
  compareBtn.addEventListener("click", deps.onCompare);
  hero.append(h1, sub, compareBtn);

  const grid = el("div", "landing-grid");
  const refs: Array<{ card: LandingCard; title: HTMLElement; desc: HTMLElement; action: HTMLElement }> =
    [];
  for (const card of deps.cards) {
    const c = el("article", `landing-card${card.live ? "" : " is-soon"}`);
    const title = el("h2", "landing-card-title");
    const desc = el("p", "landing-card-desc");
    let action: HTMLElement;
    if (card.live) {
      const open = button("landing-open");
      open.addEventListener("click", () => deps.onOpen(card.id));
      action = open;
    } else {
      action = el("span", "landing-soon");
    }
    c.append(title, desc, action);
    // The whole live card is a click target, not just the button.
    if (card.live) {
      c.classList.add("is-live");
      c.addEventListener("click", (e) => {
        if (e.target !== action) deps.onOpen(card.id);
      });
    }
    grid.appendChild(c);
    refs.push({ card, title, desc, action });
  }

  root.append(top, hero, grid);
  container.appendChild(root);

  function render(): void {
    langBtn.textContent = i18n.t("ui.langToggle");
    h1.textContent = i18n.t("ui.landing.title");
    sub.textContent = i18n.t("ui.landing.subtitle");
    compareBtn.textContent = i18n.t("ui.compare");
    for (const { card, title, desc, action } of refs) {
      title.textContent = i18n.t(`card.${card.id}.title`);
      desc.textContent = i18n.t(`card.${card.id}.desc`);
      action.textContent = card.live ? i18n.t("ui.open") : i18n.t("ui.comingSoon");
    }
  }
  render();

  return {
    dispose(): void {
      root.remove();
    },
  };
}
