/**
 * Hash router for the atlas: `#/` (and anything unrecognized) → landing;
 * `#/<arch>` / `#/<arch>/<chapter>` → that architecture's page at that
 * chapter. Unknown archs fall back to landing, so there are never dead
 * routes. The arch page echoes chapter changes back into the URL; the
 * router recognizes that echo and adjusts the open page in place instead of
 * remounting (no camera reset on every sidebar click).
 *
 * Error states (never synthetic data, per repo red lines): a model-fetch
 * failure shows a retry panel; missing WebGL2 leaves createSceneShell's
 * bilingual overlay visible. Locale changes rebuild the current view (3D
 * labels are baked per build).
 */

import type * as THREE from "three";
import { parseHash, toHash, type ChapterRegistry } from "../walkthrough/chapters";
import { loadModel, type LoadedModel } from "../compute/loader";
import type { Picker } from "../engine/picking";
import type { I18n } from "../i18n/i18n";
import type { SceneController } from "../scenes/sceneController";
import { createArchPage, type ArchPage } from "./archPage";
import type { TaskShape } from "./inputEditor";
import { createLanding, type Landing } from "./landing";
import { createComparePage, type ComparePage } from "./compare";
import { button, el } from "./dom";

export interface ArchDef {
  /** Route id and i18n card key suffix (card.<id>.title/desc). */
  id: string;
  /** i18n key for the page title. */
  titleKey: string;
  /** Input sequence fed to the model. */
  tokens: number[];
  /** Build the 3D scene from a loaded model. */
  buildScene: (
    ctx: { scene: THREE.Scene; picker: Picker; i18n: I18n },
    model: LoadedModel,
  ) => SceneController;
  /** Build the chapter registry over the built scene. */
  buildChapters: (scene: SceneController, i18n: I18n) => ChapterRegistry;
}

export interface RouterDeps {
  container: HTMLElement;
  i18n: I18n;
  /** import.meta.env.BASE_URL — the GitHub Pages path prefix. */
  baseUrl: string;
  /** Live architectures, in landing-grid order. */
  archs: ArchDef[];
  /** Coming-soon card ids (shown disabled; not routable). */
  comingSoon: string[];
  /** If given, each arch page mounts a live input-token editor. */
  task?: TaskShape;
}

export type Route =
  | { kind: "landing" }
  | { kind: "compare" }
  | { kind: "arch"; arch: string; chapterId?: string };

/**
 * Pure hash → route resolution (unit-tested). `#/compare` is the comparison
 * page; a known arch (optionally + chapter) is its page; anything else —
 * `#/`, empty, malformed, or an unimplemented arch — is the landing.
 */
export function resolveRoute(hash: string, knownArchs: readonly string[]): Route {
  const parsed = parseHash(hash);
  if (parsed && parsed.scene === "compare" && parsed.chapterId === undefined) {
    return { kind: "compare" };
  }
  if (parsed && knownArchs.includes(parsed.scene)) {
    return parsed.chapterId === undefined
      ? { kind: "arch", arch: parsed.scene }
      : { kind: "arch", arch: parsed.scene, chapterId: parsed.chapterId };
  }
  return { kind: "landing" };
}

type Mounted =
  | { kind: "landing"; page: Landing }
  | { kind: "compare"; page: ComparePage }
  | { kind: "arch"; arch: string; page: ArchPage }
  | { kind: "arch-error"; arch: string; chapterId?: string }
  | null;

export function createRouter(deps: RouterDeps): { dispose(): void } {
  const { container, i18n, baseUrl } = deps;
  const archById = new Map(deps.archs.map((a) => [a.id, a]));
  const knownArchs = deps.archs.map((a) => a.id);
  const models = new Map<string, LoadedModel>();

  let mounted: Mounted = null;
  let overlay: HTMLElement | null = null;
  /** Bumped on every mount; async work checks it to bail when superseded. */
  let mountToken = 0;

  function clearOverlay(): void {
    overlay?.remove();
    overlay = null;
  }

  function teardown(): void {
    if (mounted && mounted.kind !== "arch-error") mounted.page.dispose();
    mounted = null;
    clearOverlay();
  }

  function showOverlay(build: (root: HTMLElement) => void): void {
    clearOverlay();
    overlay = el("div", "viz-overlay");
    build(overlay);
    container.appendChild(overlay);
  }

  function mountLanding(): void {
    teardown();
    const page = createLanding({
      container,
      i18n,
      cards: [
        ...knownArchs.map((id) => ({ id, live: true })),
        ...deps.comingSoon.map((id) => ({ id, live: false })),
      ],
      onOpen: (id) => {
        location.hash = toHash(id);
      },
      onCompare: () => {
        location.hash = toHash("compare");
      },
    });
    mounted = { kind: "landing", page };
    document.body.dataset.settled = "1"; // landing has no async labels
  }

  function mountCompare(): void {
    teardown();
    document.body.dataset.settled = "0";
    const page = createComparePage({
      container,
      i18n,
      baseUrl,
      archs: knownArchs,
      onOpen: (id) => {
        location.hash = toHash(id);
      },
      onHome: () => {
        location.hash = "#/";
      },
    });
    mounted = { kind: "compare", page };
    void page.ready.then(() => {
      if (mounted?.kind === "compare" && mounted.page === page) {
        document.body.dataset.settled = "1";
      }
    });
  }

  async function mountArch(arch: string, chapterId?: string): Promise<void> {
    const def = archById.get(arch);
    if (!def) {
      mountLanding();
      return;
    }
    teardown();
    const my = ++mountToken;
    document.body.dataset.settled = "0";
    showOverlay((root) => {
      const msg = el("p", "viz-overlay-msg");
      msg.textContent = i18n.t("ui.loading");
      root.appendChild(msg);
    });

    let model = models.get(arch);
    if (!model) {
      try {
        model = await loadModel(arch, baseUrl);
        models.set(arch, model);
      } catch (err) {
        if (my !== mountToken) return; // navigated away mid-fetch
        console.error(`model load failed for "${arch}":`, err);
        showOverlay((root) => {
          const msg = el("p", "viz-overlay-msg");
          msg.textContent = i18n.t("ui.loadError");
          const retry = button("viz-btn");
          retry.textContent = i18n.t("ui.retry");
          retry.addEventListener("click", () => void mountArch(arch, chapterId));
          root.append(msg, retry);
        });
        mounted = { kind: "arch-error", arch, chapterId };
        return;
      }
    }
    if (my !== mountToken) return; // navigated away during the await
    clearOverlay();

    const loaded = model;
    try {
      const page = createArchPage({
        container,
        i18n,
        titleKey: def.titleKey,
        tokens: def.tokens,
        task: deps.task,
        buildScene: ({ scene, picker }) => def.buildScene({ scene, picker, i18n }, loaded),
        buildChapters: def.buildChapters,
        initialChapterId: chapterId,
        onChapterChange: (id) => {
          location.hash = toHash(arch, id);
        },
        onHome: () => {
          location.hash = "#/";
        },
      });
      mounted = { kind: "arch", arch, page };
      void page.ready.then(() => {
        // Only flip if this page is still the mounted one.
        if (mounted?.kind === "arch" && mounted.page === page) {
          document.body.dataset.settled = "1";
        }
      });
    } catch (err) {
      // createSceneShell mounts a bilingual .gl-error overlay then throws
      // when WebGL2 is absent. Leave it visible; track for locale rebuilds.
      console.error(`scene build failed for "${arch}":`, err);
      mounted = { kind: "arch-error", arch, chapterId };
    }
  }

  function route(): void {
    const r = resolveRoute(location.hash, knownArchs);
    if (r.kind === "compare") {
      if (mounted?.kind === "compare") return;
      mountCompare();
      return;
    }
    if (r.kind === "arch") {
      // The arch page echoes its own chapter changes into the URL. If the
      // same arch is already mounted, treat the hashchange as a deep-link /
      // echo and adjust in place rather than tearing the scene down.
      if (mounted?.kind === "arch" && mounted.arch === r.arch) {
        if (r.chapterId && mounted.page.chapterId !== r.chapterId) {
          mounted.page.goToChapterId(r.chapterId);
        }
        return;
      }
      void mountArch(r.arch, r.chapterId);
    } else {
      if (mounted?.kind === "landing") return;
      mountLanding();
    }
  }

  const onHashChange = (): void => route();
  window.addEventListener("hashchange", onHashChange);

  const unsubscribe = i18n.onChange(() => {
    if (mounted?.kind === "arch") void mountArch(mounted.arch, mounted.page.chapterId);
    else if (mounted?.kind === "arch-error") void mountArch(mounted.arch, mounted.chapterId);
    else if (mounted?.kind === "compare") mountCompare();
    else mountLanding();
  });

  route(); // initial render from the current URL

  return {
    dispose(): void {
      window.removeEventListener("hashchange", onHashChange);
      unsubscribe();
      teardown();
    },
  };
}
