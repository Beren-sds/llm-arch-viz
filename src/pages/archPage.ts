/**
 * Arch page: the interactive shell around one architecture's 3D scene —
 * canvas + orbit + hover tooltip + a chapter-driven guided tour (camera
 * fly-to, per-chapter tensor focus, and the token-by-token scan timeline)
 * + bilingual chrome (sidebar, narration, play/step controls).
 *
 * Architecture-agnostic: it drives anything satisfying SceneController plus
 * a ChapterRegistry, so Mamba and GPT share this one driver.
 *
 * Camera/orbit handoff (per cameraTour.ts's ownership contract): the
 * TourPlayer owns the camera while animating (controls.update is skipped);
 * when idle, OrbitControls drives and we feed the live view back via
 * syncCurrent so the next fly-to starts from where the user left it. A user
 * grab ('start') cancels an in-flight fly-to.
 *
 * Highlight/dim ownership (per timeline.ts): a chapter applies a STATIC
 * focus (setDim + setHighlight) once on entry; its optional timeline uses
 * only stepToken steps, so it never touches highlight/dim and cannot fight
 * the static focus. Leaving a chapter stops the timeline, clears the focus,
 * and restores the full-sequence activations (the scan leaves a prefix).
 */

import * as THREE from "three";
import { createSceneShell, type SceneShell } from "../engine/scene";
import { OrbitControls } from "../engine/controls";
import { Picker } from "../engine/picking";
import { Tooltip } from "../engine/tooltip";
import { labelsReady } from "../engine/labels";
import { TourPlayer } from "../engine/cameraTour";
import { TimelinePlayer } from "../walkthrough/timeline";
import type { ChapterRegistry, Chapter } from "../walkthrough/chapters";
import type { I18n } from "../i18n/i18n";
import type { SceneController } from "../scenes/sceneController";
import { button, el } from "./dom";
import { createInputEditor, type InputEditor, type TaskShape } from "./inputEditor";

declare global {
  interface Window {
    /** Debug hooks for scripts/hover-check.mjs; owned by the active page. */
    __viz?: {
      owner: unknown;
      cellScreen(name: string, row: number, col: number): { x: number; y: number } | null;
      cellValue(name: string, row: number, col: number): number | null;
    };
  }
}

export interface ArchPageDeps {
  /** Where the page mounts (its own canvas + chrome go inside). */
  container: HTMLElement;
  i18n: I18n;
  /** i18n key for the scene title shown in the top bar. */
  titleKey: string;
  /** The input sequence fed to the model (full-length). */
  tokens: number[];
  /** If given, mounts a live token editor that re-forwards on edit. */
  task?: TaskShape;
  /** Build the 3D scene into `scene`, registering pick targets on `picker`. */
  buildScene: (ctx: { scene: THREE.Scene; picker: Picker }) => SceneController;
  /** Build the chapter registry over the just-built scene. */
  buildChapters: (scene: SceneController, i18n: I18n) => ChapterRegistry;
  /** Open at this chapter id (deep link); falls back to the first chapter. */
  initialChapterId?: string;
  /** Notified whenever the active chapter changes (router updates the URL). */
  onChapterChange?: (chapterId: string) => void;
  /** If given, a "‹ Atlas" link in the top bar invokes this (back to landing). */
  onHome?: () => void;
}

export interface ArchPage {
  /** Current chapter id (for URL sync / locale-rebuild restore). */
  readonly chapterId: string;
  /** Navigate to a chapter by id (deep link from the router); no-op if unknown. */
  goToChapterId(id: string): void;
  /** Resolves once every troika label has typeset (screenshot handshake). */
  readonly ready: Promise<void>;
  dispose(): void;
}

/** narrationKey `…​.body` → the sidebar title key `…​.title`. */
function titleKeyOf(ch: Chapter): string {
  return ch.narrationKey.replace(/\.body$/, ".title");
}

/** Quiet time before the camera starts its gentle showcase orbit. */
const IDLE_ORBIT_MS = 6000;

/** Respect the OS "reduce motion" setting (gates idle orbit + timeline autoplay). */
function prefersReducedMotion(): boolean {
  return typeof matchMedia === "function" && matchMedia("(prefers-reduced-motion: reduce)").matches;
}

export function createArchPage(deps: ArchPageDeps): ArchPage {
  const { container, i18n } = deps;
  const reduceMotion = prefersReducedMotion();
  // Mutable: the live input editor reassigns this on every edit, and
  // goToChapter restores the full sequence from it when leaving a chapter.
  let tokens = deps.tokens.slice();

  // ---- DOM scaffold ---------------------------------------------------------
  const root = el("div", "viz-page");
  const canvasHost = el("div", "viz-canvas");
  root.appendChild(canvasHost);
  container.appendChild(root);

  // ---- GL shell + scene -----------------------------------------------------
  // createSceneShell mounts a .gl-error overlay and throws when WebGL2 is
  // absent; the caller (main/router) catches and surfaces the page error.
  const shell = createSceneShell(canvasHost);
  // Vignette over the canvas (under the chrome): darkens the edges so the
  // scene reads with depth and the eye settles on the centre.
  canvasHost.appendChild(el("div", "viz-vignette"));
  const picker = new Picker(shell.camera, shell.renderer.domElement);
  const scene = deps.buildScene({ scene: shell.scene, picker });
  const chapters = deps.buildChapters(scene, i18n);

  const tooltip = new Tooltip(canvasHost);
  picker.onPick = (result, x, y) => {
    if (result) tooltip.show(result, x, y);
    else tooltip.hide();
  };

  const controls = new OrbitControls(shell.camera, shell.renderer.domElement);
  const tour = new TourPlayer({
    applyView: (pos, target) => {
      shell.camera.position.set(pos[0], pos[1], pos[2]);
      controls.target.set(target[0], target[1], target[2]);
    },
    durationMs: 1000,
  });
  // Idle-orbit clock: any user grab (or chapter change) resets it; the render
  // loop turns on auto-rotate once the view has been quiet long enough.
  let lastUserMs = performance.now();
  controls.onStart(() => {
    tour.cancel(); // a user grab interrupts an in-flight fly-to
    picker.enabled = true;
    lastUserMs = performance.now();
  });

  const timeline = new TimelinePlayer(scene.binding);

  // ---- chrome: top bar ------------------------------------------------------
  const topbar = el("div", "viz-topbar");
  const homeBtn = button("viz-home");
  if (deps.onHome) homeBtn.addEventListener("click", deps.onHome);
  else homeBtn.hidden = true;
  const titleEl = el("div", "viz-title");
  const counterEl = el("div", "viz-counter");
  const langBtn = button("viz-lang");
  langBtn.addEventListener("click", () => {
    i18n.setLocale(i18n.locale === "en" ? "zh" : "en");
  });
  topbar.append(homeBtn, titleEl, counterEl, langBtn);

  // ---- chrome: chapter sidebar ----------------------------------------------
  const sidebar = el("nav", "viz-sidebar");
  const sidebarH = el("div", "viz-sidebar-h");
  const list = el("ul", "viz-chapter-list");
  const itemButtons: HTMLButtonElement[] = [];
  for (let i = 0; i < chapters.count; i++) {
    const li = document.createElement("li");
    const b = button("viz-chapter-item");
    const idx = i;
    b.addEventListener("click", () => goToChapter(idx));
    itemButtons.push(b);
    li.appendChild(b);
    list.appendChild(li);
  }
  sidebar.append(sidebarH, list);

  // ---- chrome: narration + transport ----------------------------------------
  const narration = el("div", "viz-narration");
  const bodyEl = el("p", "viz-narration-body");
  const ctrlRow = el("div", "viz-controls");
  const prevBtn = button("viz-btn");
  const playBtn = button("viz-btn viz-play");
  const nextBtn = button("viz-btn");
  prevBtn.addEventListener("click", () => goToChapter(current - 1));
  nextBtn.addEventListener("click", () => goToChapter(current + 1));
  playBtn.addEventListener("click", togglePlay);
  ctrlRow.append(prevBtn, playBtn, nextBtn);

  const sliderRow = el("div", "viz-slider-row");
  const sliderLabel = el("label", "viz-slider-label");
  const slider = document.createElement("input");
  slider.type = "range";
  slider.min = "0";
  slider.max = String(Math.max(0, tokens.length - 1));
  slider.step = "1";
  slider.value = "0";
  slider.className = "viz-slider";
  const sliderVal = el("span", "viz-slider-val");
  slider.addEventListener("input", () => {
    const t = Number(slider.value);
    if (timeline.state === "playing") timeline.pause();
    scene.binding.applyActivations(scene.binding.runForward(t));
    sliderVal.textContent = String(t);
    updatePlayLabel();
  });
  sliderRow.append(sliderLabel, slider, sliderVal);
  narration.append(bodyEl, ctrlRow, sliderRow);

  root.append(topbar, sidebar, narration);

  // ---- chrome: live input editor (optional) ---------------------------------
  let inputEditor: InputEditor | null = null;
  if (deps.task) {
    inputEditor = createInputEditor({
      task: deps.task,
      initial: tokens,
      i18n,
      onChange: (t) => {
        tokens = t;
        if (timeline.state === "playing") timeline.pause();
        scene.setTokens(tokens); // live re-forward with the edited input
        slider.value = "0";
        sliderVal.textContent = "0";
        updatePlayLabel();
      },
    });
    root.appendChild(inputEditor.el);
  }

  // ---- chapter state machine ------------------------------------------------
  let current = 0;
  let focused: string[] = [];

  function goToChapter(idx: number, instant = false): void {
    const clamped = Math.min(Math.max(idx, 0), chapters.count - 1);

    // Leave the current chapter: stop its timeline, drop the focus, and
    // restore the full activations (the scan chapter leaves a token prefix).
    timeline.stop();
    if (focused.length > 0) scene.binding.setHighlight(focused, false);
    scene.binding.setDim(null);
    scene.setTokens(tokens);

    current = clamped;
    lastUserMs = performance.now(); // don't auto-orbit straight after a nav
    const ch = chapters.get(clamped);

    // Camera: gate hover during the flight, restore on arrival.
    picker.enabled = false;
    tour.goTo(ch.camera, {
      instant,
      onArrive: () => {
        picker.enabled = true;
      },
    });

    // Static focus (compatible with the scan's stepToken-only timeline).
    if (ch.highlights.length > 0) {
      scene.binding.setDim(ch.highlights);
      scene.binding.setHighlight(ch.highlights, true);
    }
    focused = ch.highlights;

    if (ch.timeline) {
      // The slider scrubs this chapter's step range — token-prefix length for
      // the scan, denoising-step count for diffusion's sampler.
      let maxStep = 0;
      for (const s of ch.timeline.steps) if (s.kind === "stepToken") maxStep = Math.max(maxStep, s.token);
      slider.max = String(maxStep);
      slider.value = "0";
      sliderVal.textContent = "0";
      timeline.load(ch.timeline);
      // Reduced-motion: don't auto-run a looping animation. The user can press
      // Play or scrub the slider to step through it manually.
      if (!reduceMotion) timeline.play();
    }

    renderChrome();
    deps.onChapterChange?.(ch.id);
  }

  function togglePlay(): void {
    if (chapters.get(current).timeline === undefined) return;
    if (timeline.state === "playing") timeline.pause();
    else timeline.play();
    updatePlayLabel();
  }

  // ---- chrome rendering -----------------------------------------------------
  function updatePlayLabel(): void {
    playBtn.textContent = i18n.t(timeline.state === "playing" ? "ui.pause" : "ui.play");
  }

  function renderChrome(): void {
    titleEl.textContent = i18n.t(deps.titleKey);
    homeBtn.textContent = `‹ ${i18n.t("ui.home")}`;
    langBtn.textContent = i18n.t("ui.langToggle");
    sidebarH.textContent = i18n.t("ui.chapters");
    prevBtn.textContent = i18n.t("ui.prev");
    nextBtn.textContent = i18n.t("ui.next");
    sliderLabel.textContent = i18n.t("ui.tokenStep");
    counterEl.textContent = `${current + 1} / ${chapters.count}`;

    for (let i = 0; i < chapters.count; i++) {
      const ch = chapters.get(i);
      itemButtons[i].textContent = `${i + 1}. ${i18n.t(titleKeyOf(ch))}`;
      itemButtons[i].classList.toggle("is-current", i === current);
    }

    const ch = chapters.get(current);
    bodyEl.textContent = i18n.t(ch.narrationKey);
    // Restart the fade-up so the narration animates in on each chapter change.
    bodyEl.classList.remove("is-fading");
    void bodyEl.offsetWidth;
    bodyEl.classList.add("is-fading");
    prevBtn.disabled = current === 0;
    nextBtn.disabled = current === chapters.count - 1;

    const hasTimeline = ch.timeline !== undefined;
    playBtn.disabled = !hasTimeline;
    sliderRow.classList.toggle("is-hidden", !hasTimeline);
    updatePlayLabel();
    inputEditor?.relabel();
  }

  // ---- render loop ----------------------------------------------------------
  shell.start(() => {
    const now = performance.now();
    tour.update(now);
    if (tour.state === "idle") {
      // OrbitControls drives; report the live view so the next fly-to starts here.
      controls.autoRotate = !reduceMotion && now - lastUserMs > IDLE_ORBIT_MS;
      controls.update();
      tour.syncCurrent(
        [shell.camera.position.x, shell.camera.position.y, shell.camera.position.z],
        [controls.target.x, controls.target.y, controls.target.z],
      );
    }
    timeline.update(now);
    if (timeline.state === "playing" && chapters.get(current).timeline) {
      slider.value = String(timeline.stepIndex);
      sliderVal.textContent = String(timeline.stepIndex);
    }
    picker.update();
    scene.update(shell.camera);
  });

  // ---- dev hooks (scripts/hover-check.mjs) ----------------------------------
  installVizHooks(scene, shell);

  // ---- boot -----------------------------------------------------------------
  const startIdx = deps.initialChapterId ? chapters.indexOf(deps.initialChapterId) : -1;
  goToChapter(startIdx >= 0 ? startIdx : 0, true);

  const ready = labelsReady(...scene.labelObjects);

  function dispose(): void {
    inputEditor?.dispose();
    timeline.stop();
    tour.cancel();
    controls.dispose();
    picker.dispose();
    tooltip.dispose();
    scene.dispose();
    shell.dispose();
    root.remove();
    if (window.__viz?.owner === scene) window.__viz = undefined;
  }

  return {
    get chapterId(): string {
      return chapters.get(current).id;
    },
    goToChapterId(id: string): void {
      const idx = chapters.indexOf(id);
      if (idx >= 0) goToChapter(idx);
    },
    ready,
    dispose,
  };
}

/** Install the cell-projection debug hooks used by scripts/hover-check.mjs. */
function installVizHooks(scene: SceneController, shell: SceneShell): void {
  const v = new THREE.Vector3();
  window.__viz = {
    owner: scene,
    cellScreen(name, row, col) {
      const view = scene.views.get(name);
      if (!view || view.shape.length !== 2) return null;
      const pitch = view.layout.cellSize + view.layout.gap;
      const [ox, oy, oz] = view.layout.origin;
      v.set(ox + col * pitch, oy - row * pitch, oz).project(shell.camera);
      const rect = shell.renderer.domElement.getBoundingClientRect();
      return {
        x: rect.left + ((v.x + 1) / 2) * rect.width,
        y: rect.top + ((1 - v.y) / 2) * rect.height,
      };
    },
    cellValue(name, row, col) {
      const view = scene.views.get(name);
      if (!view || view.shape.length !== 2) return null;
      return view.lastValues[row * view.shape[1] + col];
    },
  };
}
