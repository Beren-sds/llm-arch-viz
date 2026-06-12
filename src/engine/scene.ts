/**
 * Scene shell: renderer + camera + render loop, the single place that
 * talks to the GL context. Everything else (TensorViews, scenes, tours)
 * adds objects to `scene` and registers per-frame work via `start(cb)`.
 *
 * WebGL2 is checked BEFORE the renderer is created; on failure a visible
 * `.gl-error` overlay is mounted in the container and an Error is thrown
 * so callers never proceed with a dead canvas.
 */

import * as THREE from "three";

/** Site background — keep in sync with `body` background in style.css. */
export const BACKGROUND_COLOR = 0x0b0e14;

/**
 * Frames-per-second over a sliding ~1 s window. Pure math, no DOM/RAF:
 * feed it timestamps, it returns the fps once per completed window and
 * null otherwise. The first call only opens the window (a timestamp by
 * itself is zero frames of evidence).
 */
export class FpsCounter {
  private readonly windowMs: number;
  private windowStart: number | null = null;
  private frames = 0;

  constructor(windowMs = 1000) {
    this.windowMs = windowMs;
  }

  /** Record a frame at `nowMs`; returns fps when a window completes, else null. */
  frame(nowMs: number): number | null {
    if (this.windowStart === null) {
      this.windowStart = nowMs;
      this.frames = 0;
      return null;
    }
    this.frames++;
    const elapsed = nowMs - this.windowStart;
    if (elapsed < this.windowMs) return null;
    const fps = (this.frames * 1000) / elapsed;
    this.windowStart = nowMs;
    this.frames = 0;
    return fps;
  }
}

export interface SceneShell {
  scene: THREE.Scene;
  camera: THREE.PerspectiveCamera;
  renderer: THREE.WebGLRenderer;
  /** Begin the render loop; `cb(dtSeconds)` runs before each render. */
  start(renderLoop: (dt: number) => void): void;
  /** Stop the loop, release GL resources, unmount canvas + overlays. */
  dispose(): void;
}

/**
 * Create the renderer/camera/scene triple inside `container`.
 * Throws (after mounting a visible `.gl-error` overlay) when WebGL2 is
 * unavailable. Resize is handled; in dev a tiny FPS readout is shown.
 */
export function createSceneShell(container: HTMLElement): SceneShell {
  assertWebGL2(container);

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(BACKGROUND_COLOR);

  const camera = new THREE.PerspectiveCamera(
    45,
    aspectOf(container),
    0.1,
    2000,
  );
  camera.position.set(0, 0, 10);

  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setSize(container.clientWidth, container.clientHeight);
  container.appendChild(renderer.domElement);

  // ResizeObserver instead of window `resize`: the container can change
  // size without the window doing so (panel toggles, scrollbar appearing,
  // layout shifts), and per-element observation is what we actually mean.
  const onResize = (): void => {
    camera.aspect = aspectOf(container);
    camera.updateProjectionMatrix();
    renderer.setSize(container.clientWidth, container.clientHeight);
  };
  const resizeObserver = new ResizeObserver(onResize);
  resizeObserver.observe(container);

  // Dev-only FPS readout: a fixed div updated once per second. No deps.
  let statsEl: HTMLDivElement | null = null;
  const fps = new FpsCounter();
  if (import.meta.env.DEV) {
    statsEl = document.createElement("div");
    statsEl.className = "fps-stats";
    statsEl.textContent = "— fps";
    container.appendChild(statsEl);
  }

  let last: number | null = null;
  const start = (renderLoop: (dt: number) => void): void => {
    renderer.setAnimationLoop((time: number) => {
      const dt = last === null ? 0 : (time - last) / 1000;
      last = time;
      renderLoop(dt);
      renderer.render(scene, camera);
      const f = fps.frame(time);
      if (f !== null && statsEl) statsEl.textContent = `${f.toFixed(0)} fps`;
    });
  };

  const dispose = (): void => {
    renderer.setAnimationLoop(null);
    resizeObserver.disconnect();
    renderer.dispose();
    // Browsers cap live WebGL contexts per page (~8-16); Task 22 recreates
    // shells per navigation, so explicitly lose the context instead of
    // waiting for GC to reclaim the detached canvas.
    renderer.forceContextLoss();
    renderer.domElement.remove();
    statsEl?.remove();
  };

  return { scene, camera, renderer, start, dispose };
}

function aspectOf(container: HTMLElement): number {
  // Guard 0-height containers (display:none during boot) — NaN aspect
  // poisons the projection matrix.
  return container.clientHeight > 0 ? container.clientWidth / container.clientHeight : 1;
}

/** Probe WebGL2; on failure mount the `.gl-error` overlay and throw. */
function assertWebGL2(container: HTMLElement): void {
  let ok = false;
  if (typeof WebGL2RenderingContext !== "undefined") {
    try {
      const probe = document.createElement("canvas");
      const ctx = probe.getContext("webgl2");
      ok = ctx !== null;
      // Release the probe context immediately — live contexts are capped
      // per page and this one exists only to answer "is WebGL2 there?".
      ctx?.getExtension("WEBGL_lose_context")?.loseContext();
    } catch {
      ok = false;
    }
  }
  if (ok) return;

  const overlay = document.createElement("div");
  overlay.className = "gl-error";
  const en = document.createElement("p");
  en.textContent =
    "This visualization requires WebGL2, which your browser does not support " +
    "or has disabled. Please use a recent Chrome, Firefox, Safari, or Edge " +
    "with hardware acceleration enabled.";
  const zh = document.createElement("p");
  zh.textContent =
    "本页面需要 WebGL2 支持，而你的浏览器不支持或已禁用它。" +
    "请使用较新版本的 Chrome / Firefox / Safari / Edge，并开启硬件加速。";
  overlay.append(en, zh);
  container.appendChild(overlay);
  throw new Error("WebGL2 is not available; .gl-error overlay mounted");
}
