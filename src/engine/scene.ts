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
import { EffectComposer } from "three/examples/jsm/postprocessing/EffectComposer.js";
import { RenderPass } from "three/examples/jsm/postprocessing/RenderPass.js";
import { UnrealBloomPass } from "three/examples/jsm/postprocessing/UnrealBloomPass.js";
import { ShaderPass } from "three/examples/jsm/postprocessing/ShaderPass.js";
import { OutputPass } from "three/examples/jsm/postprocessing/OutputPass.js";

/** Site background — keep in sync with `body` background in style.css. */
export const BACKGROUND_COLOR = 0x0b0e14;

/**
 * Render layer for objects that should bloom (the flow spine). Selective
 * bloom blooms ONLY this layer, so the bright data cells never blow out.
 */
export const BLOOM_LAYER = 1;

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

  // Opaque clear to the atmosphere-centre tone; the CSS vignette over the
  // canvas darkens the edges for a radial feel, and the bloom pass needs an
  // opaque target to composite cleanly.
  const scene = new THREE.Scene();

  const camera = new THREE.PerspectiveCamera(
    45,
    aspectOf(container),
    0.1,
    2000,
  );
  camera.position.set(0, 0, 10);

  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  // Pre-linearized clear: the composer renders into a linear target and the
  // OutputPass encodes linear→sRGB, so a raw dark clear would be lifted to a
  // milky haze. linear(#0b0e14) ≈ #010102 round-trips back to #0b0e14.
  renderer.setClearColor(0x010102, 1);
  renderer.setSize(container.clientWidth, container.clientHeight);
  container.appendChild(renderer.domElement);

  // Selective bloom: only objects on BLOOM_LAYER (the flow spine) glow. The
  // bloom composer renders the scene with every non-bloom mesh blacked out,
  // blooms that, and the final composer adds the result over the full scene.
  // (Full-scene bloom blew the bright data cells into white blobs.)
  const bloomLayer = new THREE.Layers();
  bloomLayer.set(BLOOM_LAYER);
  const darkMaterial = new THREE.MeshBasicMaterial({ color: 0x000000 });
  const savedMaterials = new Map<string, THREE.Material | THREE.Material[]>();
  const darken = (obj: THREE.Object3D): void => {
    const mesh = obj as THREE.Mesh;
    if (mesh.isMesh && !bloomLayer.test(obj.layers)) {
      savedMaterials.set(obj.uuid, mesh.material);
      mesh.material = darkMaterial;
    }
  };
  const restore = (obj: THREE.Object3D): void => {
    const saved = savedMaterials.get(obj.uuid);
    if (saved) {
      (obj as THREE.Mesh).material = saved;
      savedMaterials.delete(obj.uuid);
    }
  };

  const renderPass = new RenderPass(scene, camera);
  const bloomPass = new UnrealBloomPass(
    new THREE.Vector2(container.clientWidth, container.clientHeight),
    0.5, // strength — tight glow on the spine, not a scene-flooding haze
    0.2, // radius
    0, // threshold 0 — selection is by layer, not luminance
  );
  const bloomComposer = new EffectComposer(renderer);
  bloomComposer.renderToScreen = false;
  bloomComposer.addPass(renderPass);
  bloomComposer.addPass(bloomPass);

  const mixPass = new ShaderPass(
    new THREE.ShaderMaterial({
      uniforms: {
        baseTexture: { value: null },
        bloomTexture: { value: bloomComposer.renderTarget2.texture },
      },
      vertexShader: `varying vec2 vUv; void main(){ vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }`,
      fragmentShader: `uniform sampler2D baseTexture; uniform sampler2D bloomTexture; varying vec2 vUv; void main(){ gl_FragColor = texture2D(baseTexture, vUv) + texture2D(bloomTexture, vUv); }`,
    }),
    "baseTexture",
  );
  mixPass.needsSwap = true;
  const finalComposer = new EffectComposer(renderer);
  finalComposer.addPass(renderPass);
  finalComposer.addPass(mixPass);
  finalComposer.addPass(new OutputPass());

  // ResizeObserver instead of window `resize`: the container can change
  // size without the window doing so (panel toggles, scrollbar appearing,
  // layout shifts), and per-element observation is what we actually mean.
  const onResize = (): void => {
    camera.aspect = aspectOf(container);
    camera.updateProjectionMatrix();
    const w = container.clientWidth;
    const h = container.clientHeight;
    renderer.setSize(w, h);
    bloomComposer.setSize(w, h);
    finalComposer.setSize(w, h);
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
      // Selective bloom: black out non-bloom meshes, bloom the spine, restore,
      // then render the full scene with the bloom added.
      scene.traverse(darken);
      bloomComposer.render();
      scene.traverse(restore);
      finalComposer.render();
      const f = fps.frame(time);
      if (f !== null && statsEl) statsEl.textContent = `${f.toFixed(0)} fps`;
    });
  };

  const dispose = (): void => {
    renderer.setAnimationLoop(null);
    resizeObserver.disconnect();
    bloomComposer.dispose();
    finalComposer.dispose();
    darkMaterial.dispose();
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
