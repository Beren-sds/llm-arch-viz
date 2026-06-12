/**
 * Minimal ambient typing for the troika-three-text surface labels.ts uses.
 *
 * troika-three-text ships no TypeScript types and @types coverage does not
 * exist. Only labels.ts may import it (the rest of the engine goes through
 * the labels API), so this shim stays scoped to that surface. If more of
 * troika is ever needed, grow this declaration deliberately — property by
 * property — rather than reaching for `any`.
 */
declare module "troika-three-text" {
  import { BufferGeometry, Material, Mesh, Object3DEventMap } from "three";

  interface TextEventMap extends Object3DEventMap {
    syncstart: object;
    synccomplete: object;
  }

  /**
   * SDF text mesh. Set properties, then call sync(); glyph generation and
   * font loading are ASYNC.
   *
   * GOTCHA (verified against troika 0.52.4 Text.js): sync(callback) DROPS
   * the callback silently unless a NEW sync is needed at call time — it is
   * not a general "when ready" hook. Wait on the 'synccomplete' event (or a
   * non-null textRenderInfo) instead; see labelsReady() in labels.ts.
   */
  export class Text extends Mesh<BufferGeometry, Material | Material[], TextEventMap> {
    text: string;
    /** World units (cap height-ish), not px. */
    fontSize: number;
    /** Fill color (hex number or CSS string). */
    color: string | number;
    anchorX: number | "left" | "center" | "right";
    anchorY: number | "top" | "top-baseline" | "middle" | "bottom-baseline" | "bottom";
    /** Font URL; null = troika's built-in default (Noto-based). */
    font: string | null;
    /** SDF texture size per glyph; null = troika default (64). */
    sdfGlyphSize: number | null;
    /** Non-null once the FIRST typeset finished — the mesh is drawable. */
    readonly textRenderInfo: object | null;
    /**
     * Kick off async typesetting. The callback only fires when this call
     * STARTS or QUEUES a sync (see class doc) — prefer 'synccomplete'.
     */
    sync(callback?: () => void): void;
    /** Release glyph geometry + derived material. */
    dispose(): void;
  }
}
