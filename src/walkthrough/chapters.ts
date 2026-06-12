/**
 * Walkthrough chapters: an ordered, validated list of named camera stops
 * for one scene, plus the '#/scene/chapter' deep-link codec.
 *
 * All consistency rules are enforced at construction (duplicate ids,
 * untranslated narration keys, empty list), so a registry that exists is a
 * registry that's safe to drive the UI from — bad chapter data fails the
 * build/tests, not the viewer at runtime.
 */

import type { CameraKeyframe } from "../engine/cameraTour";
import type { I18n } from "../i18n/i18n";

export interface Chapter {
  /** Stable slug used in deep links ('#/mamba/ssm-scan'). */
  id: string;
  /** Where the camera flies when this chapter activates. */
  camera: CameraKeyframe;
  /** Object names/ids to highlight while the chapter is active. */
  highlights: string[];
  /** i18n key for the narration text; must exist in EVERY locale. */
  narrationKey: string;
  /** Per-chapter compute timeline — typed placeholder until Task 18. */
  timeline?: unknown;
}

export class ChapterRegistry {
  readonly scene: string;
  private readonly chapters: readonly Chapter[];
  private readonly idToIndex: Map<string, number>;

  /**
   * Validates at construction; throws on:
   * - empty chapter list
   * - duplicate chapter ids
   * - a narrationKey missing from ANY locale (en and zh must both have it —
   *   the locales stay in sync by construction)
   */
  constructor(scene: string, chapters: Chapter[], i18n: I18n) {
    if (chapters.length === 0) {
      throw new Error(`ChapterRegistry("${scene}"): empty chapter list`);
    }
    const idToIndex = new Map<string, number>();
    for (let i = 0; i < chapters.length; i++) {
      const { id, narrationKey } = chapters[i];
      if (idToIndex.has(id)) {
        throw new Error(`ChapterRegistry("${scene}"): duplicate chapter id "${id}"`);
      }
      idToIndex.set(id, i);
      const missing = i18n.missingLocales(narrationKey);
      if (missing.length > 0) {
        throw new Error(
          `ChapterRegistry("${scene}"): chapter "${id}" narrationKey ` +
            `"${narrationKey}" is missing from locale(s): ${missing.join(", ")}`,
        );
      }
    }
    this.scene = scene;
    this.chapters = [...chapters]; // shield against caller mutation
    this.idToIndex = idToIndex;
  }

  get count(): number {
    return this.chapters.length;
  }

  /** Chapter at position `idx`; throws RangeError when out of range. */
  get(idx: number): Chapter {
    if (!Number.isInteger(idx) || idx < 0 || idx >= this.chapters.length) {
      throw new RangeError(
        `ChapterRegistry("${this.scene}"): index ${idx} out of range [0, ${this.chapters.length})`,
      );
    }
    return this.chapters[idx];
  }

  /** Chapter with the given id, or undefined. */
  byId(id: string): Chapter | undefined {
    const idx = this.idToIndex.get(id);
    return idx === undefined ? undefined : this.chapters[idx];
  }

  /** Position of the chapter with the given id, or -1. */
  indexOf(id: string): number {
    return this.idToIndex.get(id) ?? -1;
  }
}

/** One slug segment: letters, digits, '_', '-' (no spaces, no '/'). */
const HASH_RE = /^#\/([\w-]+)(?:\/([\w-]+))?$/;

/**
 * Parse a '#/scene' or '#/scene/chapter' deep link. Anything else —
 * empty string, bare '#', trailing slash, extra segments, characters
 * outside [A-Za-z0-9_-] — returns null. Inverse of `toHash`.
 */
export function parseHash(hash: string): { scene: string; chapterId?: string } | null {
  const m = HASH_RE.exec(hash);
  if (m === null) return null;
  const [, scene, chapterId] = m;
  return chapterId === undefined ? { scene } : { scene, chapterId };
}

/** Build the '#/scene[/chapter]' deep link. Inverse of `parseHash`. */
export function toHash(scene: string, chapterId?: string): string {
  return chapterId === undefined ? `#/${scene}` : `#/${scene}/${chapterId}`;
}
