/**
 * Per-architecture visual identity: a signature accent colour and a small
 * mechanism glyph, shared by the landing cards, the comparison table, and the
 * arch-page chrome so a colour reads the same everywhere ("RWKV is violet").
 *
 * Glyphs are minimal inline SVG (rect / circle / line / one spline) that
 * inherit the accent via currentColor — no external icon dependency.
 */

export interface ArchVisual {
  /** Signature accent colour (hex). */
  accent: string;
  /** Inner SVG markup for the 24×24 mechanism glyph (uses currentColor). */
  glyph: string;
}

const SVG_OPEN =
  '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" ' +
  'stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">';

/** Wrap inner glyph markup in the standard 24×24 stroke SVG. */
function glyph(inner: string): string {
  return `${SVG_OPEN}${inner}</svg>`;
}

export const ARCH_VISUALS: Record<string, ArchVisual> = {
  // Selective SSM: a fixed-size state scanning forward.
  mamba: {
    accent: "#4cc9b0",
    glyph: glyph('<rect x="3" y="8" width="7" height="8" rx="2"/><path d="M12 12h8"/><path d="M17 9l3 3-3 3"/>'),
  },
  // Causal attention: the lower-triangular mask region.
  gpt: {
    accent: "#7fb4ff",
    glyph: glyph('<path d="M5 5v14h14z" fill="currentColor" fill-opacity="0.22"/><path d="M5 5v14h14"/>'),
  },
  // Linear recurrence: a per-channel decay (shrinking bars).
  rwkv: {
    accent: "#c9a0ff",
    glyph: glyph('<path d="M4 20V8"/><path d="M9 20v-8"/><path d="M14 20v-5"/><path d="M19 20v-3"/>'),
  },
  // Routed experts: one token fans to a few, top-2 solid.
  moe: {
    accent: "#e0a850",
    glyph: glyph(
      '<circle cx="4" cy="12" r="2.2"/><circle cx="20" cy="5" r="2.2"/><circle cx="20" cy="19" r="2.2"/>' +
        '<circle cx="20" cy="12" r="1.5" stroke-opacity="0.35"/>' +
        '<path d="M6 11 18 6"/><path d="M6 13 18 18"/><path d="M6 12h12" stroke-opacity="0.35"/>',
    ),
  },
  // KAN: a learned spline function on the edge between two nodes.
  kan: {
    accent: "#7fd962",
    glyph: glyph('<circle cx="4" cy="12" r="2.2"/><circle cx="20" cy="12" r="2.2"/><path d="M6 12c3-7 7 7 12 0"/>'),
  },
  // Diffusion: a masked cell row (filled / hollow).
  diffusion: {
    accent: "#ff7be0",
    glyph: glyph(
      '<rect x="2.5" y="9" width="4" height="6" rx="1.2" fill="currentColor"/>' +
        '<rect x="8.5" y="9" width="4" height="6" rx="1.2"/>' +
        '<rect x="14.5" y="9" width="4" height="6" rx="1.2" fill="currentColor"/>' +
        '<rect x="20" y="9" width="3" height="6" rx="1.2"/>',
    ),
  },
};

/** Default accent for unknown ids (the neutral chrome blue). */
const DEFAULT_ACCENT = "#7fb4ff";

export function archAccent(id: string): string {
  return ARCH_VISUALS[id]?.accent ?? DEFAULT_ACCENT;
}

export function archGlyph(id: string): string {
  return ARCH_VISUALS[id]?.glyph ?? "";
}
