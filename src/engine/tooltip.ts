/**
 * Hover tooltip for tensor cells: a fixed-position panel showing the
 * tensor name, the cell's indices, its value, and (optionally) the
 * formula that produced the tensor. Styled via `.tensor-tooltip` in
 * style.css; pointer-events: none so it never steals the hover.
 */

import type { PickResult } from "./picking";

/** Gap in px between the cursor and the tooltip's preferred corner. */
export const TOOLTIP_OFFSET = 12;

/**
 * Position a w×h box near cursor (x, y) inside a vw×vh viewport:
 * prefer below-right of the cursor by `offset`; flip to the other side
 * of the cursor per-axis when that overflows; finally clamp into the
 * viewport (pinning to 0 when the box is larger than the viewport).
 */
export function clampToViewport(
  x: number,
  y: number,
  w: number,
  h: number,
  vw: number,
  vh: number,
  offset: number,
): { x: number; y: number } {
  let px = x + offset;
  if (px + w > vw) px = x - offset - w;
  px = Math.min(Math.max(px, 0), Math.max(0, vw - w));

  let py = y + offset;
  if (py + h > vh) py = y - offset - h;
  py = Math.min(Math.max(py, 0), Math.max(0, vh - h));

  return { x: px, y: py };
}

/** Value text: toPrecision(4); special-cased -∞ (masked) and NaN. */
export function formatValue(v: number): string {
  if (Number.isNaN(v)) return "NaN";
  if (v === -Infinity) return "−∞ (masked)";
  if (v === Infinity) return "∞";
  return v.toPrecision(4);
}

export class Tooltip {
  private readonly el: HTMLDivElement;
  private readonly nameEl: HTMLDivElement;
  private readonly indicesEl: HTMLSpanElement;
  private readonly valueEl: HTMLSpanElement;
  private readonly formulaEl: HTMLDivElement;

  constructor(container: HTMLElement) {
    this.el = document.createElement("div");
    this.el.className = "tensor-tooltip";

    this.nameEl = document.createElement("div");
    this.nameEl.className = "tensor-tooltip-name";

    const cell = document.createElement("div");
    cell.className = "tensor-tooltip-cell";
    this.indicesEl = document.createElement("span");
    this.indicesEl.className = "tensor-tooltip-indices";
    this.valueEl = document.createElement("span");
    this.valueEl.className = "tensor-tooltip-value";
    cell.append(this.indicesEl, document.createTextNode(" = "), this.valueEl);

    this.formulaEl = document.createElement("div");
    this.formulaEl.className = "tensor-tooltip-formula";
    this.formulaEl.hidden = true;

    this.el.append(this.nameEl, cell, this.formulaEl);
    container.appendChild(this.el);
  }

  show(result: PickResult, clientX: number, clientY: number): void {
    this.nameEl.textContent = result.name;
    this.indicesEl.textContent = `[${result.indices.join(", ")}]`;
    this.valueEl.textContent = formatValue(result.value);
    this.valueEl.classList.toggle("is-nan", Number.isNaN(result.value));

    if (result.formula !== undefined) {
      this.formulaEl.textContent = result.formula;
      this.formulaEl.hidden = false;
    } else {
      this.formulaEl.hidden = true;
    }

    // Make it visible BEFORE measuring: display:none reports 0x0.
    this.el.classList.add("visible");
    const { x, y } = clampToViewport(
      clientX,
      clientY,
      this.el.offsetWidth,
      this.el.offsetHeight,
      window.innerWidth,
      window.innerHeight,
      TOOLTIP_OFFSET,
    );
    this.el.style.left = `${x}px`;
    this.el.style.top = `${y}px`;
  }

  hide(): void {
    this.el.classList.remove("visible");
  }

  dispose(): void {
    this.el.remove();
  }
}
