/**
 * Live input editor: a compact token row that makes the selective-copying
 * input editable. The four data tokens (the non-noise context slots) are
 * click-to-cycle buttons; the answer tail mirrors them automatically so the
 * task stays well-formed. "Randomize" draws a fresh valid instance (new
 * positions + values, exactly like the training generator); "Reset" restores
 * the canonical example. Every change calls `onChange` with the full
 * sequence, which the page feeds to scene.setTokens for a live re-forward.
 *
 * The helpers (randomInstance / dataPositionsOf / seqLenOf) are pure and
 * unit-tested; randomness flows through an injectable `rand` so tests are
 * deterministic.
 */

import type { I18n } from "../i18n/i18n";
import { button, el } from "./dom";

/** Selective-copying task shape (mirrors training/config.yaml `task`). */
export interface TaskShape {
  vocabSize: number;
  nData: number;
  contextLen: number;
  /** Allowed data-token values (e.g. [1..8]); noise/go excluded. */
  dataIds: number[];
  noiseId: number;
  goId: number;
}

export interface InputEditor {
  /** The panel element to mount. */
  readonly el: HTMLElement;
  /** Current full input sequence (a copy). */
  readonly tokens: number[];
  /** Re-apply i18n text + chip titles (call on locale change). */
  relabel(): void;
  /** Restore the canonical example and fire onChange. */
  reset(): void;
  dispose(): void;
}

/** Full sequence length implied by a task shape. */
export function seqLenOf(task: TaskShape): number {
  return task.contextLen + 1 + task.nData;
}

/** Data-token positions in a sequence: the non-noise context slots, in order. */
export function dataPositionsOf(task: TaskShape, tokens: number[]): number[] {
  const out: number[] = [];
  for (let i = 0; i < task.contextLen; i++) if (tokens[i] !== task.noiseId) out.push(i);
  return out;
}

/** Mirror the answer tail onto the current data values (the copy invariant). */
export function syncAnswerTail(task: TaskShape, tokens: number[]): void {
  const pos = dataPositionsOf(task, tokens);
  for (let k = 0; k < task.nData; k++) tokens[task.contextLen + 1 + k] = tokens[pos[k]];
}

/**
 * Build a fresh valid instance: nData distinct sorted context positions, data
 * values drawn with replacement, GO marker, mirrored answer tail.
 */
export function randomInstance(task: TaskShape, rand: () => number = Math.random): number[] {
  const tokens = new Array<number>(seqLenOf(task)).fill(task.noiseId);
  const idx = Array.from({ length: task.contextLen }, (_, i) => i);
  for (let i = idx.length - 1; i > 0; i--) {
    const j = Math.floor(rand() * (i + 1));
    [idx[i], idx[j]] = [idx[j], idx[i]];
  }
  const pos = idx.slice(0, task.nData).sort((a, b) => a - b);
  for (const p of pos) tokens[p] = task.dataIds[Math.floor(rand() * task.dataIds.length)];
  tokens[task.contextLen] = task.goId;
  syncAnswerTail(task, tokens);
  return tokens;
}

export function createInputEditor(deps: {
  task: TaskShape;
  initial: number[];
  i18n: I18n;
  onChange: (tokens: number[]) => void;
  rand?: () => number;
}): InputEditor {
  const { task, i18n, onChange } = deps;
  const rand = deps.rand ?? Math.random;
  const initial = deps.initial.slice();
  let tokens = deps.initial.slice();

  const root = el("div", "viz-input");
  const label = el("span", "viz-input-label");
  const row = el("div", "viz-input-row");
  const randBtn = button("viz-input-btn");
  const resetBtn = button("viz-input-btn");
  randBtn.addEventListener("click", () => {
    tokens = randomInstance(task, rand);
    commit();
  });
  resetBtn.addEventListener("click", () => {
    tokens = initial.slice();
    commit();
  });
  root.append(label, row, randBtn, resetBtn);

  function nextDataId(v: number): number {
    const i = task.dataIds.indexOf(v);
    return task.dataIds[(i + 1) % task.dataIds.length];
  }

  function renderRow(): void {
    row.replaceChildren();
    const dpos = new Set(dataPositionsOf(task, tokens));
    const T = seqLenOf(task);
    for (let p = 0; p < T; p++) {
      if (p === task.contextLen) {
        const go = el("span", "viz-tok is-go tok-gap");
        go.textContent = i18n.t("ui.input.go");
        row.appendChild(go);
        continue;
      }
      if (p > task.contextLen) {
        const a = el("span", "viz-tok is-ans");
        if (p === task.contextLen + 1) a.classList.add("tok-gap");
        a.textContent = String(tokens[p]);
        row.appendChild(a);
        continue;
      }
      if (dpos.has(p)) {
        const b = button("viz-tok is-data");
        b.textContent = String(tokens[p]);
        b.title = i18n.t("ui.input.hint");
        const pos = p;
        b.addEventListener("click", () => {
          tokens[pos] = nextDataId(tokens[pos]);
          syncAnswerTail(task, tokens);
          commit();
        });
        row.appendChild(b);
      } else {
        const n = el("span", "viz-tok is-noise");
        n.textContent = "·";
        row.appendChild(n);
      }
    }
  }

  function commit(): void {
    renderRow();
    onChange(tokens.slice());
  }

  function relabel(): void {
    label.textContent = i18n.t("ui.input.label");
    randBtn.textContent = i18n.t("ui.input.randomize");
    resetBtn.textContent = i18n.t("ui.input.reset");
    renderRow();
  }

  relabel();

  return {
    el: root,
    get tokens(): number[] {
      return tokens.slice();
    },
    relabel,
    reset(): void {
      tokens = initial.slice();
      commit();
    },
    dispose(): void {
      root.remove();
    },
  };
}
