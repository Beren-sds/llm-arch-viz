// @vitest-environment happy-dom

import { describe, expect, it, vi } from "vitest";
import { I18n, type Locale } from "../i18n/i18n";
import en from "../i18n/en.json";
import zh from "../i18n/zh.json";
import {
  createInputEditor,
  dataPositionsOf,
  randomInstance,
  seqLenOf,
  syncAnswerTail,
  type TaskShape,
} from "./inputEditor";

const TASK: TaskShape = {
  vocabSize: 16,
  nData: 4,
  contextLen: 16,
  dataIds: [1, 2, 3, 4, 5, 6, 7, 8],
  noiseId: 9,
  goId: 10,
};

// Canonical selective-copy example (data 4,1,4,6 at positions 1,5,7,9).
const CANON = [9, 4, 9, 9, 9, 1, 9, 4, 9, 6, 9, 9, 9, 9, 9, 9, 10, 4, 1, 4, 6];

/** Deterministic PRNG so randomInstance is reproducible in tests. */
function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function i18n(): I18n {
  return new I18n({ en, zh } satisfies Record<Locale, Record<string, string>>);
}

describe("inputEditor pure helpers", () => {
  it("seqLenOf = context + 1 + nData", () => {
    expect(seqLenOf(TASK)).toBe(21);
  });

  it("dataPositionsOf finds the non-noise context slots in order", () => {
    expect(dataPositionsOf(TASK, CANON)).toEqual([1, 5, 7, 9]);
  });

  it("syncAnswerTail mirrors data values onto the tail", () => {
    const t = CANON.slice();
    t[5] = 8; // change a data value in place
    syncAnswerTail(TASK, t);
    expect(t.slice(17)).toEqual([4, 8, 4, 6]);
  });

  it("randomInstance yields a well-formed instance", () => {
    for (let s = 1; s <= 25; s++) {
      const t = randomInstance(TASK, mulberry32(s));
      expect(t).toHaveLength(21);
      expect(t[TASK.contextLen]).toBe(TASK.goId); // GO marker
      const pos = dataPositionsOf(TASK, t);
      expect(pos).toHaveLength(TASK.nData);
      expect(new Set(pos).size).toBe(TASK.nData); // distinct positions
      for (const p of pos) expect(TASK.dataIds).toContain(t[p]); // valid values
      // answer tail mirrors the data values in context order
      expect(t.slice(17)).toEqual(pos.map((p) => t[p]));
    }
  });
});

describe("createInputEditor (happy-dom)", () => {
  it("renders one button per data token + GO + answer chips", () => {
    const ed = createInputEditor({ task: TASK, initial: CANON, i18n: i18n(), onChange: () => {} });
    expect(ed.el.querySelectorAll(".viz-tok.is-data")).toHaveLength(4);
    expect(ed.el.querySelectorAll(".viz-tok.is-ans")).toHaveLength(4);
    expect(ed.el.querySelectorAll(".viz-tok.is-go")).toHaveLength(1);
    expect(ed.el.querySelectorAll(".viz-tok.is-noise")).toHaveLength(12);
  });

  it("clicking a data chip cycles its value and re-syncs the tail", () => {
    const onChange = vi.fn();
    const ed = createInputEditor({ task: TASK, initial: CANON, i18n: i18n(), onChange });
    const first = ed.el.querySelector<HTMLButtonElement>(".viz-tok.is-data")!;
    expect(first.textContent).toBe("4");
    first.click();
    expect(onChange).toHaveBeenCalledTimes(1);
    const t = onChange.mock.calls[0][0] as number[];
    expect(t[1]).toBe(5); // 4 -> 5
    expect(t.slice(17)).toEqual([5, 1, 4, 6]); // tail tracks the change
  });

  it("cycle wraps 8 -> 1", () => {
    const start = CANON.slice();
    start[1] = 8;
    syncAnswerTail(TASK, start);
    const onChange = vi.fn();
    const ed = createInputEditor({ task: TASK, initial: start, i18n: i18n(), onChange });
    ed.el.querySelector<HTMLButtonElement>(".viz-tok.is-data")!.click();
    expect((onChange.mock.calls[0][0] as number[])[1]).toBe(1);
  });

  it("randomize fires onChange with a valid instance; reset restores canon", () => {
    const onChange = vi.fn();
    const ed = createInputEditor({
      task: TASK,
      initial: CANON,
      i18n: i18n(),
      onChange,
      rand: mulberry32(7),
    });
    const buttons = ed.el.querySelectorAll<HTMLButtonElement>(".viz-input-btn");
    buttons[0].click(); // randomize
    const rnd = onChange.mock.calls.at(-1)![0] as number[];
    expect(dataPositionsOf(TASK, rnd)).toHaveLength(4);
    buttons[1].click(); // reset
    expect(onChange.mock.calls.at(-1)![0]).toEqual(CANON);
  });

  it("reset() method restores and fires onChange", () => {
    const onChange = vi.fn();
    const ed = createInputEditor({ task: TASK, initial: CANON, i18n: i18n(), onChange, rand: mulberry32(3) });
    ed.el.querySelector<HTMLButtonElement>(".viz-input-btn")!.click(); // randomize
    ed.reset();
    expect(onChange.mock.calls.at(-1)![0]).toEqual(CANON);
    expect(ed.tokens).toEqual(CANON);
  });
});
