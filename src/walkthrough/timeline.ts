/**
 * Timeline animation primitives: a declarative sequence of steps
 * (highlight / focus / stepToken / pulse / wait) executed against a
 * SceneBinding, which is the timeline's ONLY handle to the world — the
 * player never touches three.js, models, or the picker directly, so the
 * unit tests run on a mock binding with zero GL context.
 *
 * Like TourPlayer, time is injected via `update(nowMs)` (pass
 * performance.now()); there is no Date.now()/RAF inside, which keeps the
 * player fully deterministic under test. Entering a step triggers its
 * effect exactly ONCE; visual steps undo themselves on exit.
 */

import type { T } from "../compute/tensor";

/**
 * What a scene exposes to its timelines. The scene owns the name → view
 * mapping, model re-execution, and the picker; the timeline only issues
 * intents through these channels.
 *
 * OWNERSHIP: while a TimelinePlayer is playing, the binding owns
 * highlight/dim EXCLUSIVELY — no other code may call setHighlight/setDim
 * on the same views, or the player's exit/stop cleanup will fight it.
 */
export interface SceneBinding {
  /**
   * Re-run the model on the prefix tokens[0..uptoToken] — INCLUSIVE:
   * uptoToken is the index of the LAST token fed, so uptoToken=0 runs a
   * 1-token prefix. Returns activations by name.
   */
  runForward(uptoToken: number): Map<string, T>;
  /** Push activations into the views (and picker.requestRepick()). */
  applyActivations(acts: Map<string, T>): void;
  /** Toggle highlight on the named objects. MUST be idempotent. */
  setHighlight(names: string[], on: boolean): void;
  /** Dim everything NOT in the list; null = undim all. */
  setDim(namesNotIn: string[] | null): void;
  /** Visual flow pulse from one object to another (may be a no-op). */
  pulse(from: string, to: string): void;
}

export type TimelineStep =
  /** Highlight `names` for the duration (off again at exit). */
  | { kind: "highlight"; names: string[]; durationMs: number }
  /** Dim everything but `names` + highlight them (both restored at exit). */
  | { kind: "focus"; names: string[]; durationMs: number }
  /** runForward(token) + applyActivations at entry; then hold. */
  | { kind: "stepToken"; token: number; durationMs: number }
  /** Fire a flow pulse at entry; then hold. */
  | { kind: "pulse"; from: string; to: string; durationMs: number }
  /** Hold with no effect. */
  | { kind: "wait"; durationMs: number };

export interface TimelineSpec {
  steps: TimelineStep[];
  /** Wrap back to step 0 after the last step instead of completing. */
  loop?: boolean;
}

export type TimelinePlayerState = "idle" | "playing" | "paused";

/**
 * Steps a duration of 0 (or less) would stall the no-progress guard in
 * update(); they are treated as this many ms instead (same clamp as
 * TourPlayer's durationMs).
 */
const MIN_STEP_MS = 1;

/**
 * Sequential, externally clocked step player.
 *
 *   const player = new TimelinePlayer(binding);
 *   player.load(spec);
 *   player.play();                    // enters step 0 NOW (effects fire)
 *   // per frame:
 *   player.update(performance.now()); // first call opens the clock
 *
 * play() applies step 0's effect immediately; durations start counting at
 * the first update() after play()/resume. pause() freezes mid-step and a
 * later play() resumes with the REMAINING duration (step-elapsed time is
 * tracked, never absolute timestamps). stop() resets to the start and
 * clears every highlight made during the run plus any dim.
 */
export class TimelinePlayer {
  /**
   * Fired once on entering a step, before its effect is applied. Calling
   * stop() from here cancels the step's effect; pause() lets it apply.
   */
  onStepStart?: (step: TimelineStep, index: number) => void;
  /** Fired once when the last step finishes (never while looping). */
  onComplete?: () => void;

  private readonly binding: SceneBinding;
  private spec: TimelineSpec | null = null;
  private phase: TimelinePlayerState = "idle";
  private index = 0;
  /** Time spent inside the current step (survives pause/resume). */
  private stepElapsedMs = 0;
  /** Last update() timestamp; null = clock closed (just played/resumed). */
  private lastNowMs: number | null = null;
  /** Every name highlighted since play() — stop() switches them all off. */
  private readonly lit = new Set<string>();

  constructor(binding: SceneBinding) {
    this.binding = binding;
  }

  get state(): TimelinePlayerState {
    return this.phase;
  }

  /** Index of the current step (0 while idle). */
  get stepIndex(): number {
    return this.index;
  }

  /**
   * Load a spec. The steps ARRAY is shallow-copied — swapping/reordering
   * the caller's array afterwards is safe, but the step OBJECTS are
   * shared: don't mutate them after load(). Loading while a run is
   * active behaves like stop(): visuals are cleared and the player
   * returns to 'idle' at step 0.
   */
  load(spec: TimelineSpec): void {
    this.stop();
    this.spec = { steps: [...spec.steps], loop: spec.loop ?? false };
  }

  /**
   * From 'idle': start at step 0, applying its effect immediately (the
   * clock opens on the next update()). From 'paused': resume the current
   * step with its remaining duration. While 'playing' or with no spec
   * loaded: no-op. An empty spec completes immediately.
   */
  play(): void {
    if (this.phase === "playing" || this.spec === null) return;
    if (this.phase === "paused") {
      this.phase = "playing";
      this.lastNowMs = null; // reopen the clock; the pause gap must not count
      return;
    }
    if (this.spec.steps.length === 0) {
      this.onComplete?.();
      return;
    }
    this.phase = "playing";
    this.index = 0;
    this.stepElapsedMs = 0;
    this.lastNowMs = null;
    this.enterStep(0);
  }

  /** Freeze mid-step (step-elapsed time is kept). No-op unless playing. */
  pause(): void {
    if (this.phase !== "playing") return;
    this.phase = "paused";
    this.lastNowMs = null;
  }

  /**
   * Reset to the start: undim everything and switch off every name this
   * run highlighted, return to 'idle' at step 0. No-op while idle (a
   * player that never ran has nothing to clean up).
   */
  stop(): void {
    if (this.phase === "idle") return;
    if (this.lit.size > 0) this.binding.setHighlight([...this.lit], false);
    this.binding.setDim(null);
    this.reset();
  }

  /**
   * Advance to `nowMs` (pass performance.now()). The first update after
   * play()/resume opens the clock; later ones add the delta to the current
   * step and cascade through as many step boundaries as the delta covers
   * (each exit/entry effect fires exactly once). No-op unless playing.
   */
  update(nowMs: number): void {
    if (this.phase !== "playing" || this.spec === null) return;
    if (this.lastNowMs === null) {
      this.lastNowMs = nowMs;
      return;
    }
    const delta = nowMs - this.lastNowMs;
    this.lastNowMs = nowMs;
    if (delta <= 0) return; // non-monotonic clock: hold position
    this.stepElapsedMs += delta;

    const { steps, loop } = this.spec;
    // Background-tab catch-up: when looping and the delta spans MULTIPLE
    // full passes of the spec, reduce the surplus modulo the spec's total
    // duration BEFORE cascading — each step then fires at most once for
    // finishing the current pass plus once in the final partial pass,
    // instead of replaying every skipped loop N times.
    if (loop) {
      let total = 0;
      for (const s of steps) total += this.durationOf(s);
      let rest = 0; // full durations of the current step through the last
      for (let i = this.index; i < steps.length; i++) rest += this.durationOf(steps[i]);
      if (this.stepElapsedMs >= rest + total) {
        this.stepElapsedMs = rest + ((this.stepElapsedMs - rest) % total);
      }
    }
    // Cascade across boundaries. The loop guard stops the walk when an
    // onStepStart callback called stop()/pause(); the entering step's own
    // effect is guarded inside enterStep (it re-checks phase after the
    // callback and skips the effect if stop() made the player idle).
    while (
      this.phase === "playing" &&
      this.stepElapsedMs >= this.durationOf(steps[this.index])
    ) {
      this.stepElapsedMs -= this.durationOf(steps[this.index]);
      this.exitStep(this.index);
      const next = this.index + 1;
      if (next < steps.length) {
        this.index = next;
        this.enterStep(next);
      } else if (loop) {
        this.index = 0;
        this.enterStep(0);
      } else {
        this.reset();
        this.onComplete?.();
        return;
      }
    }
  }

  private durationOf(step: TimelineStep): number {
    return Math.max(MIN_STEP_MS, step.durationMs);
  }

  /** Back to a clean 'idle' at step 0 (no binding calls). */
  private reset(): void {
    this.phase = "idle";
    this.index = 0;
    this.stepElapsedMs = 0;
    this.lastNowMs = null;
    this.lit.clear();
  }

  private enterStep(index: number): void {
    if (this.spec === null) return;
    const step = this.spec.steps[index];
    this.onStepStart?.(step, index);
    // The callback may have called stop() (which already cleaned up all
    // visuals): applying the effect now would leave a highlight/dim that
    // no later stop() can clear (idle guard). pause() keeps the effect —
    // the step is still current and resume continues it.
    if (this.phase === "idle") return;
    switch (step.kind) {
      case "highlight":
        this.binding.setHighlight(step.names, true);
        for (const n of step.names) this.lit.add(n);
        break;
      case "focus":
        this.binding.setDim(step.names);
        this.binding.setHighlight(step.names, true);
        for (const n of step.names) this.lit.add(n);
        break;
      case "stepToken":
        this.binding.applyActivations(this.binding.runForward(step.token));
        break;
      case "pulse":
        this.binding.pulse(step.from, step.to);
        break;
      case "wait":
        break;
    }
  }

  private exitStep(index: number): void {
    if (this.spec === null) return;
    const step = this.spec.steps[index];
    switch (step.kind) {
      case "highlight":
        this.binding.setHighlight(step.names, false);
        break;
      case "focus":
        this.binding.setHighlight(step.names, false);
        this.binding.setDim(null);
        break;
      case "stepToken":
      case "pulse":
      case "wait":
        break;
    }
  }
}
