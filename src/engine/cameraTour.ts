/**
 * Keyframed camera tour: eased fly-to between camera keyframes.
 *
 * Deliberately decoupled from three.js — the only output channel is the
 * `applyView(pos, target)` callback, so the unit tests never touch a GL
 * context and the caller decides how a view lands (camera.position +
 * OrbitControls.target, usually). Time is injected via `update(nowMs)`
 * (pass `performance.now()`); there is no Date.now()/RAF inside, which
 * keeps the player fully deterministic under test.
 */

export interface CameraKeyframe {
  pos: [number, number, number];
  target: [number, number, number];
}

/**
 * Smoothstep ease in/out: 3t² − 2t³, clamped to [0,1].
 * cubicEase(0)=0, cubicEase(0.5)=0.5, cubicEase(1)=1; C¹ at both ends.
 */
export function cubicEase(t: number): number {
  const c = t < 0 ? 0 : t > 1 ? 1 : t;
  return c * c * (3 - 2 * c);
}

/**
 * Componentwise lerp between two keyframes with `cubicEase` applied to t.
 * Endpoints are exact (t≤0 → a, t≥1 → b, no float drift). Pass `out` to
 * avoid allocating in per-frame code; it is written to and returned.
 */
export function interpolateView(
  a: CameraKeyframe,
  b: CameraKeyframe,
  t: number,
  out?: CameraKeyframe,
): CameraKeyframe {
  const o = out ?? { pos: [0, 0, 0], target: [0, 0, 0] };
  // Exact endpoints: a + (b-a)*1 can land off b by an ulp in float math.
  const src = t <= 0 ? a : t >= 1 ? b : null;
  if (src !== null) {
    for (let i = 0; i < 3; i++) {
      o.pos[i] = src.pos[i];
      o.target[i] = src.target[i];
    }
    return o;
  }
  const e = cubicEase(t);
  for (let i = 0; i < 3; i++) {
    o.pos[i] = a.pos[i] + (b.pos[i] - a.pos[i]) * e;
    o.target[i] = a.target[i] + (b.target[i] - a.target[i]) * e;
  }
  return o;
}

export interface TourPlayerOptions {
  /**
   * The single output channel: called with the view to show. The arrays
   * are reused across calls — copy them, don't keep references.
   */
  applyView: (
    pos: readonly [number, number, number],
    target: readonly [number, number, number],
  ) => void;
  /** Fly-to duration in milliseconds. Default 1200. */
  durationMs?: number;
}

export interface GoToOptions {
  /** Snap to the keyframe immediately instead of animating. */
  instant?: boolean;
  /**
   * Fired once when THIS goTo's animation completes. Not fired when the
   * flight is superseded by a later goTo (retarget) or by cancel().
   */
  onArrive?: () => void;
}

function copyInto(dst: CameraKeyframe, src: CameraKeyframe): void {
  for (let i = 0; i < 3; i++) {
    dst.pos[i] = src.pos[i];
    dst.target[i] = src.target[i];
  }
}

function zeroKf(): CameraKeyframe {
  return { pos: [0, 0, 0], target: [0, 0, 0] };
}

/**
 * Drives eased fly-to animations between keyframes.
 *
 *   const player = new TourPlayer({ applyView: (p, t) => { ... } });
 *   player.goTo(kf);            // animates from the current view
 *   // per frame:
 *   player.update(performance.now());
 *
 * Retargeting mid-flight is smooth: a goTo during an animation restarts
 * from the currently interpolated view, never snapping.
 */
export class TourPlayer {
  state: "idle" | "animating" = "idle";

  private readonly applyView: TourPlayerOptions["applyView"];
  private readonly durationMs: number;

  /** Last view handed to applyView; null until the first goTo. */
  private current: CameraKeyframe | null = null;
  private readonly from: CameraKeyframe = zeroKf();
  private readonly to: CameraKeyframe = zeroKf();
  /** Scratch for interpolateView — reused every frame, never exposed. */
  private readonly scratch: CameraKeyframe = zeroKf();
  /** Set by the first update() after goTo; elapsed counts from there. */
  private startMs: number | null = null;
  private onArrive: (() => void) | null = null;

  constructor(opts: TourPlayerOptions) {
    this.applyView = opts.applyView;
    this.durationMs = opts.durationMs ?? 1200;
  }

  /**
   * Fly to `kf` over durationMs from the current view. The first call
   * (no prior view) and `instant: true` snap immediately. A goTo during
   * an active flight retargets smoothly from the interpolated view; the
   * superseded goTo's onArrive is dropped.
   */
  goTo(kf: CameraKeyframe, opts?: GoToOptions): void {
    if (this.current === null || opts?.instant) {
      // Instant path: apply now, complete now.
      this.state = "idle";
      this.startMs = null;
      this.onArrive = null;
      this.apply(kf);
      opts?.onArrive?.();
      return;
    }
    // Animate from wherever we are right now — for a mid-flight retarget
    // `current` IS the interpolated view, so there is no snap.
    copyInto(this.from, this.current);
    copyInto(this.to, kf); // copy: caller may mutate kf afterwards
    this.onArrive = opts?.onArrive ?? null;
    this.startMs = null; // clock opens on the next update()
    this.state = "animating";
  }

  /**
   * Advance the animation to `nowMs` (pass performance.now()). The first
   * update after a goTo opens the clock; arrival fires onArrive once and
   * returns the player to 'idle'. No-op while idle.
   */
  update(nowMs: number): void {
    if (this.state !== "animating") return;
    if (this.startMs === null) {
      this.startMs = nowMs;
      this.apply(this.from);
      return;
    }
    const t = (nowMs - this.startMs) / this.durationMs;
    if (t >= 1) {
      this.apply(this.to);
      this.state = "idle";
      this.startMs = null;
      const cb = this.onArrive;
      this.onArrive = null;
      cb?.();
      return;
    }
    this.apply(interpolateView(this.from, this.to, t, this.scratch));
  }

  /** Stop animating, staying at the current interpolated view. onArrive is dropped. */
  cancel(): void {
    this.state = "idle";
    this.startMs = null;
    this.onArrive = null;
  }

  private apply(view: CameraKeyframe): void {
    if (this.current === null) this.current = zeroKf();
    copyInto(this.current, view);
    this.applyView(this.current.pos, this.current.target);
  }
}
